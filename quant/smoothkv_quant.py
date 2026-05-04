"""
SmoothKV: calibrated channel smoothing + per-token INT4 quantization.

Tier 1 (default):
  - K-side: diagonal scale sK applied post-RoPE (not fused since post-RoPE)
  - V-side: diagonal scale sV (fused into W_O in production; computed at runtime here)
  - No mu_V shift, no rotation (R_V = I)

Quantization: asymmetric min-max per-token INT4 with group_size along head_dim.
Stored as uint8 view for torch.cat compatibility; dequantized to FP16 for attention.
"""
import torch


def quantize_int4_pertoken(
    data: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token asymmetric INT4 quantization (groups along head_dim).

    Args:
        data: (B, nh, T, D) in fp16
        group_size: size of each group along D
    Returns:
        data_int4: (B, nh, T, D) in uint8 (values in 0..15, stored one-per-byte)
        scale:     (B, nh, T, D // group_size) in fp16
        zero:      (B, nh, T, D // group_size) in fp16
    """
    assert len(data.shape) == 4
    B, nh, T, D = data.shape
    assert D % group_size == 0
    num_groups = D // group_size

    # Reshape to expose groups: (B, nh, T, num_groups, group_size)
    data_grouped = data.view(B, nh, T, num_groups, group_size)

    # Per-group min/max
    mn = data_grouped.amin(dim=-1, keepdim=True)  # (B, nh, T, num_groups, 1)
    mx = data_grouped.amax(dim=-1, keepdim=True)
    scale = (mx - mn) / 15.0
    scale = scale.clamp(min=1e-12)

    # Quantize
    data_q = ((data_grouped - mn) / scale).round().clamp(0, 15).to(torch.uint8)

    return (
        data_q.view(B, nh, T, D),
        scale.squeeze(-1).to(torch.float16),
        mn.squeeze(-1).to(torch.float16),
    )


def dequantize_int4_pertoken(
    data_int4: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize (B, nh, T, D) uint8 INT4 tensor back to fp16."""
    B, nh, T, D = data_int4.shape
    num_groups = D // group_size

    # Expand scale and zero from (B, nh, T, num_groups) -> (B, nh, T, D)
    scale_exp = scale.unsqueeze(-1).expand(B, nh, T, num_groups, group_size).reshape(B, nh, T, D)
    zero_exp  = zero .unsqueeze(-1).expand(B, nh, T, num_groups, group_size).reshape(B, nh, T, D)

    return data_int4.to(torch.float16) * scale_exp + zero_exp


def apply_smooth_key(k_rot: torch.Tensor, s_K: torch.Tensor) -> torch.Tensor:
    """Apply K-side smoothing: K_smoothed = K_rot / s_K (channel-wise).

    Args:
        k_rot: (B, nh, T, D) — K after RoPE
        s_K:   (nh, D) or (1, nh, 1, D) per-head per-channel scale
    Returns:
        K_smoothed: (B, nh, T, D)
    """
    if s_K.dim() == 2:
        s_K = s_K.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, D)
    return k_rot / s_K


def apply_smooth_query(q_rot: torch.Tensor, s_K: torch.Tensor) -> torch.Tensor:
    """Apply K-side smoothing to Q at attention time: Q' = Q * s_K.

    This is the cheap per-step op that cancels the K-side division,
    since Q * K_rot^T = (Q * s_K) * (K_rot / s_K)^T.

    Args:
        q_rot: (B, num_heads, T, D) — Q after RoPE
        s_K:   (num_kv_heads, D) — K-side scale, broadcast across Q heads within each KV group
    Returns:
        Q_smoothed: (B, num_heads, T, D)
    """
    B, nh_q, T, D = q_rot.shape
    nh_kv = s_K.shape[0]
    n_rep = nh_q // nh_kv
    # Repeat s_K to match Q head count
    s_K_expanded = s_K.unsqueeze(1).expand(nh_kv, n_rep, D).reshape(nh_q, D)
    return q_rot * s_K_expanded.unsqueeze(0).unsqueeze(2)


def apply_smooth_value(v: torch.Tensor, s_V: torch.Tensor) -> torch.Tensor:
    """Apply V-side smoothing: V_smoothed = V / s_V.

    Args:
        v:   (B, nh, T, D)
        s_V: (nh, D) or broadcastable shape
    Returns:
        V_smoothed: (B, nh, T, D)
    """
    if s_V.dim() == 2:
        s_V = s_V.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, D)
    return v / s_V


def unsmooth_value(v_attn: torch.Tensor, s_V: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Restore the V-side scale after attention has been computed on smoothed V.

    attn_output = A @ V = A @ (V_smoothed * s_V) = (A @ V_smoothed) * s_V.

    Args:
        v_attn: (B, num_heads, T_q, D)
        s_V:    (num_kv_heads, D)
        n_rep:  num_heads / num_kv_heads
    Returns:
        attn_output: (B, num_heads, T_q, D) — scaled
    """
    nh_kv, D = s_V.shape
    nh_q = nh_kv * n_rep
    s_V_expanded = s_V.unsqueeze(1).expand(nh_kv, n_rep, D).reshape(nh_q, D)
    return v_attn * s_V_expanded.unsqueeze(0).unsqueeze(2)
