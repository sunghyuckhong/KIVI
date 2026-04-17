"""
Fine-grained FP8 (e4m3fn) quantization for KV cache.

Per-token quantization with configurable group_size along head_dim.
Each group of elements shares one scale factor (no zero-point needed
since FP8 is symmetric around zero).

Quantize:   scale = max(|x|) / FP8_MAX;  x_fp8 = (x / scale).to(float8_e4m3fn)
Dequantize: x_fp16 = x_fp8.to(float16) * scale
"""
import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0


def quantize_fp8(
    data: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (B, nh, T, D) tensor to FP8 per-token with group_size along D.

    FP8 tensors are stored as uint8 views for torch.cat compatibility.

    Returns:
        data_uint8: (B, nh, T, D) in uint8 (bit-identical to float8_e4m3fn)
        scale:      (B, nh, T, D // group_size) in float16
    """
    assert len(data.shape) == 4
    B, nh, T, D = data.shape
    assert D % group_size == 0
    num_groups = D // group_size

    # Reshape to expose groups: (B, nh, T, num_groups, group_size)
    data_grouped = data.view(B, nh, T, num_groups, group_size)

    # Compute per-group scale: max(|x|) / FP8_MAX
    amax = data_grouped.abs().amax(dim=-1)  # (B, nh, T, num_groups)
    scale = (amax / _FP8_MAX).clamp(min=1e-12)  # avoid div by zero

    # Scale and cast to FP8, then store as uint8 for cat compatibility
    data_scaled = data_grouped / scale.unsqueeze(-1)
    data_fp8 = data_scaled.to(_FP8_DTYPE).view(B, nh, T, D)
    data_uint8 = data_fp8.view(torch.uint8)

    return data_uint8, scale.to(torch.float16)


def dequantize_fp8(
    data_uint8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize a (B, nh, T, D) FP8 tensor (stored as uint8) back to float16.

    Args:
        data_uint8: (B, nh, T, D) in uint8 (bit-identical to float8_e4m3fn)
        scale:      (B, nh, T, D // group_size) in float16
    Returns:
        (B, nh, T, D) in float16
    """
    B, nh, T, D = data_uint8.shape
    num_groups = D // group_size

    # Reinterpret uint8 as FP8
    data_fp8 = data_uint8.view(_FP8_DTYPE)

    # Expand scale from (B, nh, T, num_groups) -> (B, nh, T, D)
    scale_expanded = scale.unsqueeze(-1).expand(B, nh, T, num_groups, group_size).reshape(B, nh, T, D)

    return data_fp8.to(torch.float16) * scale_expanded
