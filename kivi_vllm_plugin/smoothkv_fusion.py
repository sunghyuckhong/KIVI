"""In-place fusion of SmoothKV s_K, s_V scales into qkv_proj / o_proj weights
(and into q_norm.gamma / k_norm.gamma if the model has Q/K-norm).

Mathematical equivalence (assuming pair-max s_K so RoPE and per-channel division commute):
  K_smoothed = K / s_K, Q_smoothed = Q * s_K, attention(Q_s, K_s, V) = attention(Q, K, V)
  V_smoothed = V / s_V, attn_output = (attn_smoothed) * s_V

Two fusion paths depending on the architecture:

(A) Models WITHOUT Q/K-RMSNorm  (Llama-3, Mistral, DSR1-Distill-Llama, ...):
  qkv_proj output = [Q_block, K_block, V_block] along the output dim. To get
  smoothed projections directly:
    Q_block:  multiply by s_K  (broadcast over GQA)  →  W[Q rows]  *=  s_K_q
    K_block:  divide   by s_K                         →  W[K rows]  /=  s_K
    V_block:  divide   by s_V                         →  W[V rows]  /=  s_V
  s_K may be per-(kv_head, channel) — `num_kv_heads × head_dim/2` distinct
  values per layer (after the half-pair-equal RoPE constraint).

(B) Models WITH Q/K-RMSNorm   (Qwen3, Qwen3-MoE, ...):
  Pre-RoPE the path is  qkv_proj → reshape per-head → q_norm / k_norm → RoPE.
  Per-channel scaling on the qkv_proj rows does NOT survive RMSNorm because
  RMSNorm(x*s) ≠ s*RMSNorm(x) when s varies along the normalization axis (the
  RMS statistic itself depends on s). So we cannot fold s_K into qkv_proj
  rows. Instead we fold it into the post-norm gamma:
       new_q_norm.weight  =  q_norm.weight  *  s_K
       new_k_norm.weight  =  k_norm.weight  /  s_K
  But q_norm.weight / k_norm.weight have shape (head_dim,) shared across
  heads — so s_K must also be HEAD-UNIFORM (same row across all kv heads).
  That collapses the K-side granularity to `head_dim/2` distinct values per
  layer (vs `num_kv_heads × head_dim/2` in path A). V/O fusion still uses
  per-(head, channel) s_V — V has neither v_norm nor RoPE.

  o_proj input = attention output (per q_head, per channel). To absorb the
  required output-side s_V scaling:
    o_weight columns *= s_V_q  (broadcast over GQA)

Works in place on a TP-sharded vLLM model: each rank's qkv_proj/o_proj already
hold a partition of the head dim, so we slice s_K/s_V to the rank's local
kv_heads before applying.
"""
import re
import torch


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def fuse_smoothkv_into_model(model, sk_all, sv_all, tp_rank=0, tp_size=1):
    """In-place fuse s_K, s_V into the model's attention projections.

    Args:
        model: a loaded vLLM Qwen3ForCausalLM or Qwen3MoeForCausalLM instance.
        sk_all: tensor (num_layers, total_num_kv_heads, head_dim) — pair-max s_K.
        sv_all: tensor (num_layers, total_num_kv_heads, head_dim) — s_V.
        tp_rank, tp_size: tensor parallel rank/world-size for this worker.

    Returns:
        int: number of attention layers fused.
    """
    fused = 0
    fused_qknorm = 0
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if m is None:
            continue
        if not (hasattr(module, "qkv_proj") and hasattr(module, "o_proj")):
            continue
        # Skip non-attention layers that happen to have these (defensive).
        if not (name.endswith("self_attn") or name.endswith(".attn")):
            continue

        li = int(m.group(1))

        total_kv = sk_all.shape[1]
        # vLLM rule (qwen3.py): if total_num_kv_heads >= tp_size, partition;
        # else replicate. We mirror: kv_per_rank = max(1, total_kv // tp_size).
        kv_per_rank = max(1, total_kv // tp_size)
        if total_kv >= tp_size:
            kv_start = tp_rank * kv_per_rank
        else:
            # KV heads replicated across ranks; each rank uses all of them.
            kv_start = 0
            kv_per_rank = total_kv
        kv_end = kv_start + kv_per_rank

        device = module.qkv_proj.weight.device
        dtype = module.qkv_proj.weight.dtype
        s_K = sk_all[li, kv_start:kv_end].to(device=device, dtype=torch.float32)
        s_V = sv_all[li, kv_start:kv_end].to(device=device, dtype=torch.float32)

        n_rep = module.num_heads // module.num_kv_heads
        head_dim = module.head_dim
        q_size = module.q_size            # num_heads * head_dim (per rank)
        kv_size = module.kv_size          # num_kv_heads * head_dim (per rank)

        has_qknorm = hasattr(module, "q_norm") and hasattr(module, "k_norm")

        if has_qknorm:
            # Path B: model has Q/K-RMSNorm — pre-norm row scaling on qkv_proj
            # would be wiped by the RMS statistic, so fold s_K into the gamma
            # vectors instead. Requires HEAD-UNIFORM s_K (q_norm.weight has
            # shape (head_dim,) shared across heads).
            head_diff = (s_K - s_K[:1]).abs().max().item()
            if head_diff > 1e-5:
                raise RuntimeError(
                    f"layer {li}: q_norm/k_norm fusion requires head-uniform s_K "
                    f"but observed head-to-head diff = {head_diff:.4g}. Build the "
                    f"calib with `--head_uniform_k` (writes `_huk_…` files)."
                )
            s_K_per_channel = s_K[0]  # (head_dim,)
            with torch.no_grad():
                module.q_norm.weight.data.mul_(s_K_per_channel.to(module.q_norm.weight.dtype))
                module.k_norm.weight.data.div_(s_K_per_channel.to(module.k_norm.weight.dtype))

            # qkv_proj: only V rows get scaled (Q/K handled via norm gammas).
            s_V_flat = s_V.reshape(-1)  # (kv_size,)
            ones_q = torch.ones(q_size, device=device, dtype=torch.float32)
            ones_k = torch.ones(kv_size, device=device, dtype=torch.float32)
            qkv_scale = torch.cat([ones_q, ones_k, 1.0 / s_V_flat])
            fused_qknorm += 1
        else:
            # Path A: no Q/K-norm — fold s_K into qkv_proj rows directly.
            s_K_q = s_K.repeat_interleave(n_rep, dim=0).reshape(-1)  # (q_size,)
            s_K_flat = s_K.reshape(-1)                               # (kv_size,)
            s_V_flat = s_V.reshape(-1)                               # (kv_size,)
            qkv_scale = torch.cat([s_K_q, 1.0 / s_K_flat, 1.0 / s_V_flat])

        assert qkv_scale.shape[0] == q_size + 2 * kv_size, \
            f"layer {li} qkv scale len {qkv_scale.shape[0]} != q_size+2*kv_size {q_size + 2*kv_size}"

        with torch.no_grad():
            # weight shape (out, in); scale per-output-row
            module.qkv_proj.weight.data.mul_(qkv_scale.to(dtype).unsqueeze(-1))

        # o_proj input-channel scale: per q_head channel = s_V_q (independent of qknorm)
        s_V_q = s_V.repeat_interleave(n_rep, dim=0).reshape(-1)  # (q_size,)
        with torch.no_grad():
            # weight shape (out=hidden, in=q_size); scale per-input-column
            module.o_proj.weight.data.mul_(s_V_q.to(dtype).unsqueeze(0))

        fused += 1

    if fused_qknorm:
        # Trace marker so the activation log records which fusion path ran.
        try:
            with open("/tmp/kivi_plugin_calls.log", "a") as f:
                f.write(f"smoothkv_fused: q_norm/k_norm path used for {fused_qknorm}/{fused} layers\n")
        except Exception:
            pass

    return fused
