"""
Fake-quantization helpers for vLLM integration.

These wrap KIVI's low-level quant/dequant primitives in a shape-agnostic
"quantize-then-dequantize" op suitable for insertion into vLLM's attention
forward path.

vLLM's attention module consumes K/V tensors in (num_tokens, num_kv_heads * head_dim)
layout (prefill) or (1, num_kv_heads * head_dim) per token (decode). We reshape
into KIVI's canonical (B, num_kv_heads, T, head_dim) layout, apply quant→dequant,
and reshape back. Result goes into vLLM's PagedAttention cache as FP16 — the
"fake" part is that bit-width constraint was simulated but storage is full-precision.
"""
import torch

from quant.fp8_quant import (
    quantize_fp8 as _fp8_q,
    dequantize_fp8 as _fp8_dq,
    fake_quantize_dequantize_fp8 as _fp8_qdq,
)
from quant.new_pack import (
    quant_and_pack_kcache, unpack_and_dequant_kcache,
    quant_and_pack_vcache, unpack_and_dequant_vcache,
)


def _to_bnhtd(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape vLLM's (num_tokens, num_kv_heads * head_dim) or (num_tokens, num_kv_heads, head_dim)
    into KIVI's (B=1, nh, T, D) layout."""
    if x.dim() == 2:
        T = x.shape[0]
        return x.view(1, T, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    elif x.dim() == 3:
        T = x.shape[0]
        return x.unsqueeze(0).transpose(1, 2).contiguous()
    else:
        raise ValueError(f"unexpected KV shape {x.shape}")


def _from_bnhtd(x4: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """Invert _to_bnhtd."""
    return x4.transpose(1, 2).contiguous().view(*orig_shape)


@torch.no_grad()
def fake_quantize_fp8(x: torch.Tensor, num_kv_heads: int, head_dim: int,
                      group_size: int = 128) -> torch.Tensor:
    """Fake FP8 quant: quant→dequant round-trip at FP8 E4M3 precision.

    Internally dispatches via `fake_quantize_dequantize_fp8`:
    - sm_89+ (Hopper, H100/H200): hardware torch.float8_e4m3fn cast (bit-exact)
    - sm_80 (A100): software E4M3 rounding in fp32 (cudagraph-compatible)
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = _fp8_qdq(x4, group_size=group_size)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


@torch.no_grad()
def fake_quantize_k_perchannel(x: torch.Tensor, num_kv_heads: int, head_dim: int,
                                group_size: int, bits: int) -> torch.Tensor:
    """Fake KIVI-style per-channel K quant (groups along T dim).

    Used by KIVI-2 and any per-channel K methods.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)  # (B,nh,T,D) — keep input dtype
    B, nh, T, D = x4.shape
    # quant_and_pack_kcache requires T % group_size == 0
    pad = (-T) % group_size
    if pad:
        x4 = torch.cat([x4, torch.zeros(B, nh, pad, D, dtype=x4.dtype, device=x4.device)], dim=2)
    code, scale, mn = quant_and_pack_kcache(x4, group_size, bits)
    out = unpack_and_dequant_kcache(code, scale, mn, group_size, bits)
    if pad:
        out = out[:, :, :T, :]
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


@torch.no_grad()
def fake_quantize_v_pertoken(x: torch.Tensor, num_kv_heads: int, head_dim: int,
                              group_size: int, bits: int) -> torch.Tensor:
    """Fake KIVI-style per-token V quant (groups along D dim).

    Used by KIVI-2 (V side), pertoken (V side), SmoothKV (V side).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)  # (B,nh,T,D) — keep input dtype
    B, nh, T, D = x4.shape
    assert D % group_size == 0, f"head_dim {D} not divisible by group_size {group_size}"
    code, scale, mn = quant_and_pack_vcache(x4, group_size, bits)
    out = unpack_and_dequant_vcache(code, scale, mn, group_size, bits)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


@torch.no_grad()
def fake_quantize_k_pertoken(x: torch.Tensor, num_kv_heads: int, head_dim: int,
                              group_size: int, bits: int) -> torch.Tensor:
    """Fake per-token K quant (groups along D dim, same scheme as V).

    Used by pertoken method and SmoothKV (after smoothing).
    """
    # Same packing scheme as V (groups along last dim = head_dim)
    return fake_quantize_v_pertoken(x, num_kv_heads, head_dim, group_size, bits)
