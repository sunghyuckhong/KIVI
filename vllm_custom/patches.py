"""
Monkey-patch helpers that insert fake quantization into vLLM's LlamaAttention.forward.

Approach: replace `LlamaAttention.forward` with a version that applies
quant→dequant on K and V (and optionally Q) after RoPE but before the paged
attention op. The K and V that go into vLLM's KV cache are the dequantized
versions, simulating lossy storage while keeping the cache in FP16.

Only one method can be active per process (the patch is global). Call
`install_<method>(...)` once before creating the vLLM LLM.

Usage:
    from vllm_custom.patches import install_fp8, install_pertoken_int4, install_smoothkv
    install_fp8(group_size=128)
    # then: LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", ...)
"""
import torch

import vllm.model_executor.models.llama as _vllm_llama
from vllm_custom.fake_quant_utils import (
    fake_quantize_fp8,
    fake_quantize_k_perchannel,
    fake_quantize_v_pertoken,
    fake_quantize_k_pertoken,
)


_ORIG_FORWARD = None


def _ensure_original_saved():
    global _ORIG_FORWARD
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = _vllm_llama.LlamaAttention.forward


def restore():
    """Undo any patch applied by install_*."""
    global _ORIG_FORWARD
    if _ORIG_FORWARD is not None:
        _vllm_llama.LlamaAttention.forward = _ORIG_FORWARD


def install_fp8(group_size: int = 128):
    """Fake-quant K and V at FP8 (symmetric, e4m3fn)."""
    _ensure_original_saved()

    def fp8_forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # Fake-quant hook
        k = fake_quantize_fp8(k, self.num_kv_heads, self.head_dim, group_size)
        v = fake_quantize_fp8(v, self.num_kv_heads, self.head_dim, group_size)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_llama.LlamaAttention.forward = fp8_forward


def install_pertoken_int4(group_size: int = 128):
    """Fake-quant K and V at INT4 per-token (KIVI pertoken scheme, residual=0)."""
    _ensure_original_saved()

    def pertoken_forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # Both K and V use per-token groups along head_dim
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=4)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=4)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_llama.LlamaAttention.forward = pertoken_forward


def install_smoothkv(calib_path: str, group_size: int = 128, bits: int = 4):
    """Fake-quant K and V with SmoothKV: scale K by s_K^-1, scale V by s_V, quant, invert.

    calib_path: path to the .pt file produced by scripts/make_*_variants.py.
    """
    _ensure_original_saved()

    calib = torch.load(calib_path, weights_only=True)
    # calib layout: {"s_K": tensor[L, nh, D], "s_V": tensor[L, nh, D], ...}
    s_K_all = calib["s_K"].to(torch.float16).cuda()  # (num_layers, num_kv_heads, head_dim)
    s_V_all = calib["s_V"].to(torch.float16).cuda()

    # Map layer index → s_K and s_V tensors, keyed by module id
    _layer_scales = {}

    def _get_scales(self):
        """Derive layer index from the attention module's prefix name."""
        # vLLM's LlamaAttention sets self.prefix to e.g. "model.layers.12.self_attn.attn"
        key = id(self)
        if key in _layer_scales:
            return _layer_scales[key]
        # Walk up: prefix is stored on self.attn.prefix
        pfx = getattr(self.attn, "prefix", None) or getattr(self, "prefix", "")
        # parse layer idx
        try:
            import re
            m = re.search(r"layers\.(\d+)\.", pfx)
            layer_idx = int(m.group(1))
            sk = s_K_all[layer_idx]  # (num_kv_heads, D)
            sv = s_V_all[layer_idx]
            _layer_scales[key] = (sk, sv)
            return sk, sv
        except Exception as e:
            raise RuntimeError(f"Failed to find layer idx from prefix={pfx!r}: {e}")

    def smoothkv_forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        sk, sv = _get_scales(self)  # (num_kv_heads, head_dim)
        sk_flat = sk.reshape(-1)  # (num_kv_heads * head_dim,) broadcastable over K rows
        sv_flat = sv.reshape(-1)

        # Fake-quant round-trip in "smoothed" space, matching HFLM SmoothKV semantics
        # (models/llama_smoothkv.py + quant/smoothkv_quant.py).
        # K: k_smooth = k / s_K; quant(k_smooth); dequant back to original via * s_K.
        # V: v_smooth = v / s_V; quant(v_smooth); dequant back to original via * s_V.
        # Q is unchanged. Net effect: K and V are restored to original range with
        # quant error introduced in the smoothed space.
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        k_s = k / sk_flat
        k_s = fake_quantize_k_pertoken(k_s, kv_heads, head_dim, group_size, bits=bits)
        k = k_s * sk_flat

        v_s = v / sv_flat
        v_s = fake_quantize_v_pertoken(v_s, kv_heads, head_dim, group_size, bits=bits)
        v = v_s * sv_flat

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_llama.LlamaAttention.forward = smoothkv_forward


def install_kivi2(group_size: int = 32, residual: int = 128):
    """Fake-quant for KIVI-2 (K per-channel INT2, V per-token INT2, R=128 FP16 residual).

    CAVEAT: the residual buffer is stateful across decode steps. In vLLM we can't
    retroactively re-quantize tokens after they're cached. The simplified scheme
    used here applies quant→dequant to K/V at every step without distinguishing
    'in-residual' vs 'out-of-residual' tokens. This will lose accuracy vs the HFLM
    reference by approximately the residual-buffer contribution.

    TODO: a faithful residual implementation would require a vLLM fork that
    exposes cache-mutation hooks (or a custom KV cache layout).
    """
    _ensure_original_saved()

    def kivi2_forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # K per-channel groups along T; approximate by per-token for single-step
        # (can't really do per-channel for a single token — defaulting to per-token).
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=2)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=2)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_llama.LlamaAttention.forward = kivi2_forward
