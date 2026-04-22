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
from vllm.model_executor.models.utils import extract_layer_index
from vllm_custom.fake_quant_utils import (
    fake_quantize_fp8,
    fake_quantize_k_perchannel,
    fake_quantize_v_pertoken,
    fake_quantize_k_pertoken,
)


_ORIG_FORWARD = None
_ORIG_INIT = None


def _ensure_original_saved():
    global _ORIG_FORWARD
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = _vllm_llama.LlamaAttention.forward


def _install_layer_idx_hook():
    """Patch LlamaAttention.__init__ to stash `self._kivi_layer_idx`.

    vLLM 0.6.6 passes `prefix="model.layers.N.self_attn"` to __init__ but
    doesn't store it on the instance. SmoothKV needs per-layer scales, so we
    hook __init__ to save the parsed layer index before the LLM builds layers.
    """
    global _ORIG_INIT
    if _ORIG_INIT is not None:
        return  # already installed
    _ORIG_INIT = _vllm_llama.LlamaAttention.__init__

    def patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if not prefix and args:
            # prefix is the last positional arg when passed positionally
            last = args[-1]
            if isinstance(last, str):
                prefix = last
        try:
            self._kivi_layer_idx = extract_layer_index(prefix)
        except Exception:
            self._kivi_layer_idx = None

    _vllm_llama.LlamaAttention.__init__ = patched_init


def restore():
    """Undo any patch applied by install_*."""
    global _ORIG_FORWARD, _ORIG_INIT
    if _ORIG_FORWARD is not None:
        _vllm_llama.LlamaAttention.forward = _ORIG_FORWARD
    if _ORIG_INIT is not None:
        _vllm_llama.LlamaAttention.__init__ = _ORIG_INIT
        _ORIG_INIT = None


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
    _install_layer_idx_hook()

    calib = torch.load(calib_path, weights_only=True)
    # calib layout: {"s_K": tensor[L, nh, D], "s_V": tensor[L, nh, D], ...}
    s_K_all = calib["s_K"].to(torch.float16).cuda()  # (num_layers, num_kv_heads, head_dim)
    s_V_all = calib["s_V"].to(torch.float16).cuda()

    def _get_scales(self):
        """Layer idx is stored by the __init__ hook as self._kivi_layer_idx."""
        layer_idx = getattr(self, "_kivi_layer_idx", None)
        if layer_idx is None:
            raise RuntimeError(
                "SmoothKV: layer index unset. The __init__ hook must run before "
                "model construction — call install_smoothkv() before LLM(...)."
            )
        return s_K_all[layer_idx], s_V_all[layer_idx]

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
