"""
Monkey-patch helpers that insert fake quantization into vLLM's Qwen3Attention.forward.

Approach: replace `Qwen3Attention.forward` with a version that applies
quant→dequant on K and V (and optionally Q) after qk-norm + RoPE but before the
paged attention op. The K and V that go into vLLM's KV cache are the
dequantized versions, simulating lossy storage while keeping the cache in FP16.

Only one method can be active per process (the patch is global). Call
`install_<method>(...)` once before creating the vLLM LLM.

Usage:
    from vllm_custom.patches import install_fp8, install_pertoken_int4, install_smoothkv
    install_fp8(group_size=128)
    # then: LLM(model="Qwen/Qwen3-8B", ...)

vLLM 0.8 note: Qwen3Attention.forward takes only (positions, hidden_states)
— kv_cache and attn_metadata are accessed via thread-locals inside self.attn.
"""
import torch

import vllm.model_executor.models.qwen3 as _vllm_qwen3
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
        _ORIG_FORWARD = _vllm_qwen3.Qwen3Attention.forward


def _apply_qk_norm(self, q, k):
    """Replicates Qwen3Attention's qk-norm block (pre-RoPE, per head_dim)."""
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    q_by_head = self.q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = self.k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)
    return q, k


def _install_layer_idx_hook():
    """Patch Qwen3Attention.__init__ to stash `self._kivi_layer_idx`.

    vLLM passes `prefix="model.layers.N.self_attn"` to __init__ but doesn't
    store it on the instance. SmoothKV needs per-layer scales, so we hook
    __init__ to save the parsed layer index before the LLM builds layers.
    """
    global _ORIG_INIT
    if _ORIG_INIT is not None:
        return  # already installed
    _ORIG_INIT = _vllm_qwen3.Qwen3Attention.__init__

    def patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if not prefix and args:
            for a in args:
                if isinstance(a, str) and "layers." in a:
                    prefix = a
                    break
        try:
            self._kivi_layer_idx = extract_layer_index(prefix)
        except Exception:
            self._kivi_layer_idx = None

    _vllm_qwen3.Qwen3Attention.__init__ = patched_init


def restore():
    """Undo any patch applied by install_*."""
    global _ORIG_FORWARD, _ORIG_INIT
    if _ORIG_FORWARD is not None:
        _vllm_qwen3.Qwen3Attention.forward = _ORIG_FORWARD
    if _ORIG_INIT is not None:
        _vllm_qwen3.Qwen3Attention.__init__ = _ORIG_INIT
        _ORIG_INIT = None


def install_fp8(group_size: int = 128):
    """Fake-quant K and V at FP8 (symmetric, e4m3fn)."""
    _ensure_original_saved()

    def fp8_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = self.rotary_emb(positions, q, k)
        k = fake_quantize_fp8(k, self.num_kv_heads, self.head_dim, group_size)
        v = fake_quantize_fp8(v, self.num_kv_heads, self.head_dim, group_size)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_qwen3.Qwen3Attention.forward = fp8_forward


def install_pertoken_int4(group_size: int = 128):
    """Fake-quant K and V at INT4 per-token (KIVI pertoken scheme, residual=0)."""
    _ensure_original_saved()

    def pertoken_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = self.rotary_emb(positions, q, k)
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=4)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=4)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_qwen3.Qwen3Attention.forward = pertoken_forward


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

    def smoothkv_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = self.rotary_emb(positions, q, k)

        sk, sv = _get_scales(self)  # (num_kv_heads, head_dim)
        sk_flat = sk.reshape(-1)  # (num_kv_heads * head_dim,) broadcastable over K rows
        sv_flat = sv.reshape(-1)

        # Same SmoothKV semantics as the Llama path (models/llama_smoothkv.py +
        # quant/smoothkv_quant.py). K: k/s_K → quant → ×s_K; V: v/s_V → quant → ×s_V.
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        k_s = k / sk_flat
        k_s = fake_quantize_k_pertoken(k_s, kv_heads, head_dim, group_size, bits=bits)
        k = k_s * sk_flat

        v_s = v / sv_flat
        v_s = fake_quantize_v_pertoken(v_s, kv_heads, head_dim, group_size, bits=bits)
        v = v_s * sv_flat

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_qwen3.Qwen3Attention.forward = smoothkv_forward


def install_kivi2(group_size: int = 32, residual: int = 128):
    """Fake-quant for KIVI-2 (K per-channel INT2, V per-token INT2, R=128 FP16 residual).

    CAVEAT: the residual buffer is stateful across decode steps. In vLLM we can't
    retroactively re-quantize tokens after they're cached. The simplified scheme
    used here applies quant→dequant to K/V at every step without distinguishing
    'in-residual' vs 'out-of-residual' tokens. Lossy vs HFLM reference by roughly
    the residual-buffer contribution.
    """
    _ensure_original_saved()

    def kivi2_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = self.rotary_emb(positions, q, k)
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=2)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=2)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _vllm_qwen3.Qwen3Attention.forward = kivi2_forward
