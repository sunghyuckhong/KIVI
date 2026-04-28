"""
Monkey-patch helpers that insert fake quantization into vLLM's
Qwen3Attention, Qwen2Attention, and Exaone4Attention forwards.

Approach: replace `forward` with a version that applies quant→dequant on K
and V (and optionally Q) after qk-norm + RoPE but before the paged attention
op. The K and V that go into vLLM's KV cache are the dequantized versions,
simulating lossy storage while keeping the cache in FP16.

Per-architecture differences (handled via runtime attribute probes):
  - Qwen3 / Exaone4 have qk-norm pre-RoPE; Qwen2 does not.
  - Exaone4 hybrid configs skip RoPE on full-attention layers (NoPE).
    The patched forward checks `apply_rope_all_layers` and `sliding_window`
    on the attention instance, matching the model's own forward logic.

Only one method can be active per process (the patch is global). Call
`install_<method>(...)` once before creating the vLLM LLM.

Usage:
    from vllm_custom.patches import install_fp8, install_pertoken_int4, install_smoothkv
    install_fp8(group_size=128)
    # then: LLM(model="Qwen/Qwen3-8B", ...)
    #   or: LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ...)
    #   or: LLM(model="LGAI-EXAONE/EXAONE-4.5-33B", ...)

vLLM 0.8+ note: these Attention.forward methods take only (positions,
hidden_states) — kv_cache and attn_metadata are accessed via thread-locals
inside self.attn.
"""
import torch

import vllm.model_executor.models.qwen3 as _vllm_qwen3
import vllm.model_executor.models.qwen2 as _vllm_qwen2
from vllm.model_executor.models.utils import extract_layer_index
from vllm_custom.fake_quant_utils import (
    fake_quantize_fp8,
    fake_quantize_k_perchannel,
    fake_quantize_v_pertoken,
    fake_quantize_k_pertoken,
)


def _build_attn_classes():
    classes = [_vllm_qwen3.Qwen3Attention, _vllm_qwen2.Qwen2Attention]
    try:
        import vllm.model_executor.models.qwen3_moe as _vllm_qwen3_moe
        classes.append(_vllm_qwen3_moe.Qwen3MoeAttention)
    except ImportError:
        pass
    try:
        import vllm.model_executor.models.exaone4 as _vllm_exaone4
        classes.append(_vllm_exaone4.Exaone4Attention)
    except ImportError:
        pass
    return tuple(classes)


# Per-class originals so install/restore covers every supported attention path.
# Qwen2 has no qk-norm (added in Qwen3/Exaone4); Exaone4 hybrid configs skip
# RoPE on full-attention layers. The patched forward probes for those at runtime.
_ATTN_CLASSES = _build_attn_classes()
_ORIG_FORWARD: dict = {}
_ORIG_INIT: dict = {}


def _ensure_original_saved():
    for cls in _ATTN_CLASSES:
        if cls not in _ORIG_FORWARD:
            _ORIG_FORWARD[cls] = cls.forward


def _apply_qk_norm(self, q, k):
    """Replicates Qwen3Attention's qk-norm block (pre-RoPE, per head_dim).

    Qwen2Attention has no qk-norm — return q,k unchanged in that case so the
    same forward function works for both architectures.
    """
    if not hasattr(self, "q_norm"):
        return q, k
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    q_by_head = self.q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = self.k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)
    return q, k


def _apply_rope_if_needed(self, positions, q, k):
    """Apply RoPE matching the model's per-layer policy.

    Qwen3/Qwen2 always apply RoPE — `apply_rope_all_layers` and `sliding_window`
    attrs don't exist there, so the default-True branch fires.

    Exaone4 hybrid configs: full-attention layers have `sliding_window=None` and
    `apply_rope_all_layers=False` → skip RoPE entirely (NoPE on full-attn).
    Sliding-attention layers have `sliding_window` set → apply RoPE.
    """
    apply_all = getattr(self, "apply_rope_all_layers", True)
    sliding = getattr(self, "sliding_window", True)
    if apply_all or sliding:
        return self.rotary_emb(positions, q, k)
    return q, k


def _assign_forward(fn):
    """Install the same forward on every attention class we cover."""
    for cls in _ATTN_CLASSES:
        cls.forward = fn


def _install_layer_idx_hook():
    """Patch Qwen3/Qwen2 Attention.__init__ to stash `self._kivi_layer_idx`.

    vLLM passes `prefix="model.layers.N.self_attn"` to __init__ but doesn't
    store it on the instance. SmoothKV needs per-layer scales, so we hook
    __init__ to save the parsed layer index before the LLM builds layers.
    """
    for cls in _ATTN_CLASSES:
        if cls in _ORIG_INIT:
            continue
        orig = cls.__init__
        _ORIG_INIT[cls] = orig

        def make_patched(orig_init):
            def patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
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
            return patched_init

        cls.__init__ = make_patched(orig)


def restore():
    """Undo any patch applied by install_*."""
    for cls, fwd in list(_ORIG_FORWARD.items()):
        cls.forward = fwd
    for cls, init in list(_ORIG_INIT.items()):
        cls.__init__ = init
    _ORIG_INIT.clear()


def install_fp8(group_size: int = 128):
    """Fake-quant K and V at FP8 (symmetric, e4m3fn)."""
    _ensure_original_saved()

    def fp8_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = _apply_rope_if_needed(self, positions, q, k)
        k = fake_quantize_fp8(k, self.num_kv_heads, self.head_dim, group_size)
        v = fake_quantize_fp8(v, self.num_kv_heads, self.head_dim, group_size)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _assign_forward(fp8_forward)


def install_pertoken_int4(group_size: int = 128):
    """Fake-quant K and V at INT4 per-token (KIVI pertoken scheme, residual=0)."""
    _ensure_original_saved()

    def pertoken_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _apply_qk_norm(self, q, k)
        q, k = _apply_rope_if_needed(self, positions, q, k)
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=4)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=4)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _assign_forward(pertoken_forward)


def install_smoothkv(calib_path: str, group_size: int = 128, bits: int = 4,
                     dtype: torch.dtype = torch.bfloat16):
    """Fake-quant K and V with SmoothKV: scale K by s_K^-1, scale V by s_V, quant, invert.

    calib_path: path to the .pt file produced by scripts/make_*_variants.py.
    dtype: smoothing-factor dtype — should match the model's runtime dtype.
           Most recent models (Qwen3, Llama-3, Mistral-7B) are bfloat16; Llama-2
           is float16. Override this default if running with a non-bf16 model.
    """
    _ensure_original_saved()
    _install_layer_idx_hook()

    calib = torch.load(calib_path, weights_only=True)
    # calib layout: {"s_K": tensor[L, nh, D], "s_V": tensor[L, nh, D], ...}
    # Store smoothing factors at model_dtype so K/V division and post-quant rescale
    # stay in the model's native precision (no lossy cast chain per forward).
    s_K_all = calib["s_K"].to(dtype).cuda()  # (num_layers, num_kv_heads, head_dim)
    s_V_all = calib["s_V"].to(dtype).cuda()

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
        q, k = _apply_rope_if_needed(self, positions, q, k)

        sk, sv = _get_scales(self)  # (num_kv_heads, head_dim) — already at model_dtype
        sk_flat = sk.reshape(-1)
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

    _assign_forward(smoothkv_forward)


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
        q, k = _apply_rope_if_needed(self, positions, q, k)
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, group_size, bits=2)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, group_size, bits=2)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _assign_forward(kivi2_forward)
