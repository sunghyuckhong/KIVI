"""
Attribute-based KV-quant dispatch for vLLM Attention forwards.

Approach: install ONE unified forward on every supported Attention class. The
forward reads a class-level attribute `_kv_quant_method` and dispatches via
`if/elif` to bf16 / fp8 / pertoken / smoothkv / kivi2. torch.compile sees the
attribute as a Python constant at trace time, specializes on its value, and
emits a graph that contains ONLY the chosen branch's quant kernels — so
verify_compiled_graph.py still works without changes.

Why not per-method monkey-patch (the previous design):
  - Hard to debug: the active forward depends on import/load order.
  - Race conditions in the compile-cache wipe sentinel — when N parallel
    workers all install_*(), only one wipes; others can hit a stale
    compiled graph and silently run the unpatched bf16 forward (observed
    on smk-RAW pass1 — verify-graph FAIL).
  - Five near-identical forward bodies that drift out of sync.

This design:
  - Single forward, branches in plain `if/elif`. One compiled graph per
    (model, _kv_quant_method, group_size, bits) — dynamo specializes
    naturally on the attribute value.
  - Class attributes are set BEFORE LLM(...) is constructed, so every
    instance picks up the same value. Switching methods is a one-shot
    attribute set + cls.forward replacement.

Usage:
    from vllm_custom.patches import (
        configure_kv_quant,            # primary API
        install_fp8, install_pertoken_int4, install_smoothkv, install_kivi2,  # back-compat
    )
    configure_kv_quant("smoothkv", group_size=128, bits=4,
                       calib_path="logs/calib/smoothkv_qwen3-8b_..._halfpair.pt")
    # then: LLM(model="Qwen/Qwen3-8B", ...)

Per-architecture differences (handled via runtime attribute probes):
  - Qwen3 / Exaone4 have qk-norm pre-RoPE; Qwen2/Llama/Mistral don't.
  - Exaone4 hybrid configs skip RoPE on full-attention layers (NoPE).

vLLM 0.8+ note: Attention.forward takes only (positions, hidden_states) —
kv_cache and attn_metadata flow through thread-locals inside self.attn.
"""
import torch

from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen2 import Qwen2Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.exaone4 import Exaone4Attention
from vllm.model_executor.models.llama import LlamaAttention
from vllm.model_executor.models.mistral import MistralAttention
from vllm.model_executor.models.utils import extract_layer_index
from vllm_custom.fake_quant_utils import (
    fake_quantize_fp8,
    fake_quantize_k_perchannel,
    fake_quantize_v_pertoken,
    fake_quantize_k_pertoken,
)


# Attention classes covered. To add a new model, import its Attention class
# above and append it here. The unified forward handles qk-norm and RoPE
# policy at runtime via `_apply_qk_norm`/`_apply_rope_if_needed`.
_ATTN_CLASSES = (
    Qwen3Attention,
    Qwen2Attention,
    Qwen3MoeAttention,
    Exaone4Attention,
    LlamaAttention,
    MistralAttention,
)
_ORIG_FORWARD: dict = {}
_ORIG_INIT: dict = {}

# Registered by configure_kv_quant(method="smoothkv") so the post-load hook
# can pre-move calib scales to GPU before cudagraph capture begins.
_POSTLOAD_HOOKS: list = []

# SmoothKV state — module-level so the unified forward can reach it without
# closure capture.
_SMOOTHKV_S_K_CPU = None       # (num_layers, num_kv_heads, head_dim) bf16 on CPU
_SMOOTHKV_S_V_CPU = None
_SMOOTHKV_SCALES_GPU: dict = {}  # layer_idx -> (sk_gpu, sv_gpu) — filled by warmup hook


# ---------------------------------------------------------------------------
# Unified forward — single function, branches on `self._kv_quant_method`
# ---------------------------------------------------------------------------

def kv_quant_unified_forward(self, positions, hidden_states):
    """Single forward for all attention classes. Branches on _kv_quant_method.

    torch.compile specialization:
      - `_kv_quant_method` is a Python str class attribute, treated as constant
        at trace time. dynamo emits a guard on its value and the compiled
        graph contains ONLY the matching branch.
      - `_kv_quant_group_size` and `_kv_quant_bits` are int attributes, also
        constant-folded.

    verify_compiled_graph then sees:
      bf16     → no quant kernels (FORBIDDEN check passes)
      fp8      → fake_quantize_dequantize_fp8
      pertoken → quant_and_pack_vcache + unpack_and_dequant_vcache
      smoothkv → quant_and_pack_vcache (after K is divided by s_K)
      kivi2    → quant_and_pack_vcache (bits=2)
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = _apply_qk_norm(self, q, k)
    q, k = _apply_rope_if_needed(self, positions, q, k)

    method = getattr(self, "_kv_quant_method", "bf16")
    gs = getattr(self, "_kv_quant_group_size", 128)
    bits = getattr(self, "_kv_quant_bits", 4)

    if method == "bf16" or method == "fp16":
        pass  # baseline — no quant
    elif method == "fp8":
        k = fake_quantize_fp8(k, self.num_kv_heads, self.head_dim, gs)
        v = fake_quantize_fp8(v, self.num_kv_heads, self.head_dim, gs)
    elif method == "pertoken":
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, gs, bits=bits)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, gs, bits=bits)
    elif method == "smoothkv":
        sk, sv = _get_scales(self)
        sk_flat = sk.reshape(-1)
        sv_flat = sv.reshape(-1)
        k_s = k / sk_flat
        k_s = fake_quantize_k_pertoken(k_s, self.num_kv_heads, self.head_dim, gs, bits=bits)
        k = k_s * sk_flat
        v_s = v / sv_flat
        v_s = fake_quantize_v_pertoken(v_s, self.num_kv_heads, self.head_dim, gs, bits=bits)
        v = v_s * sv_flat
    elif method == "kivi2":
        # KIVI-2: K per-channel int2, V per-token int2, R=128 FP16 residual
        # (residual buffer is stateful — not faithfully simulated here, see comment).
        k = fake_quantize_k_pertoken(k, self.num_kv_heads, self.head_dim, gs, bits=2)
        v = fake_quantize_v_pertoken(v, self.num_kv_heads, self.head_dim, gs, bits=2)
    else:
        raise ValueError(f"Unknown _kv_quant_method: {method!r}")

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_original_saved():
    _wipe_compile_cache()
    for cls in _ATTN_CLASSES:
        if cls not in _ORIG_FORWARD:
            _ORIG_FORWARD[cls] = cls.forward
    _install_postload_assertion()


def _wipe_compile_cache():
    """Delete vLLM's torch.compile cache before any configure_kv_quant() runs.

    Race-safe: parallel workers (e.g. wave of N quant cells) would otherwise
    wipe the cache while siblings are mid-compile. Exclusive lockfile +
    per-launch sentinel: FIRST process wipes, the rest skip.
    """
    import os, shutil, fcntl
    cache_dir = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
    lock_dir = os.path.expanduser("~/.cache/vllm")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, ".kivi_wipe.lock")
    sentinel_path = os.path.join(lock_dir, ".kivi_wiped_this_launch")
    launch_id = os.environ.get("KIVI_LAUNCH_ID", "default")

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        already = False
        if os.path.exists(sentinel_path):
            try:
                if open(sentinel_path).read().strip() == launch_id:
                    already = True
            except OSError:
                pass
        if already:
            print(f"[vllm_custom.patches] another process already wiped cache "
                  f"this launch (pid {os.getpid()} skipping)")
        else:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except OSError as e:
                    print(f"[vllm_custom.patches] WARNING could not wipe "
                          f"{cache_dir}: {e}")
                    return
            with open(sentinel_path, "w") as sf:
                sf.write(launch_id)
            print(f"[vllm_custom.patches] wiped {cache_dir} "
                  f"(pid {os.getpid()} owns sentinel for launch_id={launch_id})")


def _install_postload_assertion():
    """Patch Worker.load_model so AFTER load we count attention modules whose
    class is in _ATTN_CLASSES. If zero, the patch is a silent no-op for this
    model — fail loud."""
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        return
    if getattr(Worker, "_kivi_postload_patched", False):
        return
    orig_load = Worker.load_model

    def patched_load(self, *args, **kwargs):
        ret = orig_load(self, *args, **kwargs)
        model = self.model_runner.model
        n = sum(1 for m in model.modules() if isinstance(m, _ATTN_CLASSES))
        if n == 0:
            names = [c.__name__ for c in _ATTN_CLASSES]
            raise RuntimeError(
                f"[vllm_custom.patches] FAIL-LOUD: 0 attention modules in the loaded "
                f"model match the patched classes {names}. Quantization is a silent "
                f"no-op. Add the model's attention class to _ATTN_CLASSES."
            )
        print(f"[vllm_custom.patches] post-load assertion: {n} attention modules patched")
        for hook in _POSTLOAD_HOOKS:
            hook(model)
        return ret

    Worker.load_model = patched_load
    Worker._kivi_postload_patched = True


def _apply_qk_norm(self, q, k):
    """Replicate qk-norm block (pre-RoPE, per head_dim).
    Qwen3/Exaone4 have q_norm/k_norm; Qwen2/Llama/Mistral don't (return q,k as-is)."""
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
    """Apply RoPE per the model's policy. Qwen3/Qwen2 always apply RoPE;
    Exaone4 hybrid configs skip RoPE on full-attention layers (NoPE)."""
    apply_all = getattr(self, "apply_rope_all_layers", True)
    sliding = getattr(self, "sliding_window", True)
    if apply_all or sliding:
        return self.rotary_emb(positions, q, k)
    return q, k


def _install_layer_idx_hook():
    """Patch Attention.__init__ to stash `self._kivi_layer_idx` so SmoothKV
    can index per-layer calib scales."""
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


def _get_scales(self):
    """SmoothKV: look up per-layer (s_K, s_V) GPU tensors pre-warmed by the
    post-load hook. Called from the unified forward when method == 'smoothkv'."""
    layer_idx = getattr(self, "_kivi_layer_idx", None)
    if layer_idx is None:
        raise RuntimeError(
            "SmoothKV: layer index unset. _install_layer_idx_hook() must run "
            "before model construction — call configure_kv_quant() before LLM(...)."
        )
    sk_sv = _SMOOTHKV_SCALES_GPU.get(layer_idx)
    if sk_sv is None:
        raise RuntimeError(
            f"SmoothKV: scales for layer {layer_idx} not pre-warmed. "
            f"Worker.load_model post-load hook didn't run, or this layer's "
            f"_kivi_layer_idx wasn't set."
        )
    return sk_sv


def _setup_smoothkv_calib(calib_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load SmoothKV calib (CPU-side) and register the post-load warmup hook.

    Keep scales on CPU here. Calling .cuda() in the main process initializes
    CUDA, which forces vLLM to use multiprocess spawn for its workers — spawn
    re-imports modules in the worker, so the worker's Attention class never
    picks up our forward and quant silently no-ops. We lazy-move to CUDA in
    the post-load hook (which runs INSIDE the worker)."""
    global _SMOOTHKV_S_K_CPU, _SMOOTHKV_S_V_CPU, _SMOOTHKV_SCALES_GPU
    calib = torch.load(calib_path, weights_only=True)
    _SMOOTHKV_S_K_CPU = calib["s_K"].to(dtype)  # (num_layers, num_kv_heads, head_dim)
    _SMOOTHKV_S_V_CPU = calib["s_V"].to(dtype)
    _SMOOTHKV_SCALES_GPU = {}

    def _warmup_scales(model):
        """Pre-move per-layer scales to each worker's GPU before cudagraph
        capture. CUDA→GPU memcpy inside captured graphs raises
        cudaErrorStreamCaptureUnsupported, so we must do it ahead of time.

        TP-aware: calib s_K has full_num_kv_heads but each worker only sees a
        slice → shard along kv_heads dim per tp_rank."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            tp_rank = 0
        full_kv_heads = _SMOOTHKV_S_K_CPU.shape[1]
        for m in model.modules():
            li = getattr(m, "_kivi_layer_idx", None)
            if li is None:
                continue
            try:
                dev = next(m.parameters()).device
            except StopIteration:
                continue
            per_worker_kv = getattr(m, "num_kv_heads", full_kv_heads)
            if per_worker_kv == full_kv_heads:
                sk = _SMOOTHKV_S_K_CPU[li].to(dev)
                sv = _SMOOTHKV_S_V_CPU[li].to(dev)
            else:
                lo = tp_rank * per_worker_kv
                hi = lo + per_worker_kv
                sk = _SMOOTHKV_S_K_CPU[li, lo:hi].to(dev)
                sv = _SMOOTHKV_S_V_CPU[li, lo:hi].to(dev)
            _SMOOTHKV_SCALES_GPU[li] = (sk, sv)
        print(f"[vllm_custom.patches] smoothkv warmup: pre-moved {len(_SMOOTHKV_SCALES_GPU)} "
              f"layer scales to GPU (cudagraph-safe, tp_rank={tp_rank}, "
              f"per_worker_kv={per_worker_kv if _SMOOTHKV_SCALES_GPU else '?'})")
    _POSTLOAD_HOOKS.append(_warmup_scales)


def restore():
    """Undo all patches applied by configure_kv_quant()."""
    for cls, fwd in list(_ORIG_FORWARD.items()):
        cls.forward = fwd
    for cls, init in list(_ORIG_INIT.items()):
        cls.__init__ = init
    _ORIG_INIT.clear()
    for cls in _ATTN_CLASSES:
        for attr in ("_kv_quant_method", "_kv_quant_group_size", "_kv_quant_bits"):
            if hasattr(cls, attr):
                delattr(cls, attr)


# ---------------------------------------------------------------------------
# Primary API: configure_kv_quant
# ---------------------------------------------------------------------------

def configure_kv_quant(method: str, group_size: int = 128, bits: int = 4,
                       calib_path: str = None, dtype: torch.dtype = torch.bfloat16):
    """Set up KV-cache quantization on every supported Attention class.

    Args:
        method:      "bf16" | "fp16" | "fp8" | "pertoken" | "smoothkv" | "kivi2"
        group_size:  per-channel group size (default 128)
        bits:        4 for pertoken/smoothkv, 2 for kivi2 (auto-set if method=="kivi2")
        calib_path:  required for method=="smoothkv"; ignored otherwise
        dtype:       smoothkv calib scale dtype (bf16 for Qwen3/Llama-3, fp16 for Llama-2)

    After this returns, every Attention instance constructed afterward will
    have the unified forward. The class attributes `_kv_quant_method`,
    `_kv_quant_group_size`, `_kv_quant_bits` are read at runtime; torch.compile
    specializes on their values and the FX graph contains ONLY the matching
    branch's quant kernels.
    """
    if method not in ("bf16", "fp16", "fp8", "pertoken", "smoothkv", "kivi2"):
        raise ValueError(f"Unknown method {method!r}; expected bf16/fp16/fp8/pertoken/smoothkv/kivi2")
    if method == "smoothkv" and not calib_path:
        raise ValueError("method='smoothkv' requires calib_path")
    if method == "kivi2":
        bits = 2  # KIVI-2 is hardcoded int2

    _ensure_original_saved()
    _install_layer_idx_hook()

    # Class-level config — torch.compile constant-folds these
    for cls in _ATTN_CLASSES:
        cls._kv_quant_method = method
        cls._kv_quant_group_size = group_size
        cls._kv_quant_bits = bits
        cls.forward = kv_quant_unified_forward

    if method == "smoothkv":
        _setup_smoothkv_calib(calib_path, dtype)

    print(f"[vllm_custom.patches] configure_kv_quant: method={method} "
          f"group_size={group_size} bits={bits}"
          + (f" calib_path={calib_path}" if calib_path else ""))


# ---------------------------------------------------------------------------
# Backward-compat thin wrappers (so existing run_eval_vllm.py / adaptive_pass2.py
# don't need to change).
# ---------------------------------------------------------------------------

def install_fp8(group_size: int = 128):
    configure_kv_quant("fp8", group_size=group_size)

def install_pertoken_int4(group_size: int = 128):
    configure_kv_quant("pertoken", group_size=group_size, bits=4)

def install_smoothkv(calib_path: str, group_size: int = 128, bits: int = 4,
                     dtype: torch.dtype = torch.bfloat16):
    configure_kv_quant("smoothkv", group_size=group_size, bits=bits,
                       calib_path=calib_path, dtype=dtype)

def install_kivi2(group_size: int = 32, residual: int = 128):
    """KIVI-2 (residual buffer not faithfully simulated; see KIVI paper §3.3)."""
    configure_kv_quant("kivi2", group_size=group_size)
