"""KIVI plugin — reads config from /tmp/kivi_active.json (workers strip env vars)."""
import os
import re
import json


def install_kv_quant():
    # Resolve config path. Chain scripts write `/tmp/kivi_active_${GPUS}.json`
    # (per-pair file, where GPUS is the original CUDA_VISIBLE_DEVICES). Try that
    # first, fall back to the legacy global path.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    candidates = [
        f"/tmp/kivi_active_{visible}.json",
        "/tmp/kivi_active.json",
    ]
    cfg_path = next((p for p in candidates if os.path.exists(p)), None)
    if cfg_path is None:
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    method = cfg.get("method")
    if not method:
        return

    pid = os.getpid()
    with open("/tmp/kivi_plugin_calls.log", "a") as f:
        f.write(f"plugin called pid={pid} method={method} cfg={cfg_path}\n")

    import sys
    if "/home/home-mcl/sunghyuck/kv_cache_compression/KIVI" not in sys.path:
        sys.path.insert(0, "/home/home-mcl/sunghyuck/kv_cache_compression/KIVI")

    from vllm_custom.fake_quant_utils import (
        fake_quantize_fp8 as fp8,
        fake_quantize_k_pertoken as kpt,
        fake_quantize_v_pertoken as vpt,
    )
    from vllm.model_executor.layers.attention.attention import Attention

    gs = cfg.get("group_size", 128)
    bits = cfg.get("bits", 4)
    sk_all = sv_all = None
    if method in ("smoothkv", "smoothkv_fused"):
        import torch
        calib = torch.load(cfg["calib_path"], weights_only=True)
        sk_all = calib["s_K"].to(torch.float32)
        sv_all = calib["s_V"].to(torch.float32)

    # smoothkv_fused: absorb s_K, s_V into qkv_proj/o_proj at model load time,
    # so per-step Attention.forward only needs pertoken int4 quant. Compatible
    # with CUDA graph capture (the per-step branch is identical to "pertoken").
    if method == "smoothkv_fused":
        from vllm.v1.worker.gpu_worker import Worker
        orig_load = Worker.load_model
        _sk_all_local = sk_all
        _sv_all_local = sv_all
        def patched_load(self, *args, **kwargs):
            ret = orig_load(self, *args, **kwargs)
            import sys as _sys
            _plugin_dir = "/home/home-mcl/sunghyuck/kv_cache_compression/KIVI/kivi_vllm_plugin"
            if _plugin_dir not in _sys.path:
                _sys.path.insert(0, _plugin_dir)
            from smoothkv_fusion import fuse_smoothkv_into_model
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            n = fuse_smoothkv_into_model(
                self.model_runner.model, _sk_all_local, _sv_all_local,
                tp_rank=tp_rank, tp_size=tp_size,
            )
            with open("/tmp/kivi_plugin_calls.log", "a") as f:
                f.write(f"smoothkv_fused: fused {n} layers (tp_rank={tp_rank}, tp_size={tp_size})\n")
            return ret
        Worker.load_model = patched_load

    layer_re = re.compile(r"\.layers\.(\d+)\.")
    orig_fwd = Attention.forward

    def kivi_forward(self, query, key, value, output_shape=None):
        if method == "fp8":
            key = fp8(key, self.num_kv_heads, self.head_size, gs)
            value = fp8(value, self.num_kv_heads, self.head_size, gs)
        elif method in ("pertoken", "smoothkv_fused"):
            # smoothkv_fused absorbs the smoothing into weights at load time;
            # the per-step path is identical to plain pertoken int4.
            key = kpt(key, self.num_kv_heads, self.head_size, gs, bits=bits)
            value = vpt(value, self.num_kv_heads, self.head_size, gs, bits=bits)
        elif method == "smoothkv":
            m = layer_re.search(getattr(self, "layer_name", "") or "")
            if m is None:
                raise RuntimeError(f"SmoothKV: cannot parse layer idx from {self.layer_name}")
            li = int(m.group(1))
            sk = sk_all[li].to(key.device).to(key.dtype).reshape(-1)
            sv = sv_all[li].to(value.device).to(value.dtype).reshape(-1)
            k_s = key / sk
            k_s = kpt(k_s, self.num_kv_heads, self.head_size, gs, bits=bits)
            key = k_s * sk
            v_s = value / sv
            v_s = vpt(v_s, self.num_kv_heads, self.head_size, gs, bits=bits)
            value = v_s * sv
        return orig_fwd(self, query, key, value, output_shape)

    Attention.forward = kivi_forward
    with open("/tmp/kivi_plugin_calls.log", "a") as f:
        f.write(f"plugin INSTALLED Attention.forward pid={pid} method={method}\n")
