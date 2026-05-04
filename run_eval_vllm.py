"""
Evaluation using vLLM as the inference engine, with KIVI-style fake quantization
patched into LlamaAttention.forward.

Mirrors run_eval.py's CLI as closely as practical. Output filenames are
suffixed with `_vllm` to distinguish from the HFLM runs.

Caveats:
- Only one method can be active per process (the patch is global).
- KIVI-2 residual buffer is *not* faithfully simulated in vLLM — see patches.py.
  FP8/pertoken/SmoothKV variants (residual=0) are fully supported.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Patch must happen BEFORE `from vllm import LLM` loads model registries.
# Import vLLM module first to ensure LlamaAttention class exists, then patch.
import vllm  # noqa: F401
import vllm.model_executor.models.llama  # noqa: F401

from vllm.model_executor.layers.quantization.kv_fake_quant import configure_kv_quant


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv", "smoothkv_fused"], required=True,
                   help="bf16/fp16 are the same no-quant baseline (dtype=auto picks the model's native dtype). "
                        "smoothkv_fused: fold s_K into q_norm/k_norm γ (or qkv_proj rows for non-q-norm "
                        "models) at load time — zero per-step cost beyond pertoken int4.")
    p.add_argument("--task",        required=True,
                   help="e.g. truthfulqa_gen, coqa, gsm8k_32k, gpqa_diamond_cot_n_shot_32k, math500_32k")
    p.add_argument("--model_path",  default=DEFAULT_MODEL)
    p.add_argument("--group_size",  type=int, default=128)
    p.add_argument("--bits",        type=int, default=4, help="bits for pertoken/smoothkv")
    p.add_argument("--calib_path",  default=None, help="required for --model smoothkv")
    p.add_argument("--batch_size",  type=int, default=1, help="lm_eval batch_size (vLLM handles internal batching)")
    p.add_argument("--max_gen_toks", type=int, default=None)
    p.add_argument("--tp",          type=int, default=1, help="tensor parallel size")
    p.add_argument("--limit",       type=int, default=None, help="limit eval to N samples (bench/debug)")
    p.add_argument("--max_model_len", type=int, default=None,
                   help="vLLM max context. If unset, vLLM uses the model's native max_position_embeddings.")
    p.add_argument("--max_num_seqs",  type=int, default=128, help="vLLM concurrency slots")
    p.add_argument("--log_samples",   action="store_true",
                   help="Save per-item inputs/generations/targets to logs/<out_name>_samples.json "
                        "(needed for adaptive rerun of truncated items at higher max_gen_toks).")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="Wrap the task prompt with the tokenizer's chat template. "
                        "Qwen3/instruct models expect this; raw prompts underperform.")
    p.add_argument("--num_fewshot", type=int, default=None,
                   help="Override task's num_fewshot. Useful for tasks like gpqa whose "
                        "_n_shot YAML doesn't actually specify a shot count (defaults to 0).")
    return p.parse_args()


def _isolate_compile_cache():
    """Point vLLM's torch.compile cache at a per-process directory.

    Avoids two failure modes of the shared default cache:
      1. Stale-graph contamination: hash dirs collide across variants
         (e.g. bf16/gpqa and smk/gpqa both got 1d151cc93d/), so the
         second variant loads the FIRST variant's compiled FX graph
         and the runtime monkey-patch never makes it into the kernels.
      2. Race condition under concurrent launches: multiple vLLM
         processes simultaneously writing/reading triton cubin files
         produces 'Cubin file saved by TritonBundler not found'.

    Solution: VLLM_CACHE_ROOT=/tmp/vllm_cache_<pid> per process.
    Each process has its own cache; ~60-90s recompile cost per launch.
    """
    pid = os.getpid()
    per_proc_cache = f"/tmp/vllm_cache_{pid}"
    os.makedirs(per_proc_cache, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = per_proc_cache
    os.environ["VLLM_CONFIG_ROOT"] = per_proc_cache
    print(f"  [compile-cache-guard] using per-process cache {per_proc_cache}")


def install_method(args):
    """Configure vLLM's KV fake-quant scheme for the requested method."""
    # Per-PID compile cache: avoids cross-variant FX-graph hash collisions
    # AND triton-cubin races between concurrent launches.
    _isolate_compile_cache()

    if args.model in ("bf16", "fp16"):
        return  # no quant — unquantized baseline
    if args.model == "fp8":
        configure_kv_quant("fp8", group_size=args.group_size)
    elif args.model == "pertoken":
        configure_kv_quant("pertoken", group_size=args.group_size, bits=args.bits)
    elif args.model == "smoothkv":
        assert args.calib_path, "--calib_path required for smoothkv"
        configure_kv_quant("smoothkv", group_size=args.group_size, bits=args.bits,
                           calib_path=args.calib_path)
    elif args.model == "smoothkv_fused":
        # Zero-runtime-cost path: configure_kv_quant("smoothkv_fused") loads
        # calib (CPU) into the global config; vllm fork's Worker.load_model
        # post-load hook (`maybe_run_post_load_fusion`) folds s_K / s_V into
        # qkv_proj / o_proj weights once per worker. Each layer's runtime
        # method is set to "pertoken" by attach_kv_quant_to_layer.
        assert args.calib_path, "--calib_path required for smoothkv_fused"
        configure_kv_quant("smoothkv_fused", group_size=args.group_size,
                           bits=args.bits, calib_path=args.calib_path)


def output_name(args):
    t = args.task
    m = args.model_path.rstrip("/").split("/")[-1].lower()
    chat = "_chat" if args.apply_chat_template else ""
    shot = f"_{args.num_fewshot}shot" if args.num_fewshot is not None else ""
    t = t + shot
    # `chat` suffix is appended to the method portion for ALL methods so
    # chat-templated runs don't collide with non-chat runs in the filename.
    if args.model in ("bf16", "fp16"):
        return f"{t}_{m}_{args.model}{chat}_vllm"
    if args.model == "fp8":
        return f"{t}_{m}_fp8_g{args.group_size}{chat}_vllm"
    if args.model == "pertoken":
        return f"{t}_{m}_pertoken_int{args.bits}_g{args.group_size}{chat}_vllm"
    if args.model == "smoothkv":
        stem = os.path.basename(args.calib_path).replace(".pt", "")
        # strip "smoothkv_<model>_" prefix
        try:
            idx = stem.lower().index(m) + len(m)
            calib_tag = stem[idx:].lstrip("_")
        except ValueError:
            calib_tag = stem
        return f"{t}_{m}_smoothkv_g{args.group_size}_{calib_tag}{chat}_vllm"
    if args.model == "smoothkv_fused":
        stem = os.path.basename(args.calib_path).replace(".pt", "")
        try:
            idx = stem.lower().index(m) + len(m)
            calib_tag = stem[idx:].lstrip("_")
        except ValueError:
            calib_tag = stem
        return f"{t}_{m}_smoothkv_fused_g{args.group_size}_{calib_tag}{chat}_vllm"
    raise ValueError(f"unknown model {args.model}")


def main():
    args = parse_args()
    install_method(args)

    # Imports after patching so the vLLM model registry uses the patched forward
    from lm_eval import simple_evaluate, utils as lm_utils
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"))

    print(f"\n{'='*60}")
    print(f"  [vLLM] model={args.model}  task={args.task}  path={args.model_path}")
    out_name = output_name(args)
    out_path = f"logs/{out_name}_results.json"
    print(f"  output → {out_path}")
    print(f"{'='*60}\n")

    # Cudagraphs are on by default for ALL methods on vllm 0.20+ (EXAONE env)
    # — verified ~20× throughput speedup. On older vllm (e.g. 0.8.5 in the qwen3
    # env), the quant kernels' triton compile path is buggy and cudagraphs fail
    # at engine init for fp8/pertoken/smoothkv. Detect the version and force eager
    # on older vllm for non-bf16 paths. Set FORCE_ENFORCE_EAGER=1 to override
    # explicitly; set NO_ENFORCE_EAGER=1 to skip even the version probe.
    import vllm as _vllm
    _vllm_major = int(_vllm.__version__.split(".")[0])
    _vllm_minor = int(_vllm.__version__.split(".")[1])
    _is_old_vllm = (_vllm_major, _vllm_minor) < (0, 20)
    # fp8 sm_80 used to need eager (triton's fp8e4nv codegen unavailable on A100),
    # but `quant/fp8_quant.py` now has a software E4M3 rounding path that's
    # cudagraph-compatible on sm_80. So fp8 can use cudagraphs everywhere.
    if os.environ.get("FORCE_ENFORCE_EAGER"):
        enforce_eager = True
    elif os.environ.get("NO_ENFORCE_EAGER"):
        enforce_eager = False
    else:
        # Default: cudagraphs ON for vllm 0.20+, eager for older quant paths.
        enforce_eager = _is_old_vllm and (args.model not in ("bf16", "fp16"))
    vllm_kwargs = dict(
        pretrained=args.model_path,
        dtype="auto",   # respect model's native dtype (bfloat16 for Qwen3)
        tensor_parallel_size=args.tp,
        batch_size=args.batch_size,          # MUST equal max_num_seqs to saturate concurrency
        gpu_memory_utilization=0.70,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,          # 5-shot prompts share a long prefix
        disable_log_stats=False,             # emit periodic "Running/Swapped/GPU KV cache usage"
                                             # so we can verify no preemption. LLM entrypoint
                                             # defaults this to True which hides the signal.
    )
    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len
    # EXAONE-4.5 ships as Exaone4_5_ForConditionalGeneration (multimodal). The
    # nuxlear/transformers fork is missing a video processor, so vLLM's mm-budget
    # profiling crashes at engine init. We never feed images/videos for math/QA
    # tasks, so disable mm to skip profiling. (Image processor stub also required:
    # see transformers/models/exaone4_5/image_processing_exaone4_5.py.)
    if "EXAONE-4.5" in args.model_path:
        vllm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    lm = VLLM(**vllm_kwargs)

    gen_kwargs = None
    if args.max_gen_toks is not None:
        gen_kwargs = f"max_gen_toks={args.max_gen_toks}"

    se_kwargs = dict(
        model=lm,
        tasks=[args.task],
        batch_size=args.batch_size,
        log_samples=args.log_samples,
        gen_kwargs=gen_kwargs,
        task_manager=tm,
        limit=args.limit,
    )
    if args.num_fewshot is not None:
        se_kwargs["num_fewshot"] = args.num_fewshot
    if args.apply_chat_template:
        # lm_eval 0.4.5+: wrap doc_to_text in the tokenizer's chat template
        # (enable_thinking=True by default for Qwen3 — model produces <think>...</think>
        # then answer). fewshot_as_multiturn turns N-shot demos into proper
        # user/assistant turns rather than concatenating them in one user message.
        se_kwargs["apply_chat_template"] = True
        se_kwargs["fewshot_as_multiturn"] = True
    results = simple_evaluate(**se_kwargs)
    print(lm_utils.make_table(results))

    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nSaved: {out_path}")
    if args.log_samples and "samples" in results:
        samples_path = out_path.replace("_results.json", "_samples.json")
        with open(samples_path, "w") as f:
            json.dump(results["samples"], f, indent=2, default=str)
        print(f"Saved samples: {samples_path}")


    # Verify the FX graph that vLLM compiled actually contained our patched kernels.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
        from verify_compiled_graph import verify as _verify_graph
        cache_root = os.environ.get("VLLM_CACHE_ROOT") or f"/tmp/vllm_cache_{os.getpid()}"
        ok = _verify_graph(args.model, cache_root, verbose=True)
        if not ok:
            print("[WARN] GRAPH-VERIFY FAILED -- patches may have been silently bypassed!")
    except Exception as e:
        print(f"[verify-graph] could not run post-hoc check: {e}")


if __name__ == "__main__":
    main()
