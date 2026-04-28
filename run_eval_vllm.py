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
import warnings

warnings.filterwarnings("ignore")

# Patch must happen BEFORE `from vllm import LLM` loads model registries.
# Import vLLM module first to ensure LlamaAttention class exists, then patch.
import vllm  # noqa: F401
import vllm.model_executor.models.llama  # noqa: F401

from vllm_custom import patches


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv", "kivi"], required=True,
                   help="bf16/fp16 are the same no-quant baseline (dtype=auto picks the model's native dtype)")
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
    return p.parse_args()


def install_method(args):
    """Patch vLLM's LlamaAttention with the requested fake-quant hook."""
    if args.model in ("bf16", "fp16"):
        return  # no patch — unquantized baseline
    if args.model == "fp8":
        patches.install_fp8(group_size=args.group_size)
    elif args.model == "pertoken":
        patches.install_pertoken_int4(group_size=args.group_size)
    elif args.model == "smoothkv":
        assert args.calib_path, "--calib_path required for smoothkv"
        patches.install_smoothkv(args.calib_path, group_size=args.group_size, bits=args.bits)
    elif args.model == "kivi":
        patches.install_kivi2(group_size=32, residual=128)


def output_name(args):
    t = args.task
    m = args.model_path.rstrip("/").split("/")[-1].lower()
    chat = "_chat" if args.apply_chat_template else ""
    if args.model in ("bf16", "fp16"):
        return f"{t}_{m}_{args.model}{chat}_vllm"
    if args.model == "fp8":
        return f"{t}_{m}_fp8_g{args.group_size}_vllm"
    if args.model == "pertoken":
        return f"{t}_{m}_pertoken_int{args.bits}_g{args.group_size}_vllm"
    if args.model == "smoothkv":
        stem = os.path.basename(args.calib_path).replace(".pt", "")
        # strip "smoothkv_<model>_" prefix
        try:
            idx = stem.lower().index(m) + len(m)
            calib_tag = stem[idx:].lstrip("_")
        except ValueError:
            calib_tag = stem
        return f"{t}_{m}_smoothkv_g{args.group_size}_{calib_tag}_vllm"
    if args.model == "kivi":
        return f"{t}_{m}_kivi_res128_vllm"
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

    # All fake-quant paths default to eager on A100: fp8 cast fails to compile
    # (sm_80 has no fp8e4nv), pertoken hangs during cudagraph capture
    # (the Python pack loop + triton kernels don't play nicely with inductor).
    # Only the unpatched baseline uses torch.compile.
    # Override with NO_ENFORCE_EAGER=1 env to force CUDA graphs + torch.compile
    # on the quant paths too — useful with newer vllm (0.20+) where fp8 compile
    # works on sm_80, and worth probing for ~1.5-2x speedup.
    enforce_eager = (args.model not in ("bf16", "fp16")) and not os.environ.get("NO_ENFORCE_EAGER")
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

    results = simple_evaluate(
        model=lm,
        tasks=[args.task],
        batch_size=args.batch_size,
        log_samples=args.log_samples,
        gen_kwargs=gen_kwargs,
        task_manager=tm,
        limit=args.limit,
    )
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


if __name__ == "__main__":
    main()
