#!/usr/bin/env python3
"""Pass 2: regenerate truncated pass1 samples at higher MG, merge, and re-score.

Model-agnostic — works for any vLLM model (KV fake-quant is wired into
the shared Attention class via vllm.model_executor.layers.quantization.kv_fake_quant).

Reads a pass1 _samples.json, finds items whose response hit the cap (raw_resps
token count >= MG-8), regenerates those at higher MG via vLLM, and re-scores
the merged set with task-specific scorers from ``scoring.py`` (no lm-eval
filter chain dependency to avoid version-skew bugs).

Usage:
    python adaptive_pass2.py \\
        --samples logs/<task>_<model>_<mtag>_chat_vllm_samples.json \\
        --task <task> --model <hf-id> \\
        --pass1_mg 4096 --pass2_mg 32768 \\
        --max_model_len <ctx> --max_num_seqs 8 --tp 1 \\
        --kv_quant_method bf16   # | fp8 | pertoken | smoothkv_fused
        [--group_size 128] [--bits 4] [--calib_path PATH]

Writes:
    <samples_stem>_adaptive_results.json
    <samples_stem>_adaptive_merged_samples.json
"""
import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import vllm  # noqa: F401
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from vllm.config import KVCacheQuantConfig

from scoring import SCORERS, get_raw_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True,
                   help="HF model path or hub id (e.g. Qwen/Qwen3-8B)")
    p.add_argument("--kv_quant_method", "--kvq", dest="kv_quant_method",
                   choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv", "smoothkv_fused"],
                   required=True)
    p.add_argument("--calib_path", default=None)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--pass1_mg", type=int, required=True)
    p.add_argument("--pass2_mg", type=int, default=32768)
    p.add_argument("--max_model_len", type=int, required=True)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


def _isolate_compile_cache():
    """Per-PID VLLM_CACHE_ROOT to avoid cross-variant graph-hash collisions
    AND triton-cubin races between concurrent launches. See
    run_eval_vllm.py:_isolate_compile_cache() for full rationale."""
    pid = os.getpid()
    per_proc_cache = f"/tmp/vllm_cache_{pid}"
    os.makedirs(per_proc_cache, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = per_proc_cache
    os.environ["VLLM_CONFIG_ROOT"] = per_proc_cache
    print(f"  [compile-cache-guard] using per-process cache {per_proc_cache}")


def build_kv_quant_config(args):
    """Return a KVCacheQuantConfig for the requested method, or None for
    bf16/fp16. Also sets a per-PID compile cache to avoid cross-variant
    FX-graph hash collisions."""
    _isolate_compile_cache()

    if args.kv_quant_method in ("bf16", "fp16"):
        return None
    if args.kv_quant_method == "fp8":
        return KVCacheQuantConfig(method="fp8", group_size=args.group_size)
    if args.kv_quant_method == "pertoken":
        return KVCacheQuantConfig(method="pertoken", group_size=args.group_size,
                                  bits=args.bits)
    if args.kv_quant_method == "smoothkv":
        assert args.calib_path
        return KVCacheQuantConfig(method="smoothkv", group_size=args.group_size,
                                  bits=args.bits, calib_path=args.calib_path)
    if args.kv_quant_method == "smoothkv_fused":
        assert args.calib_path
        return KVCacheQuantConfig(method="smoothkv_fused",
                                  group_size=args.group_size, bits=args.bits,
                                  calib_path=args.calib_path)
    raise ValueError(f"unknown --kv_quant_method {args.kv_quant_method}")


def extract_prompt(arguments):
    """lm-eval saves arguments as [[prompt_string, gen_kwargs_dict]] for chat tasks.
    Drill in until we hit the prompt string."""
    inner = arguments
    while isinstance(inner, list) and inner:
        inner = inner[0]
    if not isinstance(inner, str):
        raise ValueError(f"could not extract prompt string from arguments: {type(inner)}")
    return inner


def find_truncated(items, pass1_mg, tokenizer):
    """Return indices of items whose response hit the pass1 token cap (MG - 8)."""
    out = []
    for i, it in enumerate(items):
        n = len(tokenizer.encode(get_raw_text(it), add_special_tokens=False))
        if n >= pass1_mg - 8:
            out.append(i)
    return out


def main():
    args = parse_args()

    samples_path = Path(args.samples)
    raw = json.load(open(samples_path))
    if isinstance(raw, dict):
        task_key = next(iter(raw))
        items = raw[task_key]
    else:
        task_key = args.task
        items = raw
    print(f"[pass2] loaded {len(items)} items for task={task_key}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    truncated_idx = find_truncated(items, args.pass1_mg, tok)
    print(f"[pass2] {len(truncated_idx)}/{len(items)} truncated at MG={args.pass1_mg} "
          f"({100*len(truncated_idx)/len(items):.1f}%)")

    if truncated_idx:
        kv_quant_cfg = build_kv_quant_config(args)
        from vllm import LLM, SamplingParams

        prompts = [extract_prompt(items[i]["arguments"]) for i in truncated_idx]
        print(f"[pass2] launching vLLM (kv_quant_method={args.kv_quant_method}, MG={args.pass2_mg}) "
              f"on {len(prompts)} prompts; first prompt ends with: {repr(prompts[0][-100:])}")
        llm_kwargs = dict(
            model=args.model, dtype="auto",
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            enforce_eager=False,
            enable_prefix_caching=True,
            seed=1234,
        )
        if kv_quant_cfg is not None:
            llm_kwargs["kv_cache_quant_config"] = kv_quant_cfg
        llm = LLM(**llm_kwargs)
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.pass2_mg)
        outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
        new_texts = [o.outputs[0].text for o in outputs]
        n_empty = sum(1 for t in new_texts if not t.strip())
        print(f"[pass2] regen done; {n_empty}/{len(new_texts)} empty responses")
    else:
        new_texts = []

    merged = copy.deepcopy(items)
    for idx, txt in zip(truncated_idx, new_texts):
        r = merged[idx].get("resps")
        if isinstance(r, list) and r and isinstance(r[0], list):
            merged[idx]["resps"] = [[txt]]
        else:
            merged[idx]["resps"] = [txt]

    if task_key not in SCORERS:
        raise ValueError(f"no scorer for task {task_key}; add one in scripts/scoring.py")
    final = SCORERS[task_key](merged)
    print(f"[pass2] final scores on N={len(merged)}:")
    for k, v in sorted(final.items()):
        if not k.endswith(("_n,strict-match", "_n,flexible-extract", "_n,none")):
            print(f"  {k}: {100*v:.2f}")

    stem = str(samples_path).replace("_samples.json", "")
    out_samples = f"{stem}_adaptive_merged_samples.json"
    out_results = f"{stem}_adaptive_results.json"
    json.dump({task_key: merged}, open(out_samples, "w"))
    json.dump({"results": {task_key: final},
               "n_truncated_pass1": len(truncated_idx),
               "n_total": len(items),
               "pass1_mg": args.pass1_mg,
               "pass2_mg": args.pass2_mg}, open(out_results, "w"), indent=2)
    print(f"[pass2] wrote {out_samples}")
    print(f"[pass2] wrote {out_results}")

    # Verify the FX graph that vLLM compiled actually contained our patched kernels.
    try:
        from verify_compiled_graph import verify as _verify_graph
        cache_root = os.environ.get("VLLM_CACHE_ROOT") or f"/tmp/vllm_cache_{os.getpid()}"
        ok = _verify_graph(args.kv_quant_method, cache_root, verbose=True)
        if not ok:
            print("[WARN] GRAPH-VERIFY FAILED -- patches may have been silently bypassed!")
    except Exception as e:
        print(f"[verify-graph] could not run post-hoc check: {e}")


if __name__ == "__main__":
    main()
