"""
Adaptive max_gen_toks rerun.

Input: a samples JSON produced by `run_eval_vllm.py --log_samples` at a reduced
max_gen_toks (first pass). This script finds items whose generation hit the
cap, regenerates them at the full max_gen_toks (second pass), and writes a
rerun samples file. A separate merge step combines the two for final scoring.

The heavy lift is vLLM `LLM.generate()` — no lm_eval involvement. Re-evaluating
the merged samples via lm_eval's filter chain is handled by `merge_rerun.py`.

Usage:
    CUDA_VISIBLE_DEVICES=3 /opt/vllm_env/bin/python scripts/adaptive_rerun.py \
        --samples logs/<stem>_samples.json \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --new_max_gen_toks 32768 \
        --out logs/<stem>_samples_rerun32k.json
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples",          required=True,
                   help="path to *_samples.json from first pass")
    p.add_argument("--model_path",       required=True)
    p.add_argument("--new_max_gen_toks", type=int, required=True,
                   help="target max_gen_toks for rerun (e.g. 32768)")
    p.add_argument("--out",              required=True)
    p.add_argument("--max_model_len",    type=int, default=None,
                   help="vLLM max_model_len. Default: new_max_gen_toks + max prompt + 256")
    p.add_argument("--max_num_seqs",     type=int, default=8)
    p.add_argument("--tp",               type=int, default=1,
                   help="tensor_parallel_size for vLLM. Use 2+ for ≥30B BF16 models.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    p.add_argument("--trunc_slack",      type=int, default=16,
                   help="items with gen tokens >= (first_pass_MG - slack) count as truncated")
    p.add_argument("--dtype",            default="auto",
                   help="vLLM dtype. 'auto' picks model-native (bf16 for bf16-native models).")
    p.add_argument("--enforce_eager",    action="store_true",
                   help="Force vLLM to enforce_eager=True (no CUDA graph capture). "
                        "Use when adaptive_rerun follows a SmoothKV plugin pass1 — "
                        "graph state from the plugin can corrupt fresh capture.")
    return p.parse_args()


def detect_first_pass_mg(items):
    """Pull max_gen_toks from the first item's stored gen_kwargs."""
    for it in items:
        args = it.get("arguments", [])
        if args and len(args[0]) > 1 and isinstance(args[0][1], dict):
            mg = args[0][1].get("max_gen_toks")
            if mg:
                return mg
    raise ValueError("could not detect first-pass max_gen_toks from samples")


def main():
    args = parse_args()
    with open(args.samples) as f:
        data = json.load(f)
    task_key = next(iter(data))
    items = data[task_key]

    # Dedup by doc_id (gpqa double-counts due to 2 filters)
    seen = {}
    for it in items:
        seen.setdefault(it["doc_id"], it)
    items = list(seen.values())

    first_pass_mg = detect_first_pass_mg(items)
    print(f"first-pass max_gen_toks detected: {first_pass_mg}")

    # Count tokens in each gen, flag truncated
    tok = AutoTokenizer.from_pretrained(args.model_path)
    thresh = first_pass_mg - args.trunc_slack
    truncated = []
    for it in items:
        gen = it["resps"][0][0]
        n = len(tok(gen, add_special_tokens=False)["input_ids"])
        if n >= thresh:
            truncated.append(it)

    print(f"items total: {len(items)}")
    print(f"truncated (>= {thresh} gen tokens): {len(truncated)}")
    if not truncated:
        print("nothing to rerun — writing empty output")
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({task_key: []}, f)
        return

    # Extract prompts + stop tokens
    prompts = [it["arguments"][0][0] for it in truncated]
    stop_lists = [it["arguments"][0][1].get("until", []) for it in truncated]
    # All items should share the same `until` within a task; use the first
    stop = stop_lists[0] if stop_lists else []

    # Defer vLLM import until after we've parsed everything (so import errors
    # happen after truncation count is printed).
    from vllm import LLM, SamplingParams

    max_prompt_len = max(
        len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts
    )
    print(f"max prompt tokens in rerun set: {max_prompt_len}")
    mml = args.max_model_len or (max_prompt_len + args.new_max_gen_toks + 256)
    mml = ((mml + 255) // 256) * 256
    print(f"vLLM max_model_len: {mml}  max_num_seqs: {args.max_num_seqs}  MG: {args.new_max_gen_toks}")

    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,                    # "auto" preserves model-native (bf16 for DSR1/Mistral/Llama-3/Qwen3)
        tensor_parallel_size=args.tp,
        max_model_len=mml,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        disable_log_stats=False,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.new_max_gen_toks,
        stop=stop or None,
    )
    outs = llm.generate(prompts, sp)

    # Rewrite each truncated item's resps with the new (longer) gen.
    rerun_items = []
    for it, o in zip(truncated, outs):
        new_gen = o.outputs[0].text
        new_it = dict(it)
        new_it["resps"] = [[new_gen]]
        new_it["filtered_resps"] = [new_gen]  # filter re-applied in merge
        new_it["rerun_max_gen_toks"] = args.new_max_gen_toks
        new_it["rerun_gen_tokens"] = len(o.outputs[0].token_ids)
        rerun_items.append(new_it)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({task_key: rerun_items}, f, default=str)
    print(f"wrote {args.out}  ({len(rerun_items)} items)")


if __name__ == "__main__":
    main()
