"""
Qwen3-aware adaptive max_gen_toks rerun.

Mirrors `adaptive_rerun.py` but installs the Qwen3 model-level KV-quant
patches (`vllm_custom.patches_qwen3.install_*`) before constructing the
vLLM LLM. This is required because the kivi_vllm_plugin path is a silent
no-op in vLLM V1 — the only working quant entry point for Qwen3 in our
env is the model-level patches loaded by `run_eval_vllm_qwen3.py`.

Input: `_samples.json` from `run_eval_vllm_qwen3.py --log_samples` at a
reduced first-pass max_gen_toks. We find truncated items, regenerate
them at the full max_gen_toks (e.g. 32768 for Qwen3 thinking mode), and
write a rerun samples file. `merge_rerun.py` then combines the two for
final scoring.

Usage:
    /opt/vllm_qwen3_env/bin/python scripts/adaptive_rerun_qwen3.py \
        --samples logs/<stem>_samples.json \
        --model_path Qwen/Qwen3-30B-A3B \
        --quant_method bf16 \
        --new_max_gen_toks 32768 \
        --tp 2 \
        --out logs/<stem>_samples_rerun32k.json
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

# Import the same arg-shape helpers as the Llama adaptive_rerun
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_rerun import _get_first_arg_pair, detect_first_pass_mg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples",          required=True)
    p.add_argument("--model_path",       required=True)
    p.add_argument("--quant_method",     required=True,
                   choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv"])
    p.add_argument("--bits",             type=int, default=4)
    p.add_argument("--group_size",       type=int, default=128)
    p.add_argument("--calib_path",       default=None,
                   help="required for --quant_method smoothkv")
    p.add_argument("--new_max_gen_toks", type=int, required=True)
    p.add_argument("--out",              required=True)
    p.add_argument("--max_model_len",    type=int, default=None)
    p.add_argument("--max_num_seqs",     type=int, default=8)
    p.add_argument("--tp",               type=int, default=2,
                   help="Qwen3-30B-A3B / Qwen3-32B want TP=2")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--trunc_slack",      type=int, default=16)
    p.add_argument("--dtype",            default="auto")
    p.add_argument("--enforce_eager",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.samples) as f:
        data = json.load(f)
    task_key = next(iter(data))
    items = data[task_key]

    # Dedup (gpqa filters double-count)
    seen = {}
    for it in items:
        seen.setdefault(it["doc_id"], it)
    items = list(seen.values())

    first_pass_mg = detect_first_pass_mg(items)
    print(f"first-pass max_gen_toks detected: {first_pass_mg}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    thresh = first_pass_mg - args.trunc_slack
    truncated = []
    for it in items:
        gen = it["resps"][0][0]
        n = len(tok(gen, add_special_tokens=False)["input_ids"])
        if n >= thresh:
            truncated.append(it)
    print(f"items total: {len(items)}  truncated (>= {thresh}): {len(truncated)}")

    if not truncated:
        print("nothing to rerun — writing empty output")
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({task_key: []}, f)
        return

    prompts = []
    stop_lists = []
    for it in truncated:
        prompt, gk = _get_first_arg_pair(it)
        prompts.append(prompt)
        stop_lists.append((gk or {}).get("until", []))
    stop = stop_lists[0] if stop_lists else []

    max_prompt_len = max(
        len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts
    )
    print(f"max prompt tokens in rerun set: {max_prompt_len}")
    mml = args.max_model_len or (max_prompt_len + args.new_max_gen_toks + 256)
    mml = ((mml + 255) // 256) * 256
    print(f"vLLM max_model_len: {mml}  max_num_seqs: {args.max_num_seqs}  MG: {args.new_max_gen_toks}")

    # Install Qwen3 model-level patches BEFORE LLM creation. This is the
    # critical step that mirrors run_eval_vllm_qwen3.py's quant-installation
    # path. Without it, --quant_method != bf16 silently runs as bf16.
    if args.quant_method in ("fp16", "bf16"):
        pass
    else:
        # Make repo root importable so vllm_custom.patches_qwen3 resolves
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from vllm_custom import patches_qwen3 as patches
        if args.quant_method == "fp8":
            patches.install_fp8(group_size=args.group_size)
        elif args.quant_method == "pertoken":
            patches.install_pertoken_int4(group_size=args.group_size, bits=args.bits)
        elif args.quant_method == "smoothkv":
            assert args.calib_path, "--calib_path required for smoothkv"
            patches.install_smoothkv(args.calib_path,
                                     group_size=args.group_size,
                                     bits=args.bits)
        print(f"installed patches_qwen3 quant method: {args.quant_method}")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
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

    rerun_items = []
    for it, o in zip(truncated, outs):
        new_gen = o.outputs[0].text
        new_it = dict(it)
        new_it["resps"] = [[new_gen]]
        new_it["filtered_resps"] = [new_gen]
        new_it["rerun_max_gen_toks"] = args.new_max_gen_toks
        new_it["rerun_gen_tokens"] = len(o.outputs[0].token_ids)
        rerun_items.append(new_it)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({task_key: rerun_items}, f, default=str)
    print(f"wrote {args.out}  ({len(rerun_items)} items)")


if __name__ == "__main__":
    main()
