#!/usr/bin/env python3
"""Pass 2: regenerate truncated pass1 samples at higher MG, merge, and re-score.

Model-agnostic — works for any vLLM model (KV fake-quant is wired into
the shared Attention class via vllm.model_executor.layers.quantization.kv_fake_quant).

Reads a pass1 _samples.json, finds items whose response hit the cap (raw_resps
token count >= MG-8), regenerates those at higher MG via vLLM, and re-scores
the merged set with task-specific scorers (no lm-eval filter chain dependency
to avoid version-skew bugs).

Scorers:
  - minerva_math500            → math_verify (sympy boxed-aware)
  - gsm8k_32k                  → strict-match: regex "answer is +?-?\\d+"
  - gpqa_main_cot_n_shot_32k   → flexible-extract: regex "\\b\\(([A-D])\\)"

Usage:
    python adaptive_pass2.py \\
        --samples logs/<task>_<model>_<mtag>_chat_vllm_samples.json \\
        --task <task> --model_path <hf-id> \\
        --pass1_mg 4096 --pass2_mg 32768 \\
        --max_model_len <ctx> --max_num_seqs 8 --tp 1 \\
        --model bf16        # | fp8 | pertoken | smoothkv_fused
        [--group_size 128] [--bits 4] [--calib_path PATH]

Writes:
    <samples_stem>_adaptive_results.json
    <samples_stem>_adaptive_merged_samples.json
"""
import argparse
import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import vllm  # noqa: F401
sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm.model_executor.layers.quantization.kv_fake_quant import configure_kv_quant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model",
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


def install_method(args):
    _isolate_compile_cache()

    if args.model in ("bf16", "fp16"):
        return
    if args.model == "fp8":
        configure_kv_quant("fp8", group_size=args.group_size)
    elif args.model == "pertoken":
        configure_kv_quant("pertoken", group_size=args.group_size, bits=args.bits)
    elif args.model == "smoothkv":
        assert args.calib_path
        configure_kv_quant("smoothkv", group_size=args.group_size, bits=args.bits,
                           calib_path=args.calib_path)
    elif args.model == "smoothkv_fused":
        assert args.calib_path
        configure_kv_quant("smoothkv_fused", group_size=args.group_size,
                           bits=args.bits, calib_path=args.calib_path)


def extract_prompt(arguments):
    """lm-eval saves arguments as [[prompt_string, gen_kwargs_dict]] for chat tasks.
    Drill in until we hit the prompt string."""
    inner = arguments
    while isinstance(inner, list) and inner:
        inner = inner[0]
    if not isinstance(inner, str):
        raise ValueError(f"could not extract prompt string from arguments: {type(inner)}")
    return inner


def get_raw_text(it):
    r = it.get("resps") or [""]
    if isinstance(r, list) and r and isinstance(r[0], list):
        r = r[0]
    if isinstance(r, list):
        return r[0] if r else ""
    return r if isinstance(r, str) else ""


def find_truncated(items, pass1_mg, tokenizer):
    out = []
    for i, it in enumerate(items):
        n = len(tokenizer.encode(get_raw_text(it), add_special_tokens=False))
        if n >= pass1_mg - 8:
            out.append(i)
    return out


# ---------- task-specific scorers ----------

def score_minerva_math500(items):
    from math_verify import parse, verify
    n_correct = 0
    for it in items:
        resp = get_raw_text(it)
        sol = it["doc"].get("solution", "")
        try:
            ok = bool(verify(gold=parse(sol), target=parse(resp)))
        except Exception:
            ok = False
        if ok:
            n_correct += 1
    return {"math_verify,none": n_correct / len(items),
            "math_verify_n,none": len(items)}


_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_strict_gsm8k(text: str):
    """Strict match: 'answer is' or last \\boxed{} or final number after #### / Final Answer:"""
    # Try \\boxed{}
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        nums = _NUM_RE.findall(m[-1])
        if nums:
            return nums[-1].replace(",", "")
    # Try "The answer is X"
    m = re.search(r"answer is[:\s]*\$?(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Try GSM8K-style "#### X"
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def _extract_flex_gsm8k(text: str):
    """Flexible: last number in text."""
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def score_gsm8k(items):
    n_strict = 0
    n_flex = 0
    for it in items:
        resp = get_raw_text(it)
        gold = it["doc"].get("answer", "")
        # GSM8K gold is "... #### N"
        g = _NUM_RE.findall(gold)
        gold_num = g[-1].replace(",", "") if g else gold.strip()
        s = _extract_strict_gsm8k(resp)
        f = _extract_flex_gsm8k(resp)
        if s is not None and s == gold_num:
            n_strict += 1
        if f is not None and f == gold_num:
            n_flex += 1
    return {"exact_match,strict-match": n_strict / len(items),
            "exact_match,flexible-extract": n_flex / len(items),
            "exact_match_n,strict-match": len(items),
            "exact_match_n,flexible-extract": len(items)}


def score_gpqa(items):
    n_flex = 0
    for it in items:
        resp = get_raw_text(it)
        gold = str(it["doc"].get("answer", "") or it["doc"].get("Correct Answer", "") or "").strip()
        # gpqa_main_cot_n_shot answer is one of "(A)" "(B)" "(C)" "(D)"
        m = re.findall(r"\b\(?([A-D])\)?", resp.split("\n")[-1])
        if not m:
            m = re.findall(r"\b\(([A-D])\)", resp)
        pred = m[-1] if m else None
        gold_letter = gold[1] if gold.startswith("(") and len(gold) >= 3 else gold
        if pred is not None and pred.upper() == gold_letter.upper():
            n_flex += 1
    return {"exact_match,flexible-extract": n_flex / len(items),
            "exact_match_n,flexible-extract": len(items)}


SCORERS = {
    "minerva_math500": score_minerva_math500,
    "gsm8k_32k": score_gsm8k,
    "gsm8k_cot": score_gsm8k,
    "gpqa_main_cot_n_shot_32k": score_gpqa,
}


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
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)

    truncated_idx = find_truncated(items, args.pass1_mg, tok)
    print(f"[pass2] {len(truncated_idx)}/{len(items)} truncated at MG={args.pass1_mg} "
          f"({100*len(truncated_idx)/len(items):.1f}%)")

    if truncated_idx:
        install_method(args)
        from vllm import LLM, SamplingParams

        prompts = [extract_prompt(items[i]["arguments"]) for i in truncated_idx]
        print(f"[pass2] launching vLLM (model={args.model}, MG={args.pass2_mg}) on "
              f"{len(prompts)} prompts; first prompt ends with: {repr(prompts[0][-100:])}")
        llm = LLM(model=args.model_path, dtype="auto",
                  tensor_parallel_size=args.tp,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_seqs=args.max_num_seqs,
                  max_model_len=args.max_model_len,
                  enforce_eager=False,
                  enable_prefix_caching=True,
                  seed=1234)
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
        raise ValueError(f"no scorer for task {task_key}; add one in SCORERS")
    final = SCORERS[task_key](merged)
    print(f"[pass2] final scores on N={len(merged)}:")
    for k, v in sorted(final.items()):
        if not k.endswith("_n,strict-match") and not k.endswith("_n,flexible-extract") and not k.endswith("_n,none"):
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
        sys.path.insert(0, str(Path(__file__).parent))
        from verify_compiled_graph import verify as _verify_graph
        cache_root = os.environ.get("VLLM_CACHE_ROOT") or f"/tmp/vllm_cache_{os.getpid()}"
        ok = _verify_graph(args.model, cache_root, verbose=True)
        if not ok:
            print("[WARN] GRAPH-VERIFY FAILED -- patches may have been silently bypassed!")
    except Exception as e:
        print(f"[verify-graph] could not run post-hoc check: {e}")


if __name__ == "__main__":
    main()
