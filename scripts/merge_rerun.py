"""
Merge a first-pass samples JSON with an adaptive-rerun samples JSON and
rescore with the task's filter + metric chain.

Reports: old metric (first pass, truncated), merged metric (first pass +
rerun replacements), and per-item correctness flips for inspection.

Usage:
    python scripts/merge_rerun.py \
        --original logs/<stem>_samples.json \
        --rerun    logs/<stem>_samples_rerun32k.json \
        --task     gpqa_diamond_cot_n_shot_32k
"""
import argparse
import json
import os
import re
import sys
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--rerun",    required=True)
    p.add_argument("--task",     required=True,
                   choices=["gpqa_diamond_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k",
                            "math500_32k", "gsm8k_32k"])
    p.add_argument("--out",      default=None,
                   help="merged samples JSON output. default: <original>_merged.json")
    return p.parse_args()


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    key = next(iter(data))
    return key, data[key]


# ── gpqa: use lm_eval's own filters so numbers match the result file exactly ──
# strict-match = RegexFilter(regex_pattern=r"(?<=The answer is )(.*)(?=.)", take_first)
# flexible-extract = MultiChoiceRegexFilter(regex_pattern=r"(\([A-Z]\))",
#                                           group_select=-1, ignore_case=True, take_first)

from lm_eval.filters.extraction import RegexFilter, MultiChoiceRegexFilter

_gpqa_strict = RegexFilter(
    regex_pattern=r"(?<=The answer is )(.*)(?=.)",
    group_select=0,
    fallback="[invalid]",
)
_gpqa_flex = MultiChoiceRegexFilter(
    regex_pattern=r"(\([A-Z]\))",
    group_select=-1,
    fallback="[invalid]",
    ignore_case=True,
)


def gpqa_score(item, gen):
    """Apply lm_eval's filter chain to a single (item, gen) pair."""
    doc = item["doc"]
    target = item["target"]
    # Filters expect resps=[[gen]] and docs=[doc]; they return filtered_resps=[[ans]]
    strict_ans = _gpqa_strict.apply([[gen]], [doc])[0][0]
    flex_ans   = _gpqa_flex.apply([[gen]], [doc])[0][0]
    return {
        "strict-match":     1 if strict_ans == target else 0,
        "flexible-extract": 1 if flex_ans   == target else 0,
    }


# ── math500 (delegate to task's process_results) ──

def math500_score(gen, doc):
    # Lazy import; utils.py pulls sympy
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks", "math500"))
    import utils as math_utils  # noqa
    return math_utils.process_results(doc, [gen])


# ── gsm8k (lm_eval harness gsm8k uses regex "#### <num>") ──

GSM8K_ANS_RE = re.compile(r"####\s*([\-0-9\.,]+)")


def gsm8k_score(gen, target):
    m = GSM8K_ANS_RE.search(gen)
    pred = m.group(1).replace(",", "").strip() if m else "[invalid]"
    tgt_m = GSM8K_ANS_RE.search(target) if "####" in target else None
    tgt = tgt_m.group(1).replace(",", "").strip() if tgt_m else target.replace(",", "").strip()
    return {"exact_match": 1 if pred == tgt else 0}


SCORERS = {
    "gpqa_diamond_cot_n_shot_32k":
        lambda it, gen: gpqa_score(it, gen),
    "gpqa_main_cot_n_shot_32k":
        lambda it, gen: gpqa_score(it, gen),  # same filter chain as diamond
    "math500_32k":
        lambda it, gen: math500_score(gen, it["doc"]),
    "gsm8k_32k":
        lambda it, gen: gsm8k_score(gen, it["target"]),
}


def main():
    args = parse_args()
    orig_key, orig = load_samples(args.original)
    _,       rerun = load_samples(args.rerun)

    # Dedup by doc_id (gpqa lists each doc twice due to 2 filters)
    orig_by_id = {}
    for it in orig:
        orig_by_id.setdefault(it["doc_id"], it)
    rerun_by_id = {it["doc_id"]: it for it in rerun}

    print(f"task: {args.task}")
    print(f"original items (unique docs): {len(orig_by_id)}")
    print(f"rerun items: {len(rerun_by_id)}")

    scorer = SCORERS[args.task]

    # Score each item twice: with original gen and with merged gen
    # (merged = rerun gen if available, else original gen).
    metric_keys = None
    first_pass_scores = {}    # doc_id -> {metric: 0/1}
    merged_scores = {}        # doc_id -> {metric: 0/1}
    merged_items = []         # samples to write out

    for doc_id, it in orig_by_id.items():
        orig_gen = it["resps"][0][0]
        fp = scorer(it, orig_gen)
        if metric_keys is None:
            metric_keys = list(fp.keys())
        first_pass_scores[doc_id] = fp

        rerun_it = rerun_by_id.get(doc_id)
        if rerun_it is not None:
            merged_gen = rerun_it["resps"][0][0]
            merged_it = dict(it)
            merged_it["resps"] = [[merged_gen]]
            merged_it["filtered_resps"] = [merged_gen]
            merged_it["rerun_max_gen_toks"] = rerun_it.get("rerun_max_gen_toks")
            merged_it["rerun_gen_tokens"] = rerun_it.get("rerun_gen_tokens")
        else:
            merged_gen = orig_gen
            merged_it = dict(it)
        mp = scorer(it, merged_gen)
        merged_scores[doc_id] = mp
        merged_items.append(merged_it)

    # Aggregate metrics
    n = len(orig_by_id)
    print()
    for k in metric_keys:
        fp_total = sum(s[k] for s in first_pass_scores.values()) / n
        mg_total = sum(s[k] for s in merged_scores.values()) / n
        print(f"  {k:20s}  first-pass={fp_total:.4f}  merged={mg_total:.4f}  Δ=+{mg_total-fp_total:.4f}")

    # Flip analysis on truncated items only
    truncated_ids = [doc_id for doc_id in orig_by_id if doc_id in rerun_by_id]
    n_trunc = len(truncated_ids)
    print()
    print(f"--- flips on truncated items (n={n_trunc}) ---")
    for k in metric_keys:
        flip_to_correct = sum(
            1 for d in truncated_ids
            if first_pass_scores[d][k] == 0 and merged_scores[d][k] == 1
        )
        flip_to_wrong = sum(
            1 for d in truncated_ids
            if first_pass_scores[d][k] == 1 and merged_scores[d][k] == 0
        )
        print(f"  {k:20s}  →correct: {flip_to_correct:4d}    →wrong: {flip_to_wrong:4d}    net=+{flip_to_correct-flip_to_wrong}")

    # Write merged samples
    out = args.out or args.original.replace(".json", "_merged.json")
    with open(out, "w") as f:
        json.dump({orig_key: merged_items}, f, default=str)
    print(f"\nwrote merged samples: {out}")


if __name__ == "__main__":
    main()
