"""Rescore lm_eval _samples.json files (including retry-merged) per filter,
using lm_eval's actual filter classes. Avoids the heavy task.process_results
call by comparing the filter output directly against the gold target.

Usage:  python scripts/rescore_filters.py
"""
import json, os, sys, re
from pathlib import Path
from lm_eval.filters.extraction import RegexFilter, MultiChoiceRegexFilter
from lm_eval.filters.selection import TakeFirstFilter
from math_verify import parse as MV_P, verify as MV_V
import yaml

LOGS = Path("logs")
ROOT = Path("/workspace/KIVI")
TASKS_DIR = ROOT / "tasks"

class _IgnoreFunctionLoader(yaml.SafeLoader): pass
def _ignore_function(loader, node):
    return None
_IgnoreFunctionLoader.add_constructor("!function", _ignore_function)

def load_yaml_includes(path):
    """Resolve YAML includes (`include: parent.yaml`) into a single dict."""
    p = Path(path)
    with open(p) as f: y = yaml.load(f, Loader=_IgnoreFunctionLoader)
    if y.get("include"):
        parent = p.parent / y["include"]
        py = load_yaml_includes(parent)
        py.update({k: v for k, v in y.items() if k != "include"})
        y = py
    return y

def task_filter_specs(task_yaml_relpath):
    y = load_yaml_includes(TASKS_DIR / task_yaml_relpath)
    return y.get("filter_list", [])

def build_chain(steps):
    out = []
    for step in steps:
        kind = step["function"]
        a = {k: v for k, v in step.items() if k != "function"}
        if kind == "regex": out.append(RegexFilter(**a))
        elif kind == "multi_choice_regex": out.append(MultiChoiceRegexFilter(**a))
        elif kind == "take_first": out.append(TakeFirstFilter(**a))
        else: raise RuntimeError(f"unknown filter: {kind}")
    return out

def materialize(x):
    if isinstance(x, str): return x
    try: return [materialize(e) for e in x]
    except TypeError: return x

# Manual exact-match scoring (mirror lm_eval's exact_match metric defaults).
def normalize(s, ignore_case=True, ignore_punct=True, regexes_to_ignore=None):
    s = str(s)
    if regexes_to_ignore:
        for pat in regexes_to_ignore:
            s = re.sub(pat, "", s)
    if ignore_case: s = s.lower()
    if ignore_punct: s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

def gsm8k_gold(doc):
    """gsm8k gold: '... #### N' → strip to N."""
    a = doc.get("answer", "")
    m = re.search(r"####\s*([\-0-9\.,]+)", a)
    if m:
        s = m.group(1)
        for pat in [",", r"\$", r"\.$"]:
            s = re.sub(pat, "", s)
        return s.strip()
    return a.strip()

def gpqa_gold(doc):
    """gpqa gold: 'answer' field is the letter A-D string after process_docs.
    Common: doc['answer'] = '(A)' or 'A'. Try direct match."""
    a = doc.get("answer") or doc.get("Correct Answer") or ""
    a = str(a).strip()
    # Strip parens
    return a.replace("(", "").replace(")", "").strip()

def score_gsm8k(extracted, doc):
    gold = gsm8k_gold(doc)
    cand_n = normalize(extracted, regexes_to_ignore=[",", r"\$", r"(?s).*#### ", r"\.$"])
    gold_n = normalize(gold, regexes_to_ignore=[",", r"\$", r"(?s).*#### ", r"\.$"])
    return float(cand_n == gold_n)

def score_gpqa(extracted, doc):
    gold = gpqa_gold(doc)
    cand_n = normalize(extracted)
    gold_n = normalize(gold)
    # gpqa default: ignore_case=true, ignore_punctuation=true
    return float(cand_n == gold_n)

SCORERS = {
    "gsm8k_32k": score_gsm8k,
    "gpqa_main_cot_n_shot_32k": score_gpqa,
}

# Cache filter chains per task
TASK_FILTERS = {}
for task in ("gsm8k_32k", "gpqa_main_cot_n_shot_32k"):
    if task == "gsm8k_32k":
        specs = task_filter_specs("gsm8k/gsm8k_32k.yaml")
    else:
        specs = task_filter_specs("gpqa/gpqa_main_cot_n_shot_32k.yaml")
    TASK_FILTERS[task] = [(s["name"], build_chain(s["filter"])) for s in specs]

def rescore_one(task, fname):
    p = LOGS / fname
    if not p.exists(): return None
    with open(p) as f: d = json.load(f)
    items = d.get(task, []) if isinstance(d, dict) else d
    by_doc = {}; docs = {}
    for s in items:
        did = s.get("doc_id")
        if did not in by_doc:
            by_doc[did] = s; docs[did] = s.get("doc")
    out = {}
    scorer = SCORERS[task]
    for fname_, chain in TASK_FILTERS[task]:
        correct = total = 0
        for did, item in by_doc.items():
            resps = item.get("resps", [[""]])
            cand_list = resps[0] if resps and isinstance(resps[0], list) else list(resps)
            x = [list(cand_list)]
            for f in chain:
                x = materialize(f.apply(x, [docs[did]]))
            cand = x[0][0] if isinstance(x[0], list) else x[0]
            correct += scorer(cand, docs[did])
            total += 1
        out[fname_] = (correct, total)
    return out

RUNS = [
  ("bf16    | 4k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_bf16_chat_vllm_samples.json"),
  ("fp8     | 4k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_fp8_g128_chat_vllm_samples.json"),
  ("smk-NEW | 4k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_samples.json"),
  ("smk-OLD | 4k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_pair_chat_vllm_samples.json"),
  ("pert    | 4k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_samples.json"),
  ("bf16    |32k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_bf16_chat_vllm_retry32768_merged_samples.json"),
  ("fp8     |32k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_fp8_g128_chat_vllm_retry32768_merged_samples.json"),
  ("pert    |32k", "gsm8k_32k", "gsm8k_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_retry32768_merged_samples.json"),
  ("bf16    | 4k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_bf16_chat_vllm_samples.json"),
  ("fp8     | 4k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_fp8_g128_chat_vllm_samples.json"),
  ("smk-NEW | 4k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_samples.json"),
  ("smk-OLD | 4k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_pair_chat_vllm_samples.json"),
  ("pert    | 4k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_samples.json"),
  ("bf16    |32k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_bf16_chat_vllm_retry32768_merged_samples.json"),
  ("fp8     |32k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_fp8_g128_chat_vllm_retry32768_merged_samples.json"),
  ("pert    |32k", "gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_retry32768_merged_samples.json"),
  ("bf16-5shot|4k","gpqa_main_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k_5shot_qwen3-8b_bf16_chat_vllm_samples.json"),
]

print(f"{'config':18s}  {'task':30s}  {'strict':>8s}  {'flex':>8s}    n", flush=True)
print("-" * 76, flush=True)
for label, task, fname in RUNS:
    r = rescore_one(task, fname)
    if r is None:
        print(f"{label:18s}  {task:30s}  {'-':>8s}  {'-':>8s}  -", flush=True); continue
    s = r.get("strict-match"); f = r.get("flexible-extract")
    sf = f"{s[0]/s[1]:.4f}" if s else "-"
    ff = f"{f[0]/f[1]:.4f}" if f else "-"
    n = s[1] if s else (f[1] if f else 0)
    print(f"{label:18s}  {task:30s}  {sf:>8s}  {ff:>8s}  {n}", flush=True)

print("\nmath500 — math_verify boxed-aware (full 500)", flush=True)
print("-" * 50, flush=True)
M500 = [
    ("bf16    | 4k", "math500_32k_qwen3-8b_bf16_chat_vllm_samples.json"),
    ("fp8     | 4k", "math500_32k_qwen3-8b_fp8_g128_chat_vllm_samples.json"),
    ("smk-NEW | 4k", "math500_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_samples.json"),
    ("smk-OLD | 4k", "math500_32k_qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_pair_chat_vllm_samples.json"),
    ("pert    | 4k", "math500_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_samples.json"),
    ("bf16    |32k", "math500_32k_qwen3-8b_bf16_chat_vllm_retry32768_merged_samples.json"),
    ("fp8     |32k", "math500_32k_qwen3-8b_fp8_g128_chat_vllm_retry32768_merged_samples.json"),
    ("pert    |32k", "math500_32k_qwen3-8b_pertoken_int4_g128_chat_vllm_retry32768_merged_samples.json"),
]
for label, fname in M500:
    p = LOGS / fname
    if not p.exists(): print(f"{label:18s}  -", flush=True); continue
    with open(p) as f: d = json.load(f)
    items = d.get("math500_32k", []) if isinstance(d, dict) else d
    correct = total = 0
    for s in items:
        gold = s.get("doc", {}).get("answer", "")
        resps = s.get("resps", [[""]])
        raw = resps[0][0] if isinstance(resps[0], list) else resps[0]
        try:
            if MV_V(MV_P(f"\\boxed{{{gold}}}"), MV_P(raw)): correct += 1
        except Exception: pass
        total += 1
    print(f"{label:18s}  math_verify={correct/total:.4f}  ({correct}/{total})", flush=True)
