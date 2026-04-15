"""
Read all experiment log/json files and print a consolidated results table.
Run after all experiments: python generate_report.py
"""
import json, os, re

LOGS = "logs"

EXPERIMENTS = [
    ("FP16 baseline (no quantization)",                "gsm8k_fp16",                      "gpqa_fp16"),
    ("KIVI default (per-channel key, residual=32)",    "gsm8k_kivi",                      "gpqa_kivi"),
    ("PerToken, group=32, residual=32",                "gsm8k_pertoken",                  "gpqa_pertoken"),
    ("PerToken, group=32, residual=0",                 "gsm8k_pertoken_noresidual",        "gpqa_pertoken_noresidual"),
    ("PerToken, group=128 (flat), residual=32",        "gsm8k_pertoken_flat",             "gpqa_pertoken_flat"),
    ("PerToken, group=128 (flat), residual=0",         "gsm8k_pertoken_flat_noresidual",  "gpqa_pertoken_flat_noresidual"),
]

KNOWN_GSM8K = {}
KNOWN_GPQA  = {}


def load_json(name):
    path = os.path.join(LOGS, f"{name}_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_gsm8k(data):
    if data is None:
        return None, None
    task = data.get("gsm8k", {})
    strict = task.get("exact_match,strict-match") or task.get("exact_match,strict_match")
    flex   = task.get("exact_match,flexible-extract") or task.get("exact_match,flexible_extract")
    return strict, flex


def parse_gsm8k_from_log(log_path):
    """Fallback: parse from lm-eval printed table.
    Format: |gsm8k| 3|strict-match    | 5|exact_match|0.2517|± | 0.012|
    """
    if not os.path.exists(log_path):
        return None, None
    text = open(log_path).read()
    strict = re.search(r"strict.match[^\n]*exact_match\|([\d.]+)", text)
    flex   = re.search(r"flexible.extract[^\n]*exact_match\|([\d.]+)", text)
    s = float(strict.group(1)) if strict else None
    f = float(flex.group(1))   if flex   else None
    return s, f


def extract_gpqa(data):
    if data is None:
        return None
    task = data.get("gpqa_diamond_generative_n_shot", {})
    for key in ("exact_match,flexible-extract", "exact_match,strict-match"):
        if key in task:
            return task[key]
    return None


def fmt(v):
    if v is None:
        return "pending"
    return f"{v*100:.2f}%"


print()
print("=" * 82)
print("  KIVI KV-Cache Quantization — Mistral-7B-Instruct-v0.2")
print("  2-bit (where applicable) | GSM8K 5-shot | GPQA Diamond Generative")
print("=" * 82)
print(f"  {'Config':<46} {'GSM8K strict':>12} {'GSM8K flex':>10} {'GPQA':>8}")
print("-" * 82)

pending = []
for label, gsm8k_key, gpqa_key in EXPERIMENTS:
    # GSM8K
    if gsm8k_key in KNOWN_GSM8K:
        strict, flex = KNOWN_GSM8K[gsm8k_key]
    else:
        data = load_json(gsm8k_key)
        strict, flex = extract_gsm8k(data)
        if strict is None:
            strict, flex = parse_gsm8k_from_log(os.path.join(LOGS, f"{gsm8k_key}.log"))
        if strict is None:
            pending.append(gsm8k_key)

    # GPQA
    if gpqa_key in KNOWN_GPQA:
        gpqa_acc = KNOWN_GPQA[gpqa_key]
    else:
        gpqa_acc = extract_gpqa(load_json(gpqa_key))
        if gpqa_acc is None:
            pending.append(gpqa_key)

    print(f"  {label:<46} {fmt(strict):>12} {fmt(flex):>10} {fmt(gpqa_acc):>8}")

print("-" * 82)
print()
if pending:
    print(f"Still pending ({len(pending)}): {', '.join(pending)}")
else:
    print("All experiments complete.")
print()
