"""
Auto-discovering HTML report over all vLLM eval results.

Groups result files by model stem, then by method (fp16 / fp8 / pertoken /
smoothkv-variant). Within each model section, shows a table of methods × tasks
with primary metrics, deltas vs FP16 baseline, and best-quant-per-task stars.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

LOGS = "logs"

TASKS = [
    ("gsm8k_32k",                   "gsm8k (strict)"),
    ("gpqa_diamond_cot_n_shot_32k", "gpqa diamond CoT (flex)"),
    ("math500_32k",                 "math500 (exact)"),
]

PRIMARY_METRIC = {
    "gsm8k_32k":                   ("exact_match,strict-match",     "exact_match_stderr,strict-match"),
    "gpqa_diamond_cot_n_shot_32k": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
    "math500_32k":                 ("exact_match,none",             "exact_match_stderr,none"),
}

# Known model stems → display name (extend as you add new models)
MODEL_DISPLAY = {
    "meta-llama-3-8b-instruct":         "Meta-Llama-3-8B-Instruct",
    "deepseek-r1-distill-llama-8b":     "DeepSeek-R1-Distill-Llama-8B",
    "mistral-7b-instruct-v0.2":         "Mistral-7B-Instruct-v0.2",
    "qwen2.5-7b-instruct":              "Qwen2.5-7B-Instruct",
}


def parse_filename(path):
    """Parse '<task>_<model>_<method>_vllm_results.json' into components."""
    name = os.path.basename(path).replace("_vllm_results.json", "")
    for model_stem in MODEL_DISPLAY:
        m = re.match(rf"^(?P<task>.+?)_{re.escape(model_stem)}_(?P<method>.+)$", name)
        if m:
            return m.group("task"), model_stem, m.group("method")
    return None


def load_score(path, task):
    with open(path) as f:
        d = json.load(f)
    tkey = next(iter(d))
    metrics = d[tkey]
    pk, sk = PRIMARY_METRIC.get(task, (None, None))
    return metrics.get(pk), metrics.get(sk)


def method_family(method):
    """Classify method stems for grouping/coloring."""
    if method == "fp16": return "baseline"
    if method.startswith("fp8"): return "fp8"
    if method.startswith("pertoken"): return "pertoken"
    if "kOnly" in method: return "smoothkv-kOnly"
    if method.startswith("smoothkv"): return "smoothkv"
    return "other"


def family_color(fam):
    return {
        "baseline":       "#e8eef7",
        "fp8":            "#eaf7ea",
        "pertoken":       "#fff3d6",
        "smoothkv":       "#f7e8ef",
        "smoothkv-kOnly": "#f1e0ff",
        "other":          "#ffffff",
    }.get(fam, "#ffffff")


def method_display(method):
    """Prettify method stem."""
    # strip "smoothkv_g128_" prefix
    s = re.sub(r"^smoothkv_g\d+_", "smoothkv ", method)
    return s


def build_html(out_path):
    # Collect: table[model][method][task] = (val, err, path)
    table = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(f"{LOGS}/*_vllm_results.json")):
        parsed = parse_filename(path)
        if parsed is None: continue
        task, model, method = parsed
        if task not in PRIMARY_METRIC: continue
        val, err = load_score(path, task)
        table[model][method][task] = (val, err, path)

    if not table:
        print("No vLLM result files found under logs/")
        return

    # Build model sections
    sections = []
    for model in sorted(table):
        disp = MODEL_DISPLAY.get(model, model)
        methods = sorted(table[model].keys(),
                         key=lambda m: (method_family(m) != "baseline",
                                        method_family(m),
                                        m))
        fp16 = {t: table[model].get("fp16", {}).get(t, (None,)*3)[0] for t, _ in TASKS}

        # best per task among non-fp16
        best_per_task = {}
        for t, _ in TASKS:
            best_v, best_m = None, None
            for m in methods:
                if m == "fp16": continue
                v = table[model][m].get(t, (None,)*3)[0]
                if v is not None and (best_v is None or v > best_v):
                    best_v, best_m = v, m
            best_per_task[t] = best_m

        rows = []
        for m in methods:
            fam = method_family(m)
            color = family_color(fam)
            cells = []
            for task, _ in TASKS:
                v, e, _p = table[model][m].get(task, (None, None, None))
                if v is None:
                    cells.append('<td class="val missing">—</td>')
                    continue
                base = fp16[task]
                if m != "fp16" and base is not None:
                    d = (v - base) * 100
                    cls = "pos" if d > 0.2 else ("neg" if d < -0.5 else "flat")
                    sign = "+" if d >= 0 else ""
                    delta = f'<span class="delta {cls}">{sign}{d:.2f}pp</span>'
                else:
                    delta = ""
                err_s = f' <span class="err">±{e*100:.1f}</span>' if e is not None else ""
                star = " ★" if best_per_task[task] == m else ""
                cells.append(
                    f'<td class="val">'
                    f'<div class="num">{v:.4f}{star}</div>'
                    f'<div class="meta">{delta}{err_s}</div></td>'
                )
            rows.append(
                f'<tr style="background:{color}">'
                f'<td class="method"><b>{method_display(m)}</b>'
                f'<div class="stem">{m}</div></td>'
                + "".join(cells) + '</tr>'
            )

        n_results = sum(1 for m in methods for t, _ in TASKS
                        if table[model][m].get(t, (None,)*3)[0] is not None)
        sections.append(f"""
<h2>{disp} <span class="meta-count">({n_results} results across {len(methods)} methods)</span></h2>
<table>
  <thead>
    <tr>
      <th>Method</th>
      {"".join(f'<th>{label}</th>' for _, label in TASKS)}
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vLLM KV-quant sweep — all models</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 30px auto; max-width: 1200px; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ margin-top: 32px; font-size: 18px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .meta-count {{ color: #888; font-size: 12px; font-weight: normal; }}
  table {{ border-collapse: collapse; margin: 10px 0 20px 0; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f4f4f4; font-weight: 600; }}
  td.val {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.val .num {{ font-size: 14px; font-weight: 600; }}
  td.val .meta {{ font-size: 11px; color: #555; margin-top: 2px; }}
  td.method {{ text-align: left; }}
  td.method .stem {{ font-size: 10.5px; color: #888; font-family: ui-monospace, monospace; }}
  .delta {{ font-weight: 600; }}
  .delta.pos {{ color: #197a31; }}
  .delta.neg {{ color: #a82828; }}
  .delta.flat {{ color: #666; }}
  .err {{ color: #888; }}
  .missing {{ color: #bbb; }}
  .caption {{ font-size: 13px; color: #555; margin: 6px 0 14px 0; }}
  .legend {{ font-size: 12px; color: #555; margin: 10px 0; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px;
             vertical-align: middle; border: 1px solid #999; }}
</style>
</head>
<body>

<h1>vLLM KV-cache quantization sweep</h1>
<p class="caption">
  Backend: vLLM 0.6.6 + CUDA graphs + prefix caching, batch_size = max_num_seqs,
  max_model_len sized per model (see <code>scripts/launch_vllm_reasoning.sh</code>).
  SmoothKV scales are pair-equal on s_K (RoPE-mergeable). <b>★</b> = best quant
  method per task. Deltas = percentage-points vs FP16 baseline.
</p>

<div class="legend">
  Families:
  <span class="swatch" style="background:#e8eef7"></span>baseline
  <span class="swatch" style="background:#eaf7ea"></span>fp8
  <span class="swatch" style="background:#fff3d6"></span>pertoken
  <span class="swatch" style="background:#f7e8ef"></span>SmoothKV (K+V)
  <span class="swatch" style="background:#f1e0ff"></span>SmoothKV (K-only, V=1)
</div>

{"".join(sections)}

</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")
    # Also print a quick plaintext summary
    for model in sorted(table):
        n_methods = len(table[model])
        n_results = sum(1 for m in table[model] for t, _ in TASKS
                        if table[model][m].get(t, (None,)*3)[0] is not None)
        print(f"  {MODEL_DISPLAY.get(model, model):40s}  {n_methods} methods  {n_results} cells filled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="logs/report_vllm_all.html")
    args = ap.parse_args()
    build_html(args.output)
