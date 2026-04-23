"""
Generate an HTML report for the vLLM SmoothKV sweep on Llama-3-8B-Instruct.

Reads logs/{task}_meta-llama-3-8b-instruct_{method}_vllm_results.json
and writes a self-contained HTML file.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")

MODEL_STEM = "meta-llama-3-8b-instruct"
TASKS = [
    ("gsm8k_32k",                   "gsm8k (strict)"),
    ("gpqa_diamond_cot_n_shot_32k", "gpqa diamond CoT (flex)"),
    ("math500_32k",                 "math500 (exact)"),
]

METHOD_ORDER = [
    ("fp16",                                        "FP16 baseline",              "baseline"),
    ("fp8_g128",                                    "FP8 E4M3 g=128",             "fp8"),
    ("pertoken_int4_g128",                          "pertoken INT4 g=128",        "pertoken"),
    ("smoothkv_g128_puremax_a1b1_pair",             "SmoothKV puremax α=1 β=1",   "smoothkv"),
    ("smoothkv_g128_pairK99p9_pV99p9",              "SmoothKV pair K/V p99.9",    "smoothkv"),
    ("smoothkv_g128_kOnly_a1_pair",                 "SmoothKV K-only α=1",        "smoothkv-kOnly"),
    ("smoothkv_g128_kOnly_p99p9_pair",              "SmoothKV K-only p99.9",      "smoothkv-kOnly"),
    ("smoothkv_g128_kOnly_a0.5_noQ_pair",           "SmoothKV K-only α=0.5 noQ",  "smoothkv-kOnly"),
]

PRIMARY_METRIC = {
    "gsm8k_32k":                   ("exact_match,strict-match",      "exact_match_stderr,strict-match"),
    "gpqa_diamond_cot_n_shot_32k": ("exact_match,flexible-extract",  "exact_match_stderr,flexible-extract"),
    "math500_32k":                 ("exact_match,none",              "exact_match_stderr,none"),
}


def load_result(task, method):
    path = os.path.join(LOGS, f"{task}_{MODEL_STEM}_{method}_vllm_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # top-level key is the task name
    tkey = next(iter(d))
    metrics = d[tkey]
    primary_key, stderr_key = PRIMARY_METRIC[task]
    val = metrics.get(primary_key)
    err = metrics.get(stderr_key)
    return {"val": val, "err": err, "all": metrics}


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def rel_to_fp16(v, base):
    if v is None or base is None or base == 0:
        return ""
    delta = (v - base) * 100  # in pp
    cls = "pos" if delta > 0 else ("neg" if delta < -0.5 else "flat")
    sign = "+" if delta >= 0 else ""
    return f'<span class="delta {cls}">{sign}{delta:.2f}pp</span>'


def family_color(fam):
    return {
        "baseline":       "#e8eef7",
        "fp8":            "#eaf7ea",
        "pertoken":       "#fff3d6",
        "smoothkv":       "#f7e8ef",
        "smoothkv-kOnly": "#f1e0ff",
    }.get(fam, "#ffffff")


def build_html(out_path):
    # Collect data
    table = {}  # method -> task -> {val, err}
    for method, _, _ in METHOD_ORDER:
        table[method] = {}
        for task, _ in TASKS:
            table[method][task] = load_result(task, method)

    # fp16 baselines per task
    fp16 = {task: (table["fp16"][task]["val"] if table["fp16"][task] else None)
            for task, _ in TASKS}

    # Best per task among quant methods (excl fp16)
    best_per_task = {}
    for task, _ in TASKS:
        best_val, best_method = None, None
        for method, _, _ in METHOD_ORDER:
            if method == "fp16":
                continue
            r = table[method][task]
            if r and r["val"] is not None and (best_val is None or r["val"] > best_val):
                best_val, best_method = r["val"], method
        best_per_task[task] = best_method

    # HTML
    rows_html = []
    for method, label, fam in METHOD_ORDER:
        color = family_color(fam)
        cells = []
        for task, _ in TASKS:
            r = table[method][task]
            if r is None:
                cells.append('<td class="val missing">—</td>')
                continue
            val = r["val"]
            err = r["err"]
            delta = rel_to_fp16(val, fp16[task]) if method != "fp16" else ""
            best_mark = " ★" if best_per_task[task] == method else ""
            err_str = f' <span class="err">±{err*100:.1f}</span>' if err is not None else ""
            cells.append(
                f'<td class="val">'
                f'<div class="num">{fmt(val)}{best_mark}</div>'
                f'<div class="meta">{delta}{err_str}</div>'
                f'</td>'
            )
        rows_html.append(
            f'<tr style="background:{color}">'
            f'<td class="method"><b>{label}</b><div class="stem">{method}</div></td>'
            + "".join(cells)
            + '</tr>'
        )

    # Per-task detail tables: dump all metrics (strict, flex, stderrs)
    detail_html = []
    for task, task_label in TASKS:
        all_keys = set()
        for method, _, _ in METHOD_ORDER:
            r = table[method][task]
            if r:
                all_keys.update(r["all"].keys())
        all_keys.discard("alias")
        key_list = sorted(all_keys)
        if not key_list:
            continue
        header = "<tr><th>Method</th>" + "".join(f"<th>{k}</th>" for k in key_list) + "</tr>"
        body = ""
        for method, label, fam in METHOD_ORDER:
            r = table[method][task]
            color = family_color(fam)
            if not r:
                body += f'<tr style="background:{color}"><td>{label}</td>' + "<td>—</td>"*len(key_list) + "</tr>"
                continue
            cells = []
            for k in key_list:
                v = r["all"].get(k)
                if isinstance(v, float):
                    cells.append(f"<td>{v:.4f}</td>")
                elif v is None:
                    cells.append("<td>—</td>")
                else:
                    cells.append(f"<td>{v}</td>")
            body += f'<tr style="background:{color}"><td><b>{label}</b></td>' + "".join(cells) + "</tr>"
        detail_html.append(
            f'<h3>{task_label} <span class="task-stem">({task})</span></h3>'
            f'<table class="detail"><thead>{header}</thead><tbody>{body}</tbody></table>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vLLM SmoothKV sweep — Llama-3-8B-Instruct</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 30px auto; max-width: 1150px; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ margin-top: 32px; font-size: 18px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ margin-top: 24px; font-size: 15px; }}
  .task-stem {{ color: #888; font-weight: normal; font-size: 12px; }}
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
  table.detail th, table.detail td {{ font-size: 11.5px; padding: 5px 7px; }}
  .legend {{ font-size: 12px; color: #555; margin: 10px 0; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: middle; border: 1px solid #999; }}
  .caption {{ font-size: 13px; color: #555; margin: 6px 0 14px 0; }}
</style>
</head>
<body>

<h1>vLLM SmoothKV sweep — Llama-3-8B-Instruct</h1>
<p class="caption">
  Backend: vLLM 0.6.6 + CUDA graphs, max_model_len=8192, batch_size=128, prefix caching on.
  All quant methods use group_size=128. SmoothKV scales are pair-equal on s_K (RoPE-mergeable form).
  <b>★</b> = best quant method per task. Deltas are percentage-point vs FP16 baseline.
</p>

<div class="legend">
  Families:
  <span class="swatch" style="background:#e8eef7"></span>baseline
  <span class="swatch" style="background:#eaf7ea"></span>fp8
  <span class="swatch" style="background:#fff3d6"></span>pertoken
  <span class="swatch" style="background:#f7e8ef"></span>SmoothKV (K+V smoothed)
  <span class="swatch" style="background:#f1e0ff"></span>SmoothKV (K-only)
</div>

<h2>Summary</h2>
<table>
  <thead>
    <tr>
      <th>Method</th>
      {"".join(f'<th>{label}</th>' for _, label in TASKS)}
    </tr>
  </thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>

<h2>Full metrics per task</h2>
{"".join(detail_html)}

</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="logs/report_vllm_smoothkv_llama3.html")
    args = ap.parse_args()
    build_html(args.output)
