"""Structured 3-table ablation HTML report for the SmoothKV puremax sweep on Llama3."""
import argparse
import glob
import json
import os
import re

TASKS_DISPLAY = [
    ("gsm8k_32k",                   "gsm8k (strict)"),
    ("gpqa_diamond_cot_n_shot_32k", "gpqa diamond CoT"),
    ("math500_32k",                 "math500 (exact)"),
]
TASK_KEYS = {
    "gsm8k_32k":                   ("exact_match,strict-match",     "exact_match_stderr,strict-match"),
    "gpqa_diamond_cot_n_shot_32k": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
    "math500_32k":                 ("exact_match,none",             "exact_match_stderr,none"),
}
DATASETS = [("default", "default (neuralmagic)"),
            ("wikitext", "wikitext-2"),
            ("numina_concat", "numina_concat (problem+solution)")]
NS_VALUES = ["128", "256", "512", "1024"]
AB_VALUES = ["0.5", "0.75", "1"]

BASELINES = {
    "fp16": {"gsm8k_32k": 0.7619, "gpqa_diamond_cot_n_shot_32k": 0.2071, "math500_32k": 0.2840},
    "fp8 g=128": {"gsm8k_32k": 0.7536, "gpqa_diamond_cot_n_shot_32k": 0.2172, "math500_32k": 0.2660},
}


def parse_files():
    """Returns rows keyed by (method, dataset, ns, ab) → {task: (val, err)}.
    method ∈ {'puremax', 'qk'}."""
    rows = {}
    for f in sorted(glob.glob("logs/*_meta-llama-3-8b-instruct_smoothkv_g128_*_pair_vllm_results.json")):
        name = os.path.basename(f).replace("_vllm_results.json", "")
        # puremax form: ..._{ds_}?(perc_ns{N}_)?puremax_a{A}b{B}_pair
        # qk form:      ..._{ds_}?(perc_ns{N}_)?qk_a{A}b{B}_pair
        m = re.match(
            r"(?P<task>.+?)_meta-llama-3-8b-instruct_smoothkv_g128_"
            r"(?:(?:(?P<ds>wikitext|numina_concat)_)?)"
            r"(?:perc_ns(?P<ns>\d+)(?:_r\d+k)?_)?"
            r"(?P<method>puremax|qk)_a(?P<a>[0-9.]+)b(?P<b>[0-9.]+)_pair",
            name
        )
        if not m: continue
        method = m.group("method")
        d = m.group("ds") or "default"
        ns = m.group("ns") or "128"
        a, b = m.group("a"), m.group("b")
        ab = a if a == b else f"{a}/{b}"
        t = m.group("task")
        if t not in TASK_KEYS: continue
        d_js = json.load(open(f))
        metrics = list(d_js.values())[0]
        pk, sk = TASK_KEYS[t]
        rows.setdefault((method, d, ns, ab), {})[t] = (metrics.get(pk), metrics.get(sk))
    return rows


def cell(val, err=None, bold=False):
    if val is None:
        return '<td class="empty">—</td>'
    txt = f"{val:.4f}"
    if bold: txt = f"<b>{txt}</b>"
    err_s = f'<div class="err">±{err*100:.1f}</div>' if err else ""
    return f'<td class="val">{txt}{err_s}</td>'


def best_in_row(vals):
    best = None
    for v in vals:
        if v is None: continue
        if best is None or v > best: best = v
    return best


def build(out_path):
    rows = parse_files()

    def get(method, ds, ns, ab, t):
        return rows.get((method, ds, ns, ab), {}).get(t, (None, None))

    # Table 1: Dataset ablation (puremax, n=128, α=β=1)
    t1_rows = []
    for ds, label in DATASETS:
        cells_raw = [get("puremax", ds, "128", "1", t) for t, _ in TASKS_DISPLAY]
        vals = [c[0] for c in cells_raw]
        errs = [c[1] for c in cells_raw]
        best = best_in_row(vals)
        cells = "".join(cell(v, e, bold=(v == best)) for v, e in zip(vals, errs))
        t1_rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    t1 = "\n".join(t1_rows)

    # Table 2: n_calib × dataset × task (puremax, α=β=1)
    t2_rows = []
    for ds, label in DATASETS:
        for ns in NS_VALUES:
            cells_raw = [get("puremax", ds, ns, "1", t) for t, _ in TASKS_DISPLAY]
            cells = "".join(cell(v, e) for v, e in cells_raw)
            t2_rows.append(f"<tr><td>{label}</td><td>{ns}</td>{cells}</tr>")
    t2 = "\n".join(t2_rows)

    # Table 3: α=β × dataset, math500 only (puremax, n=128)
    t3_rows = []
    for ab in AB_VALUES:
        row_cells = []
        row_vals = []
        for ds, _ in DATASETS:
            v, e = get("puremax", ds, "128", ab, "math500_32k")
            row_vals.append(v)
            row_cells.append((v, e))
        best = best_in_row(row_vals)
        cells = "".join(cell(v, e, bold=(v == best)) for v, e in row_cells)
        t3_rows.append(f"<tr><td>α=β={ab}</td>{cells}</tr>")
    t3 = "\n".join(t3_rows)

    # Table 4: Scaling-method ablation (puremax vs QK-based, default × n=512)
    t4_rows = []
    for method_label, method in [("puremax (max-based)", "puremax"), ("QK-based", "qk")]:
        for ab in AB_VALUES:
            cells_raw = [get(method, "default", "512", ab, t) for t, _ in TASKS_DISPLAY]
            cells = "".join(cell(v, e) for v, e in cells_raw)
            t4_rows.append(f"<tr><td>{method_label}</td><td>α=β={ab}</td>{cells}</tr>")
    t4 = "\n".join(t4_rows)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SmoothKV puremax ablation — Llama-3-8B-Instruct</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1100px; margin: 30px auto; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .caption {{ color: #555; font-size: 13px; margin: 6px 0 14px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 10px; font-size: 13px; text-align: left; }}
  th {{ background: #f4f4f4; font-weight: 600; }}
  td.val {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 14px; }}
  td.empty {{ text-align: right; color: #bbb; }}
  .err {{ font-size: 10.5px; color: #888; margin-top: 2px; }}
  .ref {{ background: #f7f7f7; }}
</style></head>
<body>

<h1>SmoothKV puremax (α=β) ablation — Llama-3-8B-Instruct</h1>
<p class="caption">
  Backend: vLLM 0.6.6 + CUDA graphs, g=128, pair-max s_K (RoPE-mergeable). All scales are puremax
  (s_K = max|K|<sup>α</sup>, s_V = max|V|<sup>β</sup>, α=β). Bold = best in row.
</p>

<h2>Table 1 — Calibration dataset ablation <span style="color:#888;font-weight:normal;font-size:12px">(n=128, α=β=1)</span></h2>
<table>
  <thead>
    <tr><th>Calibration dataset</th>
        <th>{TASKS_DISPLAY[0][1]}</th>
        <th>{TASKS_DISPLAY[1][1]}</th>
        <th>{TASKS_DISPLAY[2][1]}</th>
    </tr>
  </thead>
  <tbody>
    {t1}
    <tr class="ref"><td>fp16 (reference)</td><td class="val">0.7619</td><td class="val">0.2071</td><td class="val">0.2840</td></tr>
    <tr class="ref"><td>fp8 g=128 (reference)</td><td class="val">0.7536</td><td class="val">0.2172</td><td class="val">0.2660</td></tr>
  </tbody>
</table>

<h2>Table 2 — n_calib ablation <span style="color:#888;font-weight:normal;font-size:12px">(α=β=1)</span></h2>
<table>
  <thead>
    <tr><th>Calib dataset</th><th>n_calib</th>
        <th>{TASKS_DISPLAY[0][1]}</th>
        <th>{TASKS_DISPLAY[1][1]}</th>
        <th>{TASKS_DISPLAY[2][1]}</th>
    </tr>
  </thead>
  <tbody>
    {t2}
  </tbody>
</table>

<h2>Table 3 — α=β ablation <span style="color:#888;font-weight:normal;font-size:12px">(n=128, math500 only)</span></h2>
<table>
  <thead>
    <tr><th>α=β</th>
        <th>default</th><th>wikitext-2</th><th>numina_concat</th>
    </tr>
  </thead>
  <tbody>
    {t3}
  </tbody>
</table>

<h2>Table 4 — Scaling-method ablation <span style="color:#888;font-weight:normal;font-size:12px">(default × n_calib=512, puremax vs QK-based)</span></h2>
<p class="caption">
  <b>puremax</b>: s_K = max|K|<sup>α</sup> &nbsp;&nbsp;
  <b>QK-based</b>: s_K = max|K|<sup>α</sup> / max|Q|<sup>(1-α)</sup> &nbsp;&nbsp;
  Both: s_V = max|V|<sup>β</sup>, β=α, pair-max on s_K.
</p>
<table>
  <thead>
    <tr><th>Scaling method</th><th>α=β</th>
        <th>{TASKS_DISPLAY[0][1]}</th>
        <th>{TASKS_DISPLAY[1][1]}</th>
        <th>{TASKS_DISPLAY[2][1]}</th>
    </tr>
  </thead>
  <tbody>
    {t4}
  </tbody>
</table>

<p class="caption">
  Generated from <code>logs/*_vllm_results.json</code> via <code>generate_ablation_report.py</code>.
</p>
</body></html>
"""
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="logs/report_ablation_tables.html")
    args = ap.parse_args()
    build(args.output)
