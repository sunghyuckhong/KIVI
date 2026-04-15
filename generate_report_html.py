"""
Generate an HTML results report from all logs/*_results.json files.

Usage:
    python generate_report_html.py            # writes report.html
    python generate_report_html.py -o out.html
"""
import argparse, json, os, math
from pathlib import Path

LOGS = Path("logs")

# ── Experiment registry ──────────────────────────────────────────────────────
# Each entry: (file_stem, display_label, bits, residual, group_size, model_type)
EXPERIMENTS = [
    # 2-bit runs (original)
    ("fp16",                        "FP16 baseline",             "—",  "—",   "—",   "fp16"),
    ("kivi",                        "KIVI (per-channel)",         "2",  "32",  "32",  "kivi"),
    ("pertoken",                    "PerToken g=32, res=32",      "2",  "32",  "32",  "pertoken"),
    ("pertoken_noresidual",         "PerToken g=32, res=0",       "2",  "0",   "32",  "pertoken"),
    ("pertoken_flat",               "PerToken g=128, res=32",     "2",  "32",  "128", "pertoken"),
    ("pertoken_flat_noresidual",    "PerToken g=128, res=0",      "2",  "0",   "128", "pertoken"),
    # 4-bit runs (res=128)
    ("kivi_int4_res128",            "KIVI int4 (per-channel)",    "4",  "128", "32",  "kivi"),
    ("pertoken_int4_res128",        "PerToken int4 g=32, res=128","4",  "128", "32",  "pertoken"),
    ("pertoken_int4_noresidual",    "PerToken int4 g=32, res=0",  "4",  "0",   "32",  "pertoken"),
    ("pertoken_int4_flat_res128",   "PerToken int4 g=128, res=128","4", "128", "128", "pertoken"),
    ("pertoken_int4_flat_noresidual","PerToken int4 g=128, res=0","4",  "0",   "128", "pertoken"),
]


def load_result(stem, task):
    path = LOGS / f"{task}_{stem}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if task == "gsm8k":
        g = d.get("gsm8k", {})
        strict  = g.get("exact_match,strict-match")
        flex    = g.get("exact_match,flexible-extract")
        if strict is None:
            return None
        return {"strict": strict, "flex": flex}
    else:
        g = d.get("gpqa_diamond_cot_n_shot", {})
        acc = g.get("exact_match,flexible-extract")
        if acc is None:
            return None
        return {"acc": acc}


def pct(v):
    return f"{v*100:.1f}%"


def delta_style(val, ref):
    """Return CSS color based on delta vs reference (FP16)."""
    if val is None or ref is None:
        return ""
    d = val - ref
    if abs(d) < 0.003:
        return "color:#555"
    intensity = min(int(abs(d) * 600), 180)
    if d > 0:
        return f"color:rgb(0,{100+intensity},0)"
    else:
        return f"color:rgb({150+intensity//2},0,0)"


def cell(val, ref, fmt_fn):
    if val is None:
        return '<td style="color:#aaa;text-align:center">—</td>'
    style = delta_style(val, ref)
    d = val - ref if ref is not None else 0
    sign = "+" if d >= 0 else ""
    delta_str = f'<span style="font-size:0.75em;opacity:0.75"> ({sign}{d*100:.1f}pp)</span>' if ref is not None and abs(d) > 0.001 else ""
    return f'<td style="text-align:center;{style}">{fmt_fn(val)}{delta_str}</td>'


def build_html():
    # Load all results
    rows = []
    fp16_gsm, fp16_gpqa = None, None
    for stem, label, bits, residual, group, mtype in EXPERIMENTS:
        gsm  = load_result(stem, "gsm8k")
        gpqa = load_result(stem, "gpqa")
        rows.append((stem, label, bits, residual, group, mtype, gsm, gpqa))
        if stem == "fp16":
            fp16_gsm  = gsm
            fp16_gpqa = gpqa

    fp16_gsm_strict = fp16_gsm["strict"] if fp16_gsm else None
    fp16_gsm_flex   = fp16_gsm["flex"]   if fp16_gsm else None
    fp16_gpqa_acc   = fp16_gpqa["acc"]   if fp16_gpqa else None

    tbody = ""
    for stem, label, bits, residual, group, mtype, gsm, gpqa in rows:
        is_fp16 = stem == "fp16"
        is_4bit = bits == "4"
        row_bg  = "#f9f9ff" if is_4bit else ("white" if not is_fp16 else "#fffbe6")
        border_top = "border-top:2px solid #ccc;" if stem == "kivi_int4_res128" else ""

        gsm_strict = cell(gsm["strict"] if gsm else None,
                          fp16_gsm_strict if not is_fp16 else None,
                          pct)
        gsm_flex   = cell(gsm["flex"]   if gsm else None,
                          fp16_gsm_flex   if not is_fp16 else None,
                          pct)
        gpqa_acc   = cell(gpqa["acc"]   if gpqa else None,
                          fp16_gpqa_acc   if not is_fp16 else None,
                          pct)

        bit_badge = (f'<span style="background:#ddeeff;border-radius:3px;padding:1px 4px;'
                     f'font-size:0.8em;margin-left:4px">INT{bits}</span>' if bits not in ("—", "2")
                     else (f'<span style="background:#ffe0cc;border-radius:3px;padding:1px 4px;'
                           f'font-size:0.8em;margin-left:4px">INT2</span>' if bits == "2" else ""))

        tbody += f"""
        <tr style="background:{row_bg};{border_top}">
          <td style="padding:6px 12px;white-space:nowrap">{label}{bit_badge}</td>
          <td style="text-align:center">{bits}</td>
          <td style="text-align:center">{group}</td>
          <td style="text-align:center">{residual}</td>
          {gsm_strict}
          {gsm_flex}
          {gpqa_acc}
        </tr>"""

    n_done = sum(1 for *_, gsm, gpqa in rows if gsm or gpqa)
    n_total = len(rows) * 2  # each experiment has 2 tasks

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KIVI KV-Cache Quantization Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1   {{ font-size: 1.4em; margin-bottom: 4px; }}
  p.sub {{ color: #666; font-size: 0.9em; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9em; }}
  th   {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: center; }}
  th.left {{ text-align: left; }}
  td   {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f4ff !important; }}
  .note {{ font-size: 0.8em; color: #888; margin-top: 8px; }}
  .section {{ margin-top: 32px; font-size: 1em; font-weight: bold; color: #2c3e50; }}
</style>
</head>
<body>
<h1>KIVI KV-Cache Quantization — Mistral-7B-Instruct-v0.2</h1>
<p class="sub">2-bit and 4-bit asymmetric min-max quantization &nbsp;|&nbsp;
GSM8K (5-shot, 1319 questions) &nbsp;|&nbsp; GPQA Diamond (198 questions) &nbsp;|&nbsp;
{n_done} / {n_total} task results available</p>

<table>
  <thead>
    <tr>
      <th class="left">Experiment</th>
      <th>Bits</th>
      <th>group_size</th>
      <th>residual</th>
      <th colspan="2">GSM8K</th>
      <th>GPQA Diamond</th>
    </tr>
    <tr>
      <th></th><th></th><th></th><th></th>
      <th>strict-match</th>
      <th>flexible-extract</th>
      <th>exact_match</th>
    </tr>
  </thead>
  <tbody>{tbody}
  </tbody>
</table>

<p class="note">
  Deltas shown in parentheses vs FP16 baseline. Green = better than FP16, red = worse.
  Yellow row = FP16 baseline. Blue rows = INT4 experiments. Orange badge = INT2, blue badge = INT4.
  <br>Hardware: NVIDIA V100 32GB &nbsp;|&nbsp; batch_size=16 (GSM8K), 8 or 4 (GPQA quantized) &nbsp;|&nbsp;
  Greedy decoding (temperature=0) &nbsp;|&nbsp; Seeds: random=0, numpy=1234, torch=1234
</p>
</body>
</html>"""
    return html


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="report.html")
    args = p.parse_args()

    html = build_html()
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report written to {args.output}")

    # Also print text summary
    print("\n── Text summary ──")
    for stem, label, bits, residual, group, mtype, gsm, gpqa in [
        (s, l, b, r, g, m, load_result(s, "gsm8k"), load_result(s, "gpqa"))
        for s, l, b, r, g, m, *_ in [(s,l,b,r,g,m,None,None) for s,l,b,r,g,m in EXPERIMENTS]
    ]:
        gsm_s  = f"{gsm['strict']*100:.1f}%" if gsm else "—"
        gsm_f  = f"{gsm['flex']*100:.1f}%"   if gsm else "—"
        gpqa_a = f"{gpqa['acc']*100:.1f}%"   if gpqa else "—"
        print(f"  {label:<38} GSM8K {gsm_s:>6} / {gsm_f:>6}   GPQA {gpqa_a:>6}")
