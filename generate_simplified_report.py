"""Simplified KV-cache quantization report.

- Strips paper-target (value-in-parens) from every cell.
- Keeps only the best α=0.75 SmoothKV row (drops α=0.25, 0.5, 1.0 rows).
- Keeps all SmoothKV percentile rows that have landed on disk.
- Otherwise identical layout to generate_summary.py.

Usage:
  python generate_simplified_report.py [--output summary_simplified.html]
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

# Reuse all loaders from the full report generator
import generate_summary as gs

LOGS = gs.LOGS


def fmt(val, _unused=None):
    """Minimal formatter — no paper-target annotation."""
    if val is None:
        return '<span style="color:#aaa">—</span>'
    return f"{val:.2f}"


# Muted greens for mergeable rows; muted reds for unmergeable
METHOD_BG = {
    **gs.METHOD_BG,
    "SmoothKV α=0.75 (unmergeable)": "#f5d5d5",
    "SmoothKV α=0.75 pair (mergeable)": "#b8ebbf",
    "SmoothKV p=95 (unmergeable)": "#f5e1e1",
    "SmoothKV p=99 (unmergeable)": "#f0d1d1",
    "SmoothKV p=99.9 (unmergeable)": "#ebc0c0",
    "SmoothKV p=95 pair (mergeable)": "#d0f4d0",
    "SmoothKV p=99 pair (mergeable)": "#a8e3b8",
    "SmoothKV p=99.9 pair (mergeable)": "#8fdba3",
}


def collect_simplified_rows():
    L, l = "Llama-2-7b-hf",              "llama-2-7b-hf"
    M, m = "Mistral-7B-v0.1",            "mistral-7b-v0.1"
    MI, mi = "Mistral-7B-Instruct-v0.2", "mistral-7b-instruct-v0.2"
    L3, l3 = "Llama-3-8B-Instruct",      "meta-llama-3-8b-instruct"

    rows = []
    for (model, short) in [(L, l), (M, m)]:
        rows.append(gs.row(model, "FP16", gs.paper_file(short, 16), env="paper"))
        rows.append(gs.row(model, "KIVI-2 (g=32,r=128)",
                           gs.paper_file(short, 2, 32, 128), env="paper"))
        rows.append(gs.row(model, "KIVI-4 (g=32,r=128)",
                           gs.paper_file(short, 4, 32, 128), env="paper"))
        rows.append(gs.row(model, "KIVI-4 (g=128,r=128)",
                           gs.paper_file(short, 4, 128, 128), env="paper"))
        rows.append(gs.row(model, "Naive INT4 per-token (g=128,r=0)",
                           gs.paper_method_file(short, "pertoken", g=128, bits=4, res=0),
                           env="paper"))
        rows.append(gs.row(model, "DeepSeekFP8 per-token (g=128)",
                           gs.paper_method_file(short, "fp8", g=128), env="paper"))
        # α=0.75 baseline (non-pair, unmergeable)
        rows.append(gs.row(model, "SmoothKV α=0.75 (unmergeable)",
                           gs.paper_method_file(short, "smoothkv", g=128, alpha="0.75"),
                           env="paper"))
        # α=0.75 pair-max (mergeable, Eq. 7)
        a075_pair = f"{{task}}_{short}_smoothkvpaper_g128_a0.75_pair_paper_results.json"
        if any((LOGS / a075_pair.format(task=t)).exists()
               for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
            rows.append(gs.row(model, "SmoothKV α=0.75 pair (mergeable)", a075_pair, env="paper"))
        # Unmergeable percentile (per-channel s_K — violates pair-equal)
        for pk_tag, pk_label in [("95", "p=95"), ("99", "p=99"), ("99p9", "p=99.9")]:
            f_tmpl = f"{{task}}_{short}_smoothkvpaper_g128_pK{pk_tag}_pV{pk_tag}_paper_results.json"
            if any((LOGS / f_tmpl.format(task=t)).exists()
                   for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
                rows.append(gs.row(model, f"SmoothKV {pk_label} (unmergeable)", f_tmpl, env="paper"))
        # Mergeable percentile (pair-max, Eq. 7)
        for pk_tag, pk_label in [("95", "p=95"), ("99", "p=99"), ("99p9", "p=99.9")]:
            f_tmpl = f"{{task}}_{short}_smoothkvpaper_g128_pairK{pk_tag}_pV{pk_tag}_paper_results.json"
            if any((LOGS / f_tmpl.format(task=t)).exists()
                   for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
                rows.append(gs.row(model, f"SmoothKV {pk_label} pair (mergeable)", f_tmpl, env="paper"))

    for (model, short) in [(MI, mi), (L3, l3)]:
        rows.append(gs.row(model, "FP16", gs.modern_file(short, "fp16"), env="modern"))
        rows.append(gs.row(model, "KIVI-2 (g=32,r=128)", gs.modern_file(short, "kivi"), env="modern"))
        rows.append(gs.row(model, "SmoothKV α=0.75 (best)", gs.modern_file(short, "smoothkv"), env="modern"))

    return rows


def build_html():
    rows = collect_simplified_rows()

    tbody_html = ""
    prev_model = None
    for r in rows:
        if prev_model is not None and r["model"] != prev_model:
            tbody_html += '<tr><td colspan="7" style="background:#eaeaea;height:3px;padding:0"></td></tr>'
        prev_model = r["model"]

        bg = METHOD_BG.get(r["method"], "white")
        tbody_html += f"""
        <tr style="background:{bg}">
          <td>{r["model"]}</td>
          <td><strong>{r["method"]}</strong></td>
          <td class="num">{fmt(r["coqa_em"])}</td>
          <td class="num">{fmt(r["tfqa"])}</td>
          <td class="num">{fmt(r["gsm8k"])}</td>
          <td class="num">{fmt(r["gpqa"])}</td>
          <td class="num">{fmt(r["math500"])}</td>
        </tr>"""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>KV-Cache Quantization — Simplified Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 30px auto; padding: 0 20px; color: #222;
  }}
  h1 {{ font-size: 1.4em; margin-bottom: 4px; }}
  p.sub {{ color: #666; font-size: 0.9em; margin-top: 0; line-height: 1.5; }}
  table {{
    border-collapse: collapse; width: 100%; margin-top: 16px;
    font-size: 0.9em; border-top: 2px solid #222; border-bottom: 2px solid #222;
  }}
  thead tr {{ border-bottom: 2px solid #222; }}
  th {{ padding: 10px 12px; text-align: center; font-weight: 600; }}
  th.left {{ text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  td.num {{ text-align: center; font-variant-numeric: tabular-nums; }}
  .footer {{
    font-size: 0.8em; color: #666; margin-top: 18px;
    border-top: 1px solid #ddd; padding-top: 12px; line-height: 1.6;
  }}
</style>
</head><body>

<h1>KV-Cache Quantization — Simplified Report</h1>
<p class="sub">
  One SmoothKV α row (α=0.75, best); all percentile rows if measured.<br>
  <strong>Generated:</strong> {generated_at}
</p>

<table>
  <thead>
    <tr>
      <th class="left">Model</th>
      <th class="left">Method</th>
      <th>CoQA<br><span style="font-weight:normal;font-size:0.82em">EM</span></th>
      <th>TruthfulQA gen<br><span style="font-weight:normal;font-size:0.82em">BLEU max</span></th>
      <th>GSM8K (5-shot)<br><span style="font-weight:normal;font-size:0.82em">EM strict</span></th>
      <th>GPQA-Diamond<br><span style="font-weight:normal;font-size:0.82em">EM 5-shot CoT</span></th>
      <th>MATH500<br><span style="font-weight:normal;font-size:0.82em">EM</span></th>
    </tr>
  </thead>
  <tbody>{tbody_html}
  </tbody>
</table>

<div class="footer">
  Full matrix (all α, paper target references, colour-coded deltas) — see <code>summary_latest.html</code> from <code>generate_summary.py</code>.
</div>

</body></html>"""
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None,
                    help="Output filename. Defaults to summary_simplified_YYYYMMDD-HHMM.html")
    ap.add_argument("--latest-alias", default="summary_simplified_latest.html")
    args = ap.parse_args()
    html = build_html()
    out = args.output or datetime.now().strftime("summary_simplified_%Y%m%d-%H%M.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Written simplified report to {out}")
    if args.latest_alias:
        with open(args.latest_alias, "w") as f:
            f.write(html)
        print(f"Latest alias at {args.latest_alias}")
