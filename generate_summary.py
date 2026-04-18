"""Generate an HTML summary of all KIVI experiments.

Scans logs/*.json for result files matching known naming patterns and builds
a full matrix HTML for the report. Paper targets shown in parentheses.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

LOGS = Path("logs")


def load_json(path):
    if not path.exists() or path.is_dir():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_gsm8k(fname):
    d = load_json(LOGS / fname)
    if not d or "gsm8k" not in d:
        return None
    r = d["gsm8k"]
    v = r.get("exact_match,strict-match") or r.get("exact_match,get-answer") \
        or r.get("exact_match,flexible-extract")
    return v * 100 if v is not None else None


def get_coqa_em(fname):
    d = load_json(LOGS / fname)
    if not d or "coqa" not in d:
        return None
    v = d["coqa"].get("em,none")
    return v * 100 if v is not None else None


def get_tfqa_bleu(fname):
    d = load_json(LOGS / fname)
    if not d or "truthfulqa_gen" not in d:
        return None
    return d["truthfulqa_gen"].get("bleu_max,none")


PAPER_TARGETS = {
    ("Llama-2-7b-hf", "FP16", "GSM8k"): 13.50,
    ("Llama-2-7b-hf", "FP16", "CoQA"): 63.88,
    ("Llama-2-7b-hf", "FP16", "TruthfulQA"): 30.76,
    ("Llama-2-7b-hf", "KIVI-2 (g=32,r=128)", "GSM8k"): 12.74,
    ("Llama-2-7b-hf", "KIVI-2 (g=32,r=128)", "CoQA"): 63.05,
    ("Llama-2-7b-hf", "KIVI-2 (g=32,r=128)", "TruthfulQA"): 33.95,
    ("Mistral-7B-v0.1", "FP16", "GSM8k"): 35.55,
    ("Mistral-7B-v0.1", "FP16", "CoQA"): 66.67,
    ("Mistral-7B-v0.1", "FP16", "TruthfulQA"): 32.57,
    ("Mistral-7B-v0.1", "KIVI-2 (g=32,r=128)", "GSM8k"): 36.01,
    ("Mistral-7B-v0.1", "KIVI-2 (g=32,r=128)", "CoQA"): 66.35,
    ("Mistral-7B-v0.1", "KIVI-2 (g=32,r=128)", "TruthfulQA"): 32.17,
}


def paper_file(model_short, k, g=None, r=None):
    """Build paper-env filename."""
    if k == 16:
        return f"{{task}}_{model_short}_fp16_paper_results.json"
    return f"{{task}}_{model_short}_kivi{k}bit_g{g}_res{r}_paper_results.json"


def paper_method_file(model_short, method, g=None, bits=None, res=None):
    """Build paper-env filename for the fp8 / pertoken / smoothkv ports
    (see the `_method_tag` helper in run_lm_eval_harness.py)."""
    if method == "fp8":
        return f"{{task}}_{model_short}_fp8paper_g{g}_paper_results.json"
    if method == "pertoken":
        return f"{{task}}_{model_short}_pertokenpaper_int{bits}_g{g}_res{res}_paper_results.json"
    if method == "smoothkv":
        return f"{{task}}_{model_short}_smoothkvpaper_g{g}_paper_results.json"
    return None


def row(model, method, file_tmpl, paper_key_method=None, env="paper"):
    """Build a result row — file_tmpl has '{task}' placeholder."""
    gsm8k   = get_gsm8k(file_tmpl.format(task="gsm8k"))
    coqa_em = get_coqa_em(file_tmpl.format(task="coqa"))
    tfqa    = get_tfqa_bleu(file_tmpl.format(task="truthfulqa_gen"))
    return dict(
        model=model, method=method, env=env, paper_key_method=paper_key_method,
        gsm8k=gsm8k, coqa_em=coqa_em, tfqa=tfqa,
    )


def fmt(val, paper_key):
    if val is None:
        return '<span style="color:#aaa">—</span>'
    s = f"{val:.2f}"
    paper = PAPER_TARGETS.get(paper_key)
    if paper is None:
        return s
    d = val - paper
    if abs(d) < 0.5:
        style = "color:#2d7a2d;font-weight:600"
    elif abs(d) < 1.5:
        style = "color:#5a8a3a"
    elif abs(d) < 3.0:
        style = "color:#b8860b"
    else:
        style = "color:#c33"
    return f'<span style="{style}">{s}</span> <span style="color:#999;font-size:0.82em">({paper:.2f})</span>'


METHOD_BG = {
    "FP16": "#fffbe6",
    "KIVI-2 (g=32,r=128)": "#f0f5ff",
    "KIVI-4 (g=32,r=128)": "#e6f2ff",
    "KIVI-4 (g=128,r=128)": "#e6f2ff",
    "KIVI-4 (g=32,r=32)": "#e6f2ff",
    "Naive INT4 per-token (g=128,r=0)": "#fff0e6",
    "DeepSeekFP8 per-token (g=128)": "#ffe6e6",
    "SmoothKV (ours, INT4, g=128)": "#e6ffe6",
}


def collect_rows():
    L, l = "Llama-2-7b-hf", "llama-2-7b-hf"
    M, m = "Mistral-7B-v0.1", "mistral-7b-v0.1"

    rows = []

    for (model, short) in [(L, l), (M, m)]:
        # FP16
        rows.append(row(model, "FP16", paper_file(short, 16), "FP16", env="paper"))
        # KIVI-2 (g=32, r=128)
        rows.append(row(model, "KIVI-2 (g=32,r=128)",
                        paper_file(short, 2, 32, 128), "KIVI-2 (g=32,r=128)", env="paper"))
        # KIVI-4 variants
        rows.append(row(model, "KIVI-4 (g=32,r=128)",
                        paper_file(short, 4, 32, 128), env="paper"))
        rows.append(row(model, "KIVI-4 (g=128,r=128)",
                        paper_file(short, 4, 128, 128), env="paper"))
        rows.append(row(model, "KIVI-4 (g=32,r=32)",
                        paper_file(short, 4, 32, 32), env="paper"))
        # Naive INT4 per-token — paper env
        rows.append(row(model, "Naive INT4 per-token (g=128,r=0)",
                        paper_method_file(short, "pertoken", g=128, bits=4, res=0), env="paper"))
        # DeepSeekFP8 — paper env
        rows.append(row(model, "DeepSeekFP8 per-token (g=128)",
                        paper_method_file(short, "fp8", g=128), env="paper"))
        # SmoothKV — paper env
        rows.append(row(model, "SmoothKV (ours, INT4, g=128)",
                        paper_method_file(short, "smoothkv", g=128), env="paper"))

    return rows


def build_html():
    rows = collect_rows()

    tbody_html = ""
    prev_model = None
    for r in rows:
        if prev_model is not None and r["model"] != prev_model:
            tbody_html += '<tr><td colspan="5" style="background:#eaeaea;height:3px;padding:0"></td></tr>'
        prev_model = r["model"]

        bg = METHOD_BG.get(r["method"], "white")
        env_tag = ('<span style="background:#d4edda;color:#155724;border-radius:3px;padding:1px 5px;font-size:0.72em;margin-left:4px">paper env</span>'
                   if r["env"] == "paper"
                   else '<span style="background:#dae6f3;color:#1a4975;border-radius:3px;padding:1px 5px;font-size:0.72em;margin-left:4px">modern env</span>')
        pkm = r["paper_key_method"]
        gsm8k_cell = fmt(r["gsm8k"], (r["model"], pkm, "GSM8k")) if pkm else fmt(r["gsm8k"], None)
        coqa_cell  = fmt(r["coqa_em"], (r["model"], pkm, "CoQA")) if pkm else fmt(r["coqa_em"], None)
        tfqa_cell  = fmt(r["tfqa"],   (r["model"], pkm, "TruthfulQA")) if pkm else fmt(r["tfqa"], None)

        tbody_html += f"""
        <tr style="background:{bg}">
          <td>{r["model"]}</td>
          <td><strong>{r["method"]}</strong>{env_tag}</td>
          <td class="num">{coqa_cell}</td>
          <td class="num">{tfqa_cell}</td>
          <td class="num">{gsm8k_cell}</td>
        </tr>"""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>KV-Cache Quantization — Experiment Report</title>
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

<h1>KV-Cache Quantization — Performance Comparison</h1>
<p class="sub">
  Paper targets from KIVI (Liu et al. 2024 ICML, Table 3) shown in parentheses.
  Color: <strong style="color:#2d7a2d">green</strong> = within 0.5pp,
  <strong style="color:#5a8a3a">olive</strong> = within 1.5pp,
  <strong style="color:#b8860b">amber</strong> = within 3pp,
  <strong style="color:#c33">red</strong> = &gt;3pp off.<br>
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
    </tr>
  </thead>
  <tbody>{tbody_html}
  </tbody>
</table>

<div class="footer">
  <strong>Method definitions:</strong><br>
  <strong>FP16</strong>: no KV quantization (baseline).<br>
  <strong>KIVI-2 / KIVI-4</strong>: asymmetric min-max, <em>K per-channel + V per-token</em>, residual buffer of last R tokens kept in FP16.<br>
  <strong>Naive INT4 per-token</strong>: both K and V quantized per-token with asymmetric min-max, group_size = head_dim (= 128), no residual buffer.<br>
  <strong>DeepSeekFP8</strong>: both K and V quantized per-token with <code>float8_e4m3fn</code>, group_size = 128, symmetric scale.<br>
  <strong>SmoothKV (ours)</strong>: calibrated channel smoothing (diagonal s<sub>K</sub> post-RoPE, s<sub>V</sub>) + per-token INT4, group_size = 128, no residual, no µ shifts, no rotation. Tier-1 default.<br>
  <br>
  <strong>Hardware:</strong> NVIDIA A100-SXM4-80GB.<br>
  <strong>Env:</strong> all rows run in paper env — torch 2.1.0, transformers 4.36.2, lm-eval commit c9bbec6e, bs=1.<br>
  <strong>SmoothKV calibration:</strong> 128 samples × 2048 tokens from <code>neuralmagic/LLM_compression_calibration</code>, α = β = 0.5.<br>
  Seeds: random=0, numpy=1234, torch=1234.
</div>

</body></html>"""
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None,
                    help="Output filename. Defaults to summary_YYYYMMDD-HHMM.html (no overwrite).")
    ap.add_argument("--paper-env", action="store_true", help="Ignored — present for orchestrator compat")
    ap.add_argument("--latest-alias", default="summary_latest.html",
                    help="Also write a 'latest' copy to this path (pass empty string to disable).")
    args = ap.parse_args()
    html = build_html()
    out = args.output or datetime.now().strftime("summary_%Y%m%d-%H%M.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Written to {out}")
    if args.latest_alias:
        with open(args.latest_alias, "w") as f:
            f.write(html)
        print(f"Latest alias at {args.latest_alias}")
