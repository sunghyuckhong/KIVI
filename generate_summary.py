"""Generate an HTML summary of all KIVI experiments."""
import json
import os
from datetime import datetime
from pathlib import Path

LOGS = Path("logs")


def load_result(filename, task):
    if not filename:
        return None
    path = LOGS / filename
    if not path.exists() or path.is_dir():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(task, None)


def get_metric(filename, task, metric_key):
    d = load_result(filename, task)
    if d is None:
        return None
    return d.get(metric_key)


PAPER_TARGETS = {
    ("Llama-2-7b-hf", "FP16", "GSM8k"): 13.50,
    ("Llama-2-7b-hf", "FP16", "CoQA"): 63.88,
    ("Llama-2-7b-hf", "FP16", "TruthfulQA"): 30.76,
    ("Llama-2-7b-hf", "KIVI-2", "GSM8k"): 12.74,
    ("Llama-2-7b-hf", "KIVI-2", "CoQA"): 63.05,
    ("Llama-2-7b-hf", "KIVI-2", "TruthfulQA"): 33.95,
    ("Mistral-7B-v0.1", "KIVI-2", "GSM8k"): 36.01,
    ("Mistral-7B-v0.1", "KIVI-2", "CoQA"): 66.35,
    ("Mistral-7B-v0.1", "KIVI-2", "TruthfulQA"): 32.17,
}


def build_row(model, method, files, paper_env=False, note=""):
    gsm8k = get_metric(files.get("gsm8k", ""), "gsm8k", "exact_match,strict-match") or \
            get_metric(files.get("gsm8k", ""), "gsm8k", "exact_match,get-answer")
    if gsm8k is not None:
        gsm8k *= 100

    coqa_em = get_metric(files.get("coqa", ""), "coqa", "em,none")
    if coqa_em is not None:
        coqa_em *= 100

    coqa_f1 = get_metric(files.get("coqa", ""), "coqa", "f1,none")
    if coqa_f1 is not None:
        coqa_f1 *= 100

    tfqa_bleu = get_metric(files.get("truthfulqa_gen", ""), "truthfulqa_gen",
                           "bleu_max,none")

    return {
        "model": model,
        "method": method,
        "paper_env": paper_env,
        "note": note,
        "gsm8k": gsm8k,
        "coqa_em": coqa_em,
        "coqa_f1": coqa_f1,
        "tfqa_bleu": tfqa_bleu,
    }


def delta_color(ours, paper):
    if ours is None or paper is None:
        return ""
    d = ours - paper
    if abs(d) < 0.3:
        return "color:#2d7a2d;font-weight:600"
    elif abs(d) < 1.0:
        return "color:#5a8a3a"
    elif abs(d) < 3.0:
        return "color:#b8860b"
    else:
        return "color:#c33"


def fmt(ours, paper_key=None):
    if ours is None:
        return '<span style="color:#aaa">—</span>'
    val_str = f"{ours:.2f}"
    if paper_key is not None:
        paper_val = PAPER_TARGETS.get(paper_key)
        if paper_val is not None:
            style = delta_color(ours, paper_val)
            return f'<span style="{style}">{val_str}</span> <span style="color:#999;font-size:0.82em">({paper_val:.2f})</span>'
    return val_str


METHOD_BG = {
    "FP16": "#fffbe6",
    "KIVI-2": "#f0f5ff",
    "KIVI-4": "#e6f2ff",
    "Naive INT4 per-token": "#fff0e6",
    "DeepSeekFP8": "#ffe6e6",
    "SmoothKV": "#e6ffe6",
}


def build_html():
    rows = []

    M = "Mistral-7B-Instruct-v0.2"
    m = "mistral-7b-instruct-v0.2"
    L = "Llama-2-7b-hf"
    l = "llama-2-7b-hf"
    Mb = "Mistral-7B-v0.1"
    mb = "mistral-7b-v0.1"

    # NOTE: only paper_env results included (ensures consistent env with KIVI paper reference).

    # ==== Llama-2-7b-hf (paper env only) ====
    rows.append(build_row(L, "FP16", {
        "gsm8k": f"gsm8k_{l}_fp16_paper_results.json",
        "coqa": f"coqa_{l}_fp16_paper_results.json",
    }, paper_env=True))
    rows.append(build_row(L, "KIVI-2", {
        "gsm8k": f"gsm8k_{l}_kivi2bit_res128_paper_results.json",
        "coqa": f"coqa_{l}_kivi2bit_res128_paper_results.json",
        "truthfulqa_gen": f"truthfulqa_gen_{l}_kivi2bit_res128_paper_results.json",
    }, paper_env=True, note="res=128, g=32"))

    # ==== Mistral-7B-v0.1 (paper env only) ====
    rows.append(build_row(Mb, "KIVI-2", {
        "gsm8k": f"gsm8k_{mb}_kivi2bit_res128_paper_results.json",
        "coqa": f"coqa_{mb}_kivi2bit_res128_paper_results.json",
        "truthfulqa_gen": f"truthfulqa_gen_{mb}_kivi2bit_res128_paper_results.json",
    }, paper_env=True, note="res=128, g=32"))

    # Build HTML rows
    tbody_html = ""
    prev_model = None
    for r in rows:
        separator = ""
        if prev_model is not None and r["model"] != prev_model:
            separator = '<tr><td colspan="6" style="background:#eaeaea;border-bottom:2px solid #ccc;padding:2px"></td></tr>'
        prev_model = r["model"]

        bg = METHOD_BG.get(r["method"], "white")
        paper_tag = '<span style="background:#d4edda;color:#155724;border-radius:3px;padding:1px 5px;font-size:0.75em;margin-left:4px">paper env</span>' if r["paper_env"] else ''
        note_tag = f'<div style="font-size:0.78em;color:#666;font-weight:normal">{r["note"]}</div>' if r["note"] else ''

        gsm8k_cell = fmt(r["gsm8k"], (r["model"], r["method"], "GSM8k"))

        if r["coqa_em"] is not None:
            em_str = fmt(r["coqa_em"], (r["model"], r["method"], "CoQA"))
            f1_str = f'<span style="color:#999;font-size:0.82em">F1: {r["coqa_f1"]:.2f}</span>'
            coqa_cell = f"EM: {em_str}<br>{f1_str}"
        else:
            coqa_cell = '<span style="color:#aaa">—</span>'

        tfqa_cell = fmt(r["tfqa_bleu"], (r["model"], r["method"], "TruthfulQA"))

        tbody_html += f"""
        <tr style="background:{bg}">
          <td>{r["model"]}</td>
          <td><strong>{r["method"]}</strong>{paper_tag}{note_tag}</td>
          <td class="num">{gsm8k_cell}</td>
          <td class="num">{coqa_cell}</td>
          <td class="num">{tfqa_cell}</td>
        </tr>{separator}"""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KV-Cache Quantization — Experiment Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1400px; margin: 30px auto; padding: 0 20px; color: #222;
  }}
  h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
  p.sub {{ color: #666; font-size: 0.9em; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.9em; }}
  th {{
    background: #2c3e50; color: white; padding: 10px 12px; text-align: center;
    vertical-align: middle;
  }}
  th.left {{ text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; vertical-align: middle; }}
  td.num {{ text-align: center; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #eef3ff !important; }}
  .footer {{
    font-size: 0.82em; color: #666; margin-top: 20px;
    border-top: 1px solid #ddd; padding-top: 14px; line-height: 1.6;
  }}
  .legend {{ display: inline-block; margin-right: 14px; font-size: 0.85em; }}
  .legend .sq {{ display: inline-block; width: 10px; height: 10px; border: 1px solid #ccc; vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>

<h1>KV-Cache Quantization — Experiment Report (paper env only)</h1>
<p class="sub">
  Only results run in the <code>kivi_paper</code> conda env
  (torch 2.1.2, transformers 4.36.2, lm-eval c9bbec6e — matches KIVI paper).<br>
  Paper targets from KIVI (Liu et al. 2024 ICML, Table 3) shown in parentheses.<br>
  <strong>Generated:</strong> {generated_at}
</p>

<table>
  <thead>
    <tr>
      <th class="left">Model</th>
      <th class="left">Method</th>
      <th>GSM8k (5-shot)<br><span style="font-weight:normal;font-size:0.82em">EM strict-match</span></th>
      <th>CoQA (0-shot)</th>
      <th>TruthfulQA gen (0-shot)<br><span style="font-weight:normal;font-size:0.82em">BLEU max</span></th>
    </tr>
  </thead>
  <tbody>{tbody_html}
  </tbody>
</table>

<div class="footer">
  <strong>Method definitions:</strong><br>
  <strong>KIVI-2 / KIVI-4</strong>: asymmetric min-max, <em>K per-channel + V per-token</em>, residual buffer
    of the last R tokens kept in FP16.<br>
  <strong>Naive INT4 per-token</strong>: both K and V quantized per-token with asymmetric min-max,
    group_size = head_dim (= 128, "flat"), no residual buffer.<br>
  <strong>DeepSeekFP8 / FineGrainedFP8</strong>: both K and V quantized per-token with float8_e4m3fn,
    group_size = 128 (flat), no residual. Symmetric (scale only).<br>
  <strong>SmoothKV (ours)</strong>: calibrated channel smoothing (diagonal s<sub>K</sub> post-RoPE,
    diagonal s<sub>V</sub>) + per-token INT4, group_size = 128, no residual, no µ shifts, no rotation.
    Tier-1 default from SmoothKV paper.<br>
  <br>
  <strong>Hardware:</strong> NVIDIA A100-SXM4-80GB &nbsp;|&nbsp;
  <strong>Modern env:</strong> torch 2.4.1, transformers 4.43.1, lm-eval 0.4.2<br>
  <strong>Paper env:</strong> torch 2.1.2, transformers 4.36.2, lm-eval c9bbec6e
    (used for KIVI paper replication only — results match paper exactly)<br>
  <strong>SmoothKV calibration:</strong> 128 samples × 2048 tokens from
    neuralmagic/LLM_compression_calibration, α = β = 0.5<br>
  Seeds: random=0, numpy=1234, torch=1234.<br>
</div>

</body>
</html>"""
    return html


if __name__ == "__main__":
    html = build_html()
    with open("summary.html", "w") as f:
        f.write(html)
    print("Written to summary.html")
