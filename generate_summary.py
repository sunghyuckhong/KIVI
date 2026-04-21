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
    if not d:
        return None
    for key in ("gsm8k_32k", "gsm8k"):
        if key in d:
            r = d[key]
            v = r.get("exact_match,strict-match") or r.get("exact_match,get-answer") \
                or r.get("exact_match,flexible-extract")
            return v * 100 if v is not None else None
    return None


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


def get_gpqa(fname):
    d = load_json(LOGS / fname)
    if not d:
        return None
    for key in ("gpqa_diamond_cot_n_shot_32k", "gpqa_diamond_cot_n_shot", "gpqa_diamond_cot_zeroshot"):
        if key in d:
            r = d[key]
            v = r.get("exact_match,flexible-extract") or r.get("exact_match,strict-match")
            return v * 100 if v is not None else None
    return None


def get_math500(fname):
    d = load_json(LOGS / fname)
    if not d:
        return None
    for key in ("math500_32k", "math500"):
        if key in d:
            v = d[key].get("exact_match,none")
            return v * 100 if v is not None else None
    return None


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


def paper_method_file(model_short, method, g=None, bits=None, res=None, alpha=None):
    """Build paper-env filename for the fp8 / pertoken / smoothkv ports
    (see the `_method_tag` helper in run_lm_eval_harness.py)."""
    if method == "fp8":
        return f"{{task}}_{model_short}_fp8paper_g{g}_paper_results.json"
    if method == "pertoken":
        return f"{{task}}_{model_short}_pertokenpaper_int{bits}_g{g}_res{res}_paper_results.json"
    if method == "smoothkv":
        alpha_tag = f"_a{alpha}" if alpha is not None else ""
        return f"{{task}}_{model_short}_smoothkvpaper_g{g}{alpha_tag}_paper_results.json"
    return None


def row(model, method, file_tmpl, paper_key_method=None, env="paper"):
    """Build a result row — file_tmpl has '{task}' placeholder."""
    # Prefer _32k (reasoning benchmarks) results; fall back to legacy 256/1024 files if _32k missing.
    gsm8k = get_gsm8k(file_tmpl.format(task="gsm8k_32k")) or get_gsm8k(file_tmpl.format(task="gsm8k"))
    coqa_em = get_coqa_em(file_tmpl.format(task="coqa"))
    tfqa    = get_tfqa_bleu(file_tmpl.format(task="truthfulqa_gen"))
    gpqa = get_gpqa(file_tmpl.format(task="gpqa_diamond_cot_n_shot_32k")) \
        or get_gpqa(file_tmpl.format(task="gpqa_diamond_cot_n_shot")) \
        or get_gpqa(file_tmpl.format(task="gpqa_diamond_cot_zeroshot"))
    math500 = get_math500(file_tmpl.format(task="math500_32k")) or get_math500(file_tmpl.format(task="math500"))
    return dict(
        model=model, method=method, env=env, paper_key_method=paper_key_method,
        gsm8k=gsm8k, coqa_em=coqa_em, tfqa=tfqa, gpqa=gpqa, math500=math500,
    )


def modern_file(model_short, method, g=None, bits=None, res=None, alpha=None):
    """Build modern-env filename (run_eval.py output)."""
    if method == "fp16":
        return f"{{task}}_{model_short}_fp16_results.json"
    if method == "kivi":
        return f"{{task}}_{model_short}_kivi_res128_results.json"
    if method == "smoothkv":
        return f"{{task}}_{model_short}_smoothkv_g128_results.json"
    return None


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
    "SmoothKV α=0.25": "#e6ffe6",
    "SmoothKV α=0.5":  "#d0f4d0",
    "SmoothKV α=0.75": "#b8ebbf",
    "SmoothKV α=1.0":  "#a0e3b0",
}


def collect_rows():
    L, l = "Llama-2-7b-hf", "llama-2-7b-hf"
    M, m = "Mistral-7B-v0.1", "mistral-7b-v0.1"
    MI, mi = "Mistral-7B-Instruct-v0.2", "mistral-7b-instruct-v0.2"
    L3, l3 = "Llama-3-8B-Instruct", "meta-llama-3-8b-instruct"

    rows = []

    for (model, short) in [(L, l), (M, m)]:
        # Paper-env base models
        rows.append(row(model, "FP16", paper_file(short, 16), "FP16", env="paper"))
        rows.append(row(model, "KIVI-2 (g=32,r=128)",
                        paper_file(short, 2, 32, 128), "KIVI-2 (g=32,r=128)", env="paper"))
        rows.append(row(model, "KIVI-4 (g=32,r=128)",
                        paper_file(short, 4, 32, 128), env="paper"))
        rows.append(row(model, "KIVI-4 (g=128,r=128)",
                        paper_file(short, 4, 128, 128), env="paper"))
        rows.append(row(model, "Naive INT4 per-token (g=128,r=0)",
                        paper_method_file(short, "pertoken", g=128, bits=4, res=0), env="paper"))
        rows.append(row(model, "DeepSeekFP8 per-token (g=128)",
                        paper_method_file(short, "fp8", g=128), env="paper"))
        for a_tag, a_label in [("0.25", "α=0.25"), ("0.5", "α=0.5"),
                                ("0.75", "α=0.75"), ("1",   "α=1.0")]:
            rows.append(row(model, f"SmoothKV {a_label}",
                            paper_method_file(short, "smoothkv", g=128, alpha=a_tag),
                            env="paper"))
        # α=0.75 pair-max (mergeable, Eq. 7 form)
        a075_pair = f"{{task}}_{short}_smoothkvpaper_g128_a0.75_pair_paper_results.json"
        if any((LOGS / a075_pair.format(task=t)).exists()
               for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
            rows.append(row(model, "SmoothKV α=0.75 pair (mergeable)", a075_pair, env="paper"))
        # Unmergeable per-channel percentile
        for pk_tag, pk_label in [("95", "p=95"), ("99", "p=99"), ("99p9", "p=99.9")]:
            f_tmpl = f"{{task}}_{short}_smoothkvpaper_g128_pK{pk_tag}_pV{pk_tag}_paper_results.json"
            if any((LOGS / f_tmpl.format(task=t)).exists()
                   for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
                rows.append(row(model, f"SmoothKV {pk_label} (unmergeable)", f_tmpl, env="paper"))
        # Mergeable pair-max percentile (Eq. 7)
        for pk_tag, pk_label in [("95", "p=95"), ("99", "p=99"), ("99p9", "p=99.9")]:
            f_tmpl = f"{{task}}_{short}_smoothkvpaper_g128_pairK{pk_tag}_pV{pk_tag}_paper_results.json"
            if any((LOGS / f_tmpl.format(task=t)).exists()
                   for t in ("coqa", "gsm8k", "truthfulqa_gen", "math500", "gpqa_diamond_cot_n_shot")):
                rows.append(row(model, f"SmoothKV {pk_label} pair (mergeable)", f_tmpl, env="paper"))

    # Modern-env instruct models (only FP16 / KIVI-2 / SmoothKV α=0.75)
    for (model, short) in [(MI, mi), (L3, l3)]:
        rows.append(row(model, "FP16", modern_file(short, "fp16"), env="modern"))
        rows.append(row(model, "KIVI-2 (g=32,r=128)", modern_file(short, "kivi"), env="modern"))
        rows.append(row(model, "SmoothKV α=0.75", modern_file(short, "smoothkv"), env="modern"))

    return rows


def build_html():
    rows = collect_rows()

    tbody_html = ""
    prev_model = None
    for r in rows:
        if prev_model is not None and r["model"] != prev_model:
            tbody_html += '<tr><td colspan="7" style="background:#eaeaea;height:3px;padding:0"></td></tr>'
        prev_model = r["model"]

        bg = METHOD_BG.get(r["method"], "white")
        env_tag = ('<span style="background:#d4edda;color:#155724;border-radius:3px;padding:1px 5px;font-size:0.72em;margin-left:4px">paper env</span>'
                   if r["env"] == "paper"
                   else '<span style="background:#dae6f3;color:#1a4975;border-radius:3px;padding:1px 5px;font-size:0.72em;margin-left:4px">modern env</span>')
        pkm = r["paper_key_method"]
        gsm8k_cell   = fmt(r["gsm8k"], (r["model"], pkm, "GSM8k")) if pkm else fmt(r["gsm8k"], None)
        coqa_cell    = fmt(r["coqa_em"], (r["model"], pkm, "CoQA")) if pkm else fmt(r["coqa_em"], None)
        tfqa_cell    = fmt(r["tfqa"],   (r["model"], pkm, "TruthfulQA")) if pkm else fmt(r["tfqa"], None)
        gpqa_cell    = fmt(r["gpqa"], None)
        math500_cell = fmt(r["math500"], None)

        tbody_html += f"""
        <tr style="background:{bg}">
          <td>{r["model"]}</td>
          <td><strong>{r["method"]}</strong>{env_tag}</td>
          <td class="num">{coqa_cell}</td>
          <td class="num">{tfqa_cell}</td>
          <td class="num">{gsm8k_cell}</td>
          <td class="num">{gpqa_cell}</td>
          <td class="num">{math500_cell}</td>
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
      <th>GPQA-Diamond<br><span style="font-weight:normal;font-size:0.82em">EM 5-shot CoT</span></th>
      <th>MATH500<br><span style="font-weight:normal;font-size:0.82em">EM</span></th>
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
