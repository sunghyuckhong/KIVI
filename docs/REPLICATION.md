# Replication guide — SmoothKV pair-mergeable + long-gen reasoning evals

This doc describes how to reproduce the pair-mergeable SmoothKV experiments (Eq. 7
pair-equal constraint on s_K) and the reasoning-benchmark protocol used in this repo.

## 1. Environment layout

Three Python environments, one per evaluation role:

| env path | python | transformers | lm-eval | used for |
|---|---|---|---|---|
| `/opt/paper_env_fast` | 3.10 | **4.36.2** | **0.4.2** | Llama-2 / Mistral-v0.1 KIVI/SmoothKV runs via `run_lm_eval_harness.py` |
| `/opt/modernenv` | 3.10 | 4.43.1 | 0.4.2 | Llama-3, Mistral-Instruct, base Llama-3-8B KIVI/SmoothKV runs via `run_eval.py` |
| `/opt/recent_env` | 3.10 | 4.51.3 | 0.4.11 | Qwen3 / AIME task (reasoning-model evals needing newer `DynamicCache` API) |

Install the KIVI triton extension inside each env:
```bash
cd quant && <ENV>/bin/pip install -e . && cd ..
```

## 2. Pipelines

### 2.1 Paper-env pipeline (Llama-2, Mistral-v0.1)

Entry point: `run_lm_eval_harness.py` (takes `--method kivi|fp8|pertoken|smoothkv`).
Results land at `logs/{task}_{model_short}_{_method_tag}_paper_results.json`.

### 2.2 Modern-env pipeline (Llama-3 / Mistral-Instruct / Llama-3-Instruct)

Entry point: `run_eval.py` (takes `--model fp16|kivi|pertoken|fp8|smoothkv`).
Results at `logs/{task}_{model_short}_{method_tag}_results.json`.

Both entry points accept `--max_gen_toks <N>` to override the task-yaml generation
length at runtime (no yaml edits needed).

## 3. Generation-length policy

### 3.1 What we changed

- `tasks/math500/math500.yaml` — renamed task to `math500_32k`, bumped
  `max_gen_toks` from 1024 to 32768.
- `tasks/gsm8k/gsm8k_32k.yaml` — new local override of upstream `gsm8k`
  (5-shot EM-strict) with `max_gen_toks: 32768`.
- `tasks/gpqa/gpqa_diamond_cot_n_shot_32k.yaml` + `_gpqa_cot_n_shot_yaml_32k.yaml`
  + `utils.py` — local overrides of the upstream GPQA-Diamond CoT n-shot task,
  adding `max_gen_toks: 32768`.
- `tasks/aime/*` — copied verbatim from lm-eval 0.4.11 (`aime`, `aime24`, `aime25`).

Both `run_lm_eval_harness.py` and `run_eval.py` register all of `tasks/*` via
`TaskManager(include_path="tasks/")`.

### 3.2 Per-model max_gen_toks

Use `max_gen_toks = model_max_length // 2` for non-reasoning models, 32768 for
reasoning models. Concretely:

| model | `max_position_embeddings` | `--max_gen_toks` |
|---|---|---|
| Llama-2-7B | 4096 | 2048 |
| Mistral-7B-v0.1 | 32768 | 16384 |
| Mistral-7B-Instruct-v0.2 | 32768 | 16384 |
| Llama-3-8B / -8B-Instruct | 8192 | 4096 |
| DeepSeek-R1-Distill-Llama-8B | 131072 | 32768 |
| DeepSeek-R1-Distill-Qwen-7B | 131072 | 32768 |
| Qwen3-8B | 131072 | 32768 |

**Why not uniform 32k?** Setting `max_gen_toks=32768` on Llama-2 truncates the
prompt to 0 tokens (`4096 - 32768 = -28672` → `IndexError` in transformers 4.36.2
`generate()`). Use the half-context rule for non-reasoning models.

### 3.3 Per-config batch size (reasoning tasks)

KV-cache at `max_gen_toks = N × bs` is the OOM bottleneck. Safe bs on 80GB A100:

| model family | FP16 | KIVI-quantized (≤4-bit) |
|---|---|---|
| Llama-2-7B (32 KV heads, no GQA) | 2 | 8 |
| Mistral-7B (GQA, 8 KV heads) | 8 | 16 |
| Llama-3-8B (GQA, 8 KV heads) | 16 | 16 |
| R1-Distill-Llama-8B at 32k gen | 4 | 8 |
| R1-Distill-Qwen-7B at 32k gen (GQA, 4 KV) | 8 | 16 |

`launch_wave.sh` now:
- Chains python invocations with `&&` (first failure aborts).
- Emits `DONE_<name>` on success, `FAIL_<name>` on any non-zero exit.
- `overnight_watcher.sh` treats both as terminal.

## 4. SmoothKV pair-mergeable calibration

Two-stage setup:

### Stage A — calibrate (once per model, ~30 min, 1 GPU)
```bash
python run_smoothkv_calibrate.py \
    --model_path <model> --alpha 0.5 --beta 0.5 \
    --samples_per_channel 10000 \
    --output logs/calib/smoothkv_<short>_perc.pt
```
This produces **both** max stats and a reservoir of |K|/|V| samples per channel —
needed for later variant generation.

### Stage B — offline variant generation (CPU, seconds)

```bash
# α-based mergeable variant (SmoothQuant form with pair-max on s_K)
python scripts/make_alpha_variants.py \
    --base logs/calib/smoothkv_<short>_a0.5.pt \
    --alphas 0.75 --pair_max_k
# → logs/calib/smoothkv_<short>_a0.75_pair.pt

# Percentile mergeable variants (Eq. 7)
python scripts/make_percentile_variants.py \
    --base logs/calib/smoothkv_<short>_perc.pt \
    --pk 90 95 99 99.9 --symmetric
# → logs/calib/smoothkv_<short>_pairK{90,95,99,99p9}_pV{...}.pt
```

Both generators apply **pair-max on s_K** by default (Eq. 7:
`(s_K)_pair_i = max(|K_{:,2i}|^p, |K_{:,2i+1}|^p)`) so the smoothing is
mergeable with RoPE's channel-pair rotation. To reproduce the non-mergeable
baseline, pass `--no-pair-max-k`.

**No geo-mean normalization.** Earlier versions of these scripts applied
log-space centering; we removed it because it's a no-op on attention outputs
(Q × s_K cancels against K / s_K regardless of overall scale).

### Stage C — eval a calibrated variant

```bash
python run_lm_eval_harness.py \
    --model_name_or_path <model> --method smoothkv \
    --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 \
    --calib_path logs/calib/smoothkv_<short>_a0.75_pair.pt \
    --tasks gsm8k_32k --batch_size 8 --max_gen_toks 16384
```

## 5. One-shot model matrix launcher

`scripts/phaseX_instruct_launch.sh` is a reference template:

1. Parallel calibration for 2 models on 2 GPUs.
2. CPU variant generation (5 pair-mergeable variants per model).
3. 9 eval streams per model on 4 GPUs with polling GPU-availability.

Methods per stream:
- `FP16`, `KIVI-2 (g=32, r=128)`, `pertoken int4 (g=128, r=0)`, `fp8 (g=128)`
- `SmoothKV α=0.75 pair` + `SmoothKV pair-percentile {90, 95, 99, 99.9}`

Each stream runs `{gsm8k_32k, gpqa_diamond_cot_n_shot_32k, math500_32k}` chained
with `&&` (TQA + CoQA recycled from disk if already present at 256 default).

## 6. Overnight watcher

`scripts/overnight_watcher_v3.sh`:
- Polls tmux pane content for `^(DONE|FAIL)_<stream>` markers.
- Chains phases: calibration → variant generation → evaluation waves.
- Regenerates `summary_latest.html` + `summary_simplified_latest.html`
  at each phase boundary.
- Fully shell-based — does not depend on a live Claude Code session.

Launch in tmux to survive terminal close:
```bash
tmux new-window -t kivi: -n overnight_watcher \
  "bash scripts/overnight_watcher_v3.sh; read"
```

## 7. Reports

- `generate_summary.py` — full matrix with paper-target deltas, env tags,
  all α rows, color-coded deltas.
- `generate_simplified_report.py` — trimmed report: best α only, mergeable
  rows in green / unmergeable in red.

Both readers look for `*_32k` task names in result JSON with fallback to the
legacy names (so pre-policy results still display).

## 8. Known pitfalls

- **Don't set `max_gen_toks > max_position_embeddings / 2`** on any non-reasoning
  model — you'll get `IndexError: index -1 is out of bounds` in transformers
  ≤ 4.36.2 or silently-truncated prompts in newer versions.
- **Don't chain with `;`.** If a python invocation OOMs, later ones will too,
  and you get a fake "done" with no saved results. Use `&&` (as
  `launch_wave.sh` now does) and check for `FAIL_` markers.
- **Paper_env_fast** (lm-eval 0.4.2) doesn't have AIME in its task catalog;
  copy the 0.4.11 task files into `tasks/aime/` via `include_path`.
- **Qwen3 port** requires transformers ≥ 4.47 (`DynamicCache` API) and a
  port rewrite — the Llama/Qwen2 template assumes tuple `past_key_value`
  which is removed there.
