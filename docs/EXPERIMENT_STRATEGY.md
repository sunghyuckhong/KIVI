# Experiment strategy — KV-cache fake-quant ablations (Llama3 / Mistral / DSR1 / Qwen3)

What, why, and how we run the KV-cache quantization matrix on reasoning tasks
(gsm8k, gpqa_diamond_cot_n_shot, math500). This is the operational playbook —
read alongside `docs/REPLICATION.md` for per-paper reproducibility and
`docs/GUIDE_VLLM_PATH.md` for the vLLM fake-quant monkey-patch mechanics.

## Goal

Compare KV-cache quantization methods on reasoning accuracy, holding generation
faithful to full-context eval (32k output cap) as a fairness baseline.

Methods tested (`run_eval_vllm.py --model`):

| Method     | Storage    | Calibration | Notes |
|------------|------------|-------------|-------|
| `fp16`     | FP16       | none        | Upper-bound reference |
| `fp8`      | FP8 g=128  | none        | Group-wise FP8 fake-quant |
| `pertoken` | INT4 g=128 | none        | Per-token INT4 asym |
| `smoothkv` | INT4 g=128 | required    | Post-RoPE K/V scaling; pair-max s_K (RoPE-mergeable); α=β |
| `kivi`     | INT4 +res  | none        | KIVI-2 with residual=128 — note: residual buffer not faithfully simulated in vLLM |

SmoothKV is the contribution — everything else is a baseline for context.

## Models under test

Four models, all 7-8B class:

| Preset (in `scripts/launch_vllm_reasoning.sh`) | Model                                        | Native ctx | MG    | MAX_NS | vLLM env              |
|-----------------------------------------------|----------------------------------------------|------------|-------|--------|-----------------------|
| `llama3-instruct`                             | `meta-llama/Meta-Llama-3-8B-Instruct`        | 8192       | 4096  | 128    | `/opt/vllm_env` (0.6.6) |
| `mistral-instruct`                            | `mistralai/Mistral-7B-Instruct-v0.2`         | 32768      | 16384 | 15     | `/opt/vllm_env` (0.6.6) |
| `dsr1-llama-8b`                               | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`   | 131072     | 32768 | 8-16   | `/opt/vllm_env` (0.6.6) |
| `qwen3-8b`                                    | `Qwen/Qwen3-8B`                              | 131072     | 32768 | TBD    | `/opt/vllm_qwen3_env` (0.8+) — see `docs/GUIDE_QWEN3_PORT.md` |

`MG = max_gen_toks` at launch. `MAX_NS = max_num_seqs` (vLLM concurrency slots).

`max_gen_toks` rule: if native ctx > 32768 → cap at 32768; else → ctx / 2.
Every method is run at the same MG within a model to keep comparisons fair.

## Per-model test plan

### Llama-3-8B-Instruct — ablation surface (done)

This is the model we explored to death. Drives method-selection decisions that
are then validated on the other three models at a single sweet-spot config.

Ablations run (all 3 tasks × each cell):

- **α=β sweep** ∈ {0.5, 0.75, 1.0}, n_calib=128, default calib dataset.
- **n_calib sweep** ∈ {128, 256, 512, 1024}, α=β=1, default calib dataset.
- **Calibration dataset sweep** at n=128, α=β=1: `neuralmagic/LLM_compression_calibration` (default), `wikitext-2-raw-v1`, `AI-MO/NuminaMath-1.5` (problem+solution concat).
- **Scaling-method sweep** at default×n=512: max-based puremax (`s_K = max|K|^α`) vs QK-based (`s_K = max|K|^α / max|Q|^(1-α)`), both α=β.

All SmoothKV variants use **pair-max s_K** (per `feedback_pair_max_always.md` —
RoPE mergeability constraint).

Sweet-spot identified: **puremax, α=β=1, n_calib=512, default dataset, pair-max**.
That single config is what we validate on the other models.

Aggregated via `generate_ablation_report.py` → `logs/report_ablation_tables.html`.

### Mistral-7B-Instruct-v0.2 — cross-arch validation

Single SmoothKV config (sweet-spot from Llama3) on all 3 tasks. Baselines for
Mistral (FP16/FP8/pertoken) not rerun here — user has those from elsewhere.

Calibration: 512 samples × seq_length 2048 × `samples_per_channel=300000`
(reservoir size; irrelevant for max-based scales but kept consistent).

### DeepSeek-R1-Distill-Llama-8B — full 4-method matrix

Reasoning-tuned distill, long-CoT generations. All 4 methods × all 3 tasks.
Same SmoothKV sweet-spot config as above (puremax α=β=1 n=512 default pair-max).

Ran at `MG=16384 MAX_NS=16 LOG_SAMPLES=1` (adaptive two-pass — see below) to
keep wall-clock tolerable while preserving fairness at 32k.

### Qwen3-8B — port

Separate env (`/opt/vllm_qwen3_env`) and separate patch targets. Full guide:
`docs/GUIDE_QWEN3_PORT.md`. Same 4-method matrix planned once patches land.

## Per-task setup

Measured prompt lengths (via `scripts/measure_prompt_lens.py`, 5-shot fewshot):

| Task                               | Max prompt tokens | Items |
|-----------------------------------|-------------------|-------|
| `gsm8k_32k`                        | 1404              | 1319  |
| `gpqa_diamond_cot_n_shot_32k`      | 2798              | 198   |
| `math500_32k`                      | 1373              | 500   |

All three use the `*_32k` task YAMLs in `tasks/` with `max_gen_toks` overridable
per run. Per-task `max_model_len = max_prompt + MG`, rounded up to a 256-multiple
(done inside the launcher when `USE_TASK_LEN=1`).

## vLLM infra settings (all runs)

From `run_eval_vllm.py`:

- `gpu_memory_utilization=0.70` — leaves headroom for graphs + Python overhead on 80 GB.
- `enable_prefix_caching=True` — 5-shot prompts share a long fewshot prefix.
- `dtype=float16`, `tensor_parallel_size=1`.
- `disable_log_stats=False` — emits periodic "Running/Swapped/GPU KV cache usage" so preemption is observable.
- `batch_size=128` (hardcoded in launcher) — must be ≥ task item count so lm_eval submits everything in one `generate()` call; otherwise vLLM drains between submits and throughput tanks 10-20×.
- `max_num_seqs` per preset — sized to KV-cache ceiling at the per-task max_model_len. Verify via log: "Max concurrency" must be ≥ `max_num_seqs`, else vLLM will preempt and swap.

### max_num_seqs sizing formula

At 80 GB with `gpu_util=0.70`, Llama-family GQA (128 KB/token cached):

```
total_kv_tokens ≈ 36 GB / 128 KB ≈ 280,000
max_num_seqs_ceiling = total_kv_tokens / max_model_len
```

For DSR1 at `max_model_len ≈ 35k`: 280k / 35k ≈ 8. We bumped to 16 when using
`MG=16k` (max_model_len drops to ~17-19k → ceiling ≈ 14-16).

### Environment overrides (launcher)

`scripts/launch_vllm_reasoning.sh` supports three env-var overrides:

- `MG_OVERRIDE=<int>` — truncate `max_gen_toks` for the adaptive first pass.
- `MAX_NS_OVERRIDE=<int>` — bump `max_num_seqs` when MG is lowered (more KV headroom).
- `LOG_SAMPLES=1` — enable `--log_samples` for per-item capture.

Example (DSR1 at 16k):

```bash
MG_OVERRIDE=16384 MAX_NS_OVERRIDE=16 LOG_SAMPLES=1 \
  TASKS='gsm8k_32k' \
  bash scripts/launch_vllm_reasoning.sh 2 dsr1-llama-8b \
    stream_name "--model fp16" out_stem
```

## Adaptive max_gen_toks policy (two-pass)

**Problem.** vLLM's scheduler waits for the slowest item in each batch-128 chunk
before submitting the next. A single item hitting a 16k or 32k cap can hold up
127 items that finished at 500 tokens. Wall-clock dominated by outliers.

**Policy.**

1. **Pass 1 (fast):** short `max_gen_toks` (2k-4k) + `LOG_SAMPLES=1`. Captures per-item generations to `logs/<out_stem>_samples.json`.
2. **Pass 2 (fair):** scan samples, find items where `len(gen_tokens) == MG`, rerun just those at full 32k. Merge metrics.

**Why:** for reasoning items, avg gen is usually <20% of the cap (e.g. Mistral
gsm8k avg ~683 tokens at 16k cap = 4%). Pass 1 covers the majority cheaply;
Pass 2 corrects the minority for fairness. Net: ~5-10× faster than a naive
single-pass 32k run.

See `memory/feedback_adaptive_max_gen_toks.md` for the short version.

## Calibration pipeline (SmoothKV only)

Driver: `run_smoothkv_calibrate.py`. Captures post-RoPE K/V stats via a
transformers `apply_rotary_pos_emb` monkey-patch.

Default invocation:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/vllm_env/bin/python run_smoothkv_calibrate.py \
  --model_path <model> \
  --num_samples 512 --seq_length 2048 --samples_per_channel 300000 \
  --output logs/calib/smoothkv_<model_stem>_perc_ns512.pt
```

Variants (dataset / n_calib / α=β) are generated from this base `.pt` via a
small in-script transform — see `make_variant()` in `tmp/mistral_dsr1_pipeline.sh`
for the canonical snippet. Core step (puremax, pair-max, α=β=1):

```python
import torch
b = torch.load("base.pt", weights_only=True, map_location="cpu")
s_K = b["max_k"].clamp(min=1e-5)
L, nh, D = s_K.shape
# Pair-max: RoPE pairs (d, d+D/2) must share scale to stay RoPE-mergeable
s_K = s_K.view(L, nh, D // 2, 2).max(dim=-1, keepdim=True).values \
        .expand(L, nh, D // 2, 2).reshape(L, nh, D).clone()
s_V = b["max_v"].clamp(min=1e-5)
torch.save({**b, "s_K": s_K, "s_V": s_V, "alpha": 1.0, "beta": 1.0,
            "pair_max_k": True}, "variant.pt")
```

Flags and their meaning:

- `--num_samples` (`ns`): calibration sample count.
- `--seq_length`: tokens per sample; 2048 is plenty for stat estimation.
- `--samples_per_channel`: reservoir size. **Irrelevant for max-based scales** (only `max_k`/`max_v` used). Keep at 300k to avoid 256 GB files from ns=1024 full-retention.
- `--dataset_config` / `--text_columns`: swap calibration corpus (wikitext, NuminaMath concat, etc.).

Canonical calib output filename pattern:

```
logs/calib/smoothkv_<model_stem>_perc_ns<N>[_<dataset_tag>]_puremax_a<α>b<β>_pair.pt
```

## Launch patterns

### Single run

```bash
bash scripts/launch_vllm_reasoning.sh <gpu> <preset> <stream_name> \
    "<method_args>" <out_stem>
```

e.g.:

```bash
TASKS='gsm8k_32k' bash scripts/launch_vllm_reasoning.sh \
  0 dsr1-llama-8b dsr1_skv_gsm8k \
  "--model smoothkv --calib_path logs/calib/smoothkv_dsr1-llama-8b_perc_ns512_puremax_a1b1_pair.pt" \
  dsr1-llama-8b_smoothkv_g128_perc_ns512_puremax_a1b1_pair
```

### Autonomous multi-phase pipelines

Per `memory/feedback_autonomous_pipelines.md`: overnight runs should be **one
tmux bash script**, not chained agent wake-ups. Canonical examples:

- `tmp/mistral_dsr1_pipeline.sh` — full 3-phase pipeline: calibs (GPU 0, 1) in parallel with FP16/FP8 downstream (GPU 2-7), then variant gen, then SmoothKV + pertoken on freed GPUs.
- `tmp/dsr1_phase3_16k.sh` — standalone phase-3 watcher: polls for 6 FP16/FP8 result files, then fires SmoothKV + pertoken × 3 on GPU 2-7.

Pattern: `{ … } > $LOG 2>&1` at script level so everything is captured; ends
with a unique `DONE_<label>` marker line for post-run grep.

## Preemption monitoring

After every vLLM launch, verify no preemption (per
`memory/feedback_vllm_preemption_check.md`):

```bash
grep -E "(Max concurrency|Swapped|preempt)" logs/run_out/<stream>_vllm.log
```

- `Max concurrency = X` must be ≥ `max_num_seqs`. If lower, vLLM didn't get the KV budget it wanted (bump `gpu_memory_utilization` or drop `max_model_len`).
- `Swapped: 0 reqs` should hold throughout the run. Any non-zero = preemption happened = re-size `max_num_seqs` down.

## Aggregation and reporting

Result files land at:

```
logs/<task>_<out_stem>_vllm_results.json         # always
logs/<task>_<out_stem>_vllm_samples.json         # when LOG_SAMPLES=1
```

Report generators:

- `generate_ablation_report.py` — 4-table HTML summary of the Llama3 SmoothKV matrix. Auto-parses `(method, ds, ns, ab)` tuples from filenames.
- `generate_vllm_report.py` — cross-model / cross-method comparison table. Auto-discovers `logs/*_vllm_results.json`; add new model stems to `MODEL_DISPLAY`.

## Git / reproducibility

- Branch: `pertoken-experiments` (vLLM scaffold + per-task max_model_len sizing + adaptive MG overrides).
- Results are **not** committed — only code, configs, scripts, and docs. Regenerate via the above scripts.
- Calibration `.pt` files are large (hundreds of MB to hundreds of GB) — not committed. Regenerate from `run_smoothkv_calibrate.py`.

## Files touched by this workflow

- `run_eval_vllm.py` — vLLM entrypoint; `--log_samples`, `--max_model_len`, `--max_num_seqs`, method dispatch.
- `run_smoothkv_calibrate.py` — calibration driver; post-RoPE capture via RoPE monkey-patch; `--dataset_config`, `--text_columns`.
- `vllm_custom/patches.py` — installs fake-quant hooks into `LlamaAttention.forward`.
- `scripts/launch_vllm_reasoning.sh` — preset + per-task max_model_len + MG/MAX_NS/LOG_SAMPLES overrides + BS=128.
- `scripts/measure_prompt_lens.py` — prompt-length audit for sizing `max_model_len`.
- `generate_ablation_report.py`, `generate_vllm_report.py` — aggregation.
- `docs/GUIDE_QWEN3_PORT.md` — Qwen3 port guide.
- `docs/GUIDE_LLAMA3_INSTRUCT.md` — Llama3-specific setup notes.
- `docs/GUIDE_VLLM_PATH.md` — vLLM patching mechanics.
- `docs/REPLICATION.md` — per-paper reproducibility.
