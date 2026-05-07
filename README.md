# SmoothKV Evaluation

A harness for accuracy evaluation of KV-cache fake-quantization methods on
reasoning benchmarks. Each (model × method × task) cell runs through a
two-pass adaptive eval with a graph-verification trust gate at each pass.

## Methods

| Method | Description | Calibration | Per-step cost |
|--------|-------------|:-----------:|---------------|
| `bf16` / `fp16` | Unquantized baseline | none | none |
| `fp8` | FP8 E4M3, group=128, per-token | none | 1× round-trip |
| `pertoken` | INT4, group=128, per-token | none | 1× round-trip |
| `smoothkv` | Per-step SmoothQuant rescale + INT4 round-trip | `s_K`, `s_V` `.pt` | `÷ s_K` + INT4 + `× s_K` |
| `smoothkv_fused` | Load-time fold of `s_K` / `s_V` into projection weights, then plain pertoken at inference | `s_K`, `s_V` `.pt` (HUK for q-norm models) | same as `pertoken` |

**Variant tags accepted by `--variant` (CLI shorthands for the methods above):**

| Variant flag | Method | Granularity of `s_K` (Qwen3-8B example) |
|---|---|---|
| `bf16` / `fp8` / `pertoken` | as named | n/a |
| `smkv` | `smoothkv_fused` (HUK auto-on for q-norm models, halfpair) | per-(layer, head_dim) — shared across heads → 36 × 128 = **4,608** unique values |
| `smkv_per_channel` | `smoothkv` runtime kernel, `--no_head_uniform_k`, no halfpair | per-(layer, kv_head, head_dim) → 36 × 8 × 128 = **36,864** unique values |

## Tasks

| Task | Domain | Set size | Canonical metric |
|------|--------|:--------:|------------------|
| **gsm8k_cot** | Grade-school math | 2638 | `exact_match,flexible-extract` (lm-eval flex regex) |
| **minerva_math500** | Competition math | 500 | `math_verify,none` (sympy boxed-aware) |
| **gpqa_main_cot_n_shot** | Graduate Q&A | 1170 | `exact_match,flexible-extract` |

## Models

| Family | Sizes tested | Tensor parallel |
|--------|--------------|:---------------:|
| Qwen3 | 8B, 32B, 30B-A3B (MoE) | 1 / 2 / 2 |
| Llama-3 | 8B-Instruct (default), any HF id | 1 (8B) / 2-4 (70B) |
| Mistral | 7B-Instruct-v0.2 | 1 |
| EXAONE-4.5 | 33B | 2 |

## Structure

```
smoothkv-exp-clean/
├── Makefile                       # setup / test / per-family sweeps
├── README.md                      # this file
├── pytest.ini
├── requirements_smoothkv.txt
├── run_eval_vllm.py               # single-pass eval driver (lm-eval + vLLM)
├── run_smoothkv_calibrate.py      # SmoothKV calibration (s_K, s_V) generator
├── quant/
│   ├── __init__.py
│   └── smoothkv_quant.py          # calibration math (SmoothQuant formulas)
├── scripts/
│   ├── adaptive_pass2.py          # pass-2 driver (MG=32k retry + merge + rescore)
│   ├── parallel_sweep.py          # auto-parallel cell scheduler over idle GPUs
│   ├── verify_compiled_graph.py   # graph-verification trust gate
│   ├── run_eval_qwen3.sh          # per-family runner: Qwen3 8B / 32B
│   ├── run_eval_llama.sh          # per-family runner: Llama-3
│   ├── run_eval_mistral.sh        # per-family runner: Mistral (shim over llama)
│   └── run_eval_exaone.sh         # per-family runner: EXAONE-4.5
└── tests/
    ├── conftest.py
    ├── test_quant_dtype.py        # dtype + MSE bounds on quant kernels
    └── test_verify_graph.py       # FX-graph capture asserts
```

## Quick Start

### 1. Setup

The actual KV fake-quantization logic lives in the
[`sunghyuckhong/vllm-compression-part`](https://github.com/sunghyuckhong/vllm-compression-part)
fork (branch `kv_cache_quant`, pinned in `Makefile`). `make setup` clones it
into `VLLM_FORK_PATH` if missing and builds it from source (~15-30 min on
first run).

```bash
git clone <fork>/KIVI.git -b smoothkv-exp-clean
cd KIVI
make setup        # auto-clones vllm fork, creates .venv, builds at pinned commit, installs deps
make test         # unit tests (verify-graph + dtype/MSE bounds)
```

Override the fork URL or path if you have your own copy:

```bash
make setup VLLM_FORK_URL=git@github.com:my-org/vllm-compression-part.git \
          VLLM_FORK_PATH=$HOME/src/vllm-compression-part
```

To bump the vllm fork later, edit `VLLM_FORK_COMMIT` in the `Makefile` and run:

```bash
make setup-fork   # re-checkout fork at pinned commit + reinstall (skips venv/deps)
```

### 2. Generate calibration (SmoothKV variants only)

`bf16`, `fp16`, `fp8`, `pertoken` need no calibration. For `smoothkv` and
`smoothkv_fused`, generate `(s_K, s_V)` once per (model, α, β):

```bash
python run_smoothkv_calibrate.py \
    --model Qwen/Qwen3-8B \
    --num_samples 512 --seq_length 2048 \
    --alpha 1.0 --beta 1.0 \
    --apply_chat_template \
    --output logs/calib/smoothkv_qwen3-8b_a1b1_chat.pt
```

For `smoothkv_fused` on models with `q_norm`/`k_norm` (Qwen3, Qwen3-MoE,
EXAONE-4), pass `--head_uniform_k` to produce the `_huk_*` variant — the
fold path requires head-uniform `s_K` because RMSNorm doesn't commute with
per-channel scaling. The per-family runners take care of this automatically.

### 3. Run an eval

The simplest entry point — sweeps a single (model, family) over all
4 methods × 3 tasks, runs cells in parallel across idle GPUs:

```bash
make run-qwen3-8b               # 1 GPU per cell, all idle GPUs used
make run-qwen3-32b              # 2 GPUs per cell (TP=2)
make run-qwen3-30b-a3b          # 2 GPUs per cell (TP=2; MoE, ~3B active)
make run-llama                  # default LLAMA_MODEL=Meta-Llama-3-8B-Instruct
make run-mistral                # default Mistral-7B-Instruct-v0.2
make run-exaone                 # TP=2

make run-all                    # qwen3-8b + llama + mistral
```

Behind the scenes, each invocation goes through `scripts/parallel_sweep.py`
which detects idle GPUs (memory.used < 2GB), chunks them into TP-sized
streams, and greedily schedules `(variant, task)` cells across the streams.
Per-cell, the runner does pass1 at MG=4k → verify-graph → pass2 at MG=32k
→ verify-graph → write `_adaptive_results.json`.

To run a single cell manually:

```bash
bash scripts/run_eval_qwen3.sh \
    --size 8b --variant smkv --task gsm8k_cot \
    --gpus 0 --alpha 1.0 --beta 1.0
```

## Configuration

All `make run-*` targets read these variables. Override on the command line.

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUS` | `auto` | `auto` = scan idle (memory < 2GB), chunk by `--tp`. Or comma-separated explicit pool: `0,1,2,3`. `0,1` for TP=2 = single sequential stream. |
| `VARIANTS` | `bf16 fp8 pertoken smkv` | Methods to sweep. |
| `TASKS` | `gsm8k_cot minerva_math500 gpqa_main_cot_n_shot` | Tasks to sweep. |
| `NS` | `512` | SmoothKV calibration sample count. |
| `ALPHA` | `1.0` | SmoothQuant α (K-side migration strength). |
| `BETA` | `1.0` | SmoothQuant β (V-side power). |
| `LLAMA_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | HF id for `make run-llama`. |
| `MISTRAL_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HF id for `make run-mistral`. |
| `VLLM_FORK_URL` | `https://github.com/sunghyuckhong/vllm-compression-part.git` | Fork remote — `make setup` clones from here if `VLLM_FORK_PATH` is missing. |
| `VLLM_FORK_PATH` | `/workspace/sunghyuck/vllm-compression-part` | Where the fork lives on disk. |
| `VLLM_FORK_COMMIT` | (pinned in Makefile) | vllm-compression-part `kv_cache_quant` commit to install. |
| `TORCH_VERSION` | `2.11.0` | torch version pinned by the vllm fork's `pyproject.toml`. `make setup` detects driver major version and installs the matching cu128 / cu130 wheel (cu128 for driver major ≥ 555, cu130 for ≥ 575). Aborts with a clear error if driver is older. |
| `MIN_DRIVER_MAJOR` | `555` | Minimum NVIDIA driver major version compatible with the vllm fork (CUDA 12.8 → driver 555+). Setup errors out below this. |
| `PY` | `.venv/bin/python` (set by Makefile) | Python interpreter used by the runners. The runners default to `./.venv/bin/python` if `PY` is unset; the Makefile exports `PY` so subprocesses inherit it. Override with `PY=/path/to/python make run-qwen3-8b ...` if you need a different env. |
| `FORCE` | `0` | Set to `1` to bypass the runner's "skip if outputs exist" gate. Forces both pass-1 and pass-2 to re-run; calibration is unaffected. See "Re-running an existing cell" below. |

Example overrides:

```bash
make run-qwen3-8b VARIANTS="bf16 smkv"            # 2 methods only
make run-qwen3-8b VARIANTS="smkv_per_channel"        # per-(head, channel) smoothing (Qwen3 + Llama-3 + Mistral)
make run-qwen3-8b VARIANTS="..." NS=1024             # bump SmoothKV calib sample count
make run-llama TASKS=gsm8k_cot                     # 1 task only
make run-qwen3-32b GPUS=0,1,2,3                    # explicit 2-stream pool
make run-qwen3-8b ALPHA=0.5 BETA=0.5               # different SmoothKV α/β
make run-llama LLAMA_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Auto-parallel scheduling

When `GPUS=auto` (default), `parallel_sweep.py` picks idle GPUs via
`nvidia-smi` and runs as many concurrent cells as fit. On an 8-GPU host:

| Target | TP | Streams | Concurrent cells |
|--------|:--:|:-------:|:----------------:|
| `make run-qwen3-8b` | 1 | 8 | 8 |
| `make run-qwen3-32b` | 2 | 4 | 4 |
| `make run-llama` | 1 | 8 | 8 |
| `make run-exaone` | 2 | 4 | 4 |

Output during a run:

```
[parallel] auto-detected idle GPUs (< 2000MB): [0, 1, 2, 3, 4, 5, 6, 7]
[parallel] 4 stream(s) × 2 GPU(s): [0,1], [2,3], [4,5], [6,7]
[parallel] 12 cells: variants=['bf16', 'fp8', 'pertoken', 'smkv'] × tasks=['gsm8k_cot', 'minerva_math500', 'gpqa_main_cot_n_shot']
[parallel] START  bf16       gsm8k_cot                          gpu=[0,1]  → logs/run_out/parallel_bf16_gsm8k_cot_g0_1.log
...
[parallel] DONE   bf16       gsm8k_cot                          gpu=[0,1]  ✅ PASS  1842s  (1/12 cells; elapsed 1842s)
...
[parallel] ✅ all 12 cells passed in 11052s (4-way parallel)
```

## Results

Each cell writes three artifacts under `logs/`:

| File | Purpose |
|------|---------|
| `<task>_<model>_<variant>_chat_vllm_results.json` | Pass-1 score (preliminary; doesn't account for truncation) |
| `<task>_<model>_<variant>_chat_vllm_samples.json` | Per-item generations from pass 1 |
| `<task>_<model>_<variant>_chat_vllm_adaptive_results.json` | **Final headline number** (pass-1 + pass-2 retried truncated items, merged + rescored via `scripts/scoring.py`, which delegates to lm-eval's own filter classes — `lm_eval.filters.get_filter` + `lm_eval.api.metrics.exact_match_hf_evaluate` — so the metric is byte-equivalent to what lm-eval would have produced if it had scored the merged set itself) |

Plus per-pass logs in `logs/run_out/*_pass{1,2}_<task>.log`.

### Re-running an existing cell

By default the runners **skip** any cell whose pass-1 `_results.json` and
`_adaptive_results.json` already exist. To force a re-run (e.g. to pick up
a new verify-graph stamp after upgrading the fork, or because the previous
output didn't carry a stamp), pass `FORCE=1`:

```bash
# Re-run only Qwen3-8B BF16 on gsm8k_cot, overwriting existing outputs:
make run-qwen3-8b VARIANTS=bf16 TASKS=gsm8k_cot FORCE=1
```

`FORCE=1` skips both the pass-1 and pass-2 skip checks, but does **not**
regenerate calibration `.pt` files — those are deterministic w.r.t.
(model, dataset, NS, α, β) so manual deletion is the right escape hatch
if you need a fresh calib.

Alternative: move the existing artifacts to `logs/_unstamped/` first if
you want to keep them as a baseline for comparison.

Read the headline number:

```bash
python -c "
import json
d = json.load(open('logs/gsm8k_cot_qwen3-8b_smoothkv_fused_g128_..._chat_vllm_adaptive_results.json'))
print(d['results']['gsm8k_cot']['exact_match,flexible-extract'])
"
```

Example file:

```json
{
  "results": {
    "gsm8k_cot": {
      "exact_match,strict-match":      0.595,
      "exact_match,flexible-extract":  0.909,
      "exact_match_n,strict-match":    2638,
      "exact_match_n,flexible-extract": 2638
    }
  },
  "n_truncated_pass1": 178,
  "n_total":           2638,
  "pass1_mg":          4096,
  "pass2_mg":          32768
}
```

## Trust gate

A cell is trusted iff:

- **only pass-1 ran** (pass-2 was skipped because `pass1_mg == pass2_mg`,
  e.g. Llama-3-8B with mml=8192): pass-1's log carries `[verify-graph] [PASS]`, **or**
- **both passes ran**: each pass's log carries `[verify-graph] [PASS]`.

Anything else is a `[FAIL]` and the cell is not trustworthy.

How "what ran" is decided per cell:

| Scenario | When | What's enforced |
|---|---|---|
| **A** — pass-1 only | `pass1_mg == pass2_mg` (e.g. Llama mml=8192). Runner copies `_results.json` → `_adaptive_results.json` and never invokes pass-2. | Pass-1 must emit `[PASS]`. |
| **B** — both passes | `pass1_mg < pass2_mg` and pass-1 had ≥1 truncated sample. Pass-2 launches vLLM, retries the truncated subset at MG=32k, merges + rescores. | Pass-1 AND pass-2 must each emit `[PASS]`. |
| **C** — both passes, pass-2 noop | `pass1_mg < pass2_mg` but every pass-1 response fit under MG=4k. Pass-2 process runs but doesn't launch vLLM (nothing to retry). | Pass-1 AND pass-2 must each emit `[PASS]`. Pass-2's stamp is the noop variant emitted by `adaptive_pass2.py` directly. |
| **D** — cached skip | `FORCE=0` and outputs already exist on disk. The runner skips both passes for this invocation. | Nothing — the cell's existing stamps from a prior invocation are trusted. To re-validate, set `FORCE=1`. |

| Stamp | What it means |
|---|---|
| `[PASS]` | The compiled FX graph contains the expected `vllm_kv_quant::*` ops (or, for `bf16`, no forbidden quant ops). Pass-2 also emits `[PASS]` when it had no truncated samples to retry — pass-1's stamp covers the actual eval. |
| `[FAIL]` | The graph was inspected and was **wrong** — e.g. expected ops missing (silent fallback to bf16 via a stale compile-cache hit), or forbidden quant ops present in a `bf16` baseline. **Don't trust the number; clear the cache and rerun.** |

The gate is enforced at three levels:

1. **Unit-level** (`tests/test_verify_graph.py`, run via `make test`) — runs each
   method through `torch._dynamo.export` and asserts the captured FX graph
   contains the expected ops.
2. **Eval-level** (`scripts/verify_compiled_graph.py`, called at the end of
   `run_eval_vllm.py` and `scripts/adaptive_pass2.py`) — greps inductor's
   `computation_graph.py` dump for the kernel names and prints the stamp.
3. **Pipeline-level** (each shell runner) — greps the log; aborts with
   `exit 2` unless it sees a `[PASS]`. A bare `[verify-graph] ...` line
   without an explicit verdict does **not** satisfy the gate.

(Older runs may carry the equivalent `✅ PASS` / `❌ ...` emoji form;
the gate accepts both.)

Per-method expected substrings:

| Method | Expected in FX graph | Forbidden |
|--------|---------------------|-----------|
| `bf16` / `fp16` | (none) | `quant_and_pack`, `fake_quantize` |
| `fp8` | `fake_quantize_dequantize_fp8` | — |
| `pertoken` | `quant_and_pack_vcache`, `unpack_and_dequant_vcache` | — |
| `smoothkv` | `quant_and_pack_vcache` | — |
| `smoothkv_fused` | `quant_and_pack_vcache` | — |

If verify-graph fails, the result is not trustable: the runtime almost
certainly fell back to BF16 (e.g. via a stale compile-cache hash collision).

## `max_gen_tokens` rule

The runners derive `max_gen_tokens` from the model's native context length:

```
pass2_mg = 32768                      if model_max_len >= 32768
         = model_max_len / 2          otherwise
pass1_mg = min(4096, pass2_mg)
pass2 skipped if pass1_mg == pass2_mg (pass1 stands as the result)
```

Concretely:

| Model | `model_max_len` | `pass1_mg` | `pass2_mg` |
|-------|:---------------:|:----------:|:----------:|
| Llama-3-8B-Instruct | 8192 | 4096 | 4096 (pass2 skipped) |
| Llama-3.1-8B-Instruct | 131072 | 4096 | 32768 |
| Mistral-7B-Instruct-v0.2 | 32768 | 4096 | 32768 |
| Qwen3-8B / Qwen3-32B | 40960 | 4096 | 32768 |
| EXAONE-4.5-33B | 32768 | n/a | 32768 (single-pass) |

Each runner prints the resolved values once at startup:

```
[mg] model_max_len=40960  →  pass1_mg=4096 (mml=5632), pass2_mg=32768 (mml=34304)
```

## Common commands

```bash
make help          # list available targets
make print-pin     # print pinned vllm-compression-part commit
make setup         # build vllm fork + venv + Python deps (~15-30 min)
make setup-fork    # re-checkout the pinned fork commit + reinstall (faster)
make test          # run unit tests
make run-qwen3-8b  # full sweep on Qwen3-8B
make run-all       # qwen3-8b + llama + mistral (skips 32b/exaone)
make clean         # remove .venv (keeps logs/ and calib/)
```

## Architecture

```
                       ┌──────────────────────────────────────────┐
                       │ vllm-compression-part @ kv_cache_quant     │
                       │                                          │
                       │  vllm/config/kv_cache_quant.py           │
                       │      KVCacheQuantConfig                  │
                       │                                          │
                       │  vllm/.../quantization/kv_fake_quant/    │
                       │      kernels.py     (FP8 + INT4 ops)     │
                       │      layer_hooks.py (LayerKVQuantState   │
                       │                      + apply_kv_quant)   │
                       │      fusion.py      (smoothkv_fused      │
                       │                      weight folding)     │
                       │                                          │
                       │  vllm/.../attention/attention.py         │
                       │      __init__: attach_kv_quant_to_layer  │
                       │      forward:  apply_kv_quant            │
                       │                                          │
                       │  vllm/v1/worker/gpu_worker.py            │
                       │      load_model: maybe_run_post_load_fusion │
                       └──────────────────────────────────────────┘
                                          ▲
                                          │   LLM(kv_cache_quant_config=...)
                                          │
                       ┌──────────────────────────────────────────┐
                       │ this repo                                │
                       │  run_eval_vllm.py        single-pass     │
                       │  scripts/adaptive_pass2  pass-2 retry    │
                       │  scripts/run_eval_*.sh   per-family      │
                       │  scripts/parallel_sweep  auto-scheduler  │
                       │  scripts/verify_*        trust gate      │
                       │  run_smoothkv_calibrate  calib gen       │
                       └──────────────────────────────────────────┘
```

The vllm fork wires KV fake-quant into the shared `Attention` class via three
inline reads — one in `__init__` (registers per-instance state from
`KVCacheQuantConfig`), one in `forward` (dispatches to the configured method's
QDQ kernels), and one in `Worker.load_model` (folds smoothkv_fused scales
into projection weights). No monkey-patches.

This repo wraps that with eval drivers, calibration generation, and the
verify-graph trust gate.
