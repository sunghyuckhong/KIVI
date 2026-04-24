# Guide: Llama-3-8B-Instruct full-matrix eval

Hand this to another Claude to replicate the Llama-3-8B-Instruct evaluation on a fresh pod.

## What you're running

9 methods × 5 tasks on `meta-llama/Meta-Llama-3-8B-Instruct`:

**Methods** (9 streams, each chains all 5 tasks):
1. FP16
2. KIVI-2 (g=32, r=128)
3. Naive INT4 per-token (g=128, r=0)
4. DeepSeekFP8 (g=128)
5. SmoothKV α=0.75 pair-mergeable
6. SmoothKV p=90 pair-mergeable
7. SmoothKV p=95 pair-mergeable
8. SmoothKV p=99 pair-mergeable
9. SmoothKV p=99.9 pair-mergeable

**Tasks**: `truthfulqa_gen`, `coqa` (both at 256 default), `gsm8k_32k`, `gpqa_diamond_cot_n_shot_32k`, `math500_32k` (reasoning at `--max_gen_toks 4096`).

**Why these settings:** Llama-3-8B-Instruct has `max_position_embeddings=8192`. Half-context rule: `max_gen_toks = 4096`. Leaves 4096 tokens for prompt (plenty for few-shot MATH).

## Prerequisites

Clone and checkout the branch with pair-mergeable + 32k-reasoning infra:
```bash
git clone https://github.com/sunghyuckhong/KIVI.git
cd KIVI
git checkout pertoken-experiments   # commit 3db97bd or later
export HF_TOKEN=<your_hf_token>
# If your repo root is not /workspace/KIVI, set:
export REPO_ROOT=$(pwd)
```

## Environment — you need ONE env with transformers 4.43+ and lm-eval 0.4.2+

Pinned spec (from the original paper-matching setup):
```bash
python3.10 -m venv /opt/modernenv
/opt/modernenv/bin/pip install --upgrade pip setuptools wheel
/opt/modernenv/bin/pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0
/opt/modernenv/bin/pip install "transformers==4.43.1" "lm_eval==0.4.2" accelerate datasets sentencepiece "numpy<2"

cd quant && /opt/modernenv/bin/pip install -e . && cd ..
```

**Tested-working drift (April 2026 pod):** `torch==2.4.0+cu121`, `transformers==4.57.6`,
`lm_eval==0.4.x`, `flash_attn==2.8.3`. If modernenv has drifted to newer versions,
the pipeline still works but requires matching CUDA_HOME when rebuilding kivi_gemv:

```bash
# If torch is cu121:
cd quant && rm -f kivi_gemv*.so
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH \
    /opt/modernenv/bin/pip install -e . --no-build-isolation
cd ..

# flash_attn is not pulled in by default — install it for the FP16 path:
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH \
    /opt/modernenv/bin/pip install flash-attn==2.8.3 --no-build-isolation
```

Sanity check:
```bash
/opt/modernenv/bin/python -c "
import torch, transformers, lm_eval, kivi_gemv
print(f'torch={torch.__version__}  transformers={transformers.__version__}')
print(f'GPUs visible: {torch.cuda.device_count()}  kivi_gemv OK')
"
```

Smoke test end-to-end (~3 min):
```bash
CUDA_VISIBLE_DEVICES=0 /opt/modernenv/bin/python run_eval.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model fp16 --k_bits 16 --v_bits 16 \
    --task truthfulqa_gen --batch_size 16
# Should finish in ~3 min, write logs/truthfulqa_gen_meta-llama-3-8b-instruct_fp16_results.json
```

## Steps

### Step 1 — calibration (~30 min, 1 GPU)

```bash
mkdir -p logs/calib logs/run_out
/opt/modernenv/bin/python run_smoothkv_calibrate.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --alpha 0.5 --beta 0.5 \
    --samples_per_channel 10000 \
    --output logs/calib/smoothkv_meta-llama-3-8b-instruct_perc.pt
```

Output file is ~5 GB (contains both max stats + reservoir samples for percentile variants).

### Step 2 — generate pair-mergeable variants (~2 min, CPU)

```bash
CALIB=logs/calib/smoothkv_meta-llama-3-8b-instruct_perc.pt

# α=0.75 pair variant (needs max_k / max_q / max_v — all present in $CALIB)
python scripts/make_alpha_variants.py \
    --base "$CALIB" --alphas 0.75 --pair_max_k

# Percentile variants (needs reservoir samples — present in $CALIB)
python scripts/make_percentile_variants.py \
    --base "$CALIB" --pk 90 95 99 99.9 --symmetric
```

Produces 5 files:
```
logs/calib/smoothkv_meta-llama-3-8b-instruct_a0.75_pair.pt
logs/calib/smoothkv_meta-llama-3-8b-instruct_pairK{90,95,99,99p9}_pV{90,95,99,99p9}.pt
```

Both generators apply pair-max on s_K (Eq. 7) by default; pass `--no-pair-max-k` for non-mergeable baseline.

### Step 3 — launch 9 eval streams (4 GPUs in waves, ~2h)

Use the bundled launcher (expects 4 GPUs):
```bash
bash scripts/phaseX_llama3_launch.sh
```

Or adapt `scripts/phaseX_instruct_launch.sh` which already targets
`meta-llama/Meta-Llama-3-8B-Instruct` + `mistralai/Mistral-7B-Instruct-v0.2`.

Either launcher chains python invocations with `&&` (first failure aborts rest).
Each stream prints `DONE_<name>` on success or `FAIL_<name>` on error so an
overnight watcher can tell the two apart.

### Step 4 — monitor

Launch the watcher in a tmux window so it survives disconnects:
```bash
tmux new-window -n watcher "bash scripts/overnight_watcher_v3.sh; read"
tail -f logs/run_out/overnight_watcher.log
```

Individual stream logs at `logs/run_out/pX_meta_<method>.log`.

### Step 5 — regenerate HTML reports

After all 9 streams finish (or any time — will show partial):
```bash
/opt/modernenv/bin/python generate_summary.py
/opt/modernenv/bin/python generate_simplified_report.py
```

Opens `summary_latest.html` + `summary_simplified_latest.html`. Readers prefer
`_32k` task keys and fall back to legacy names.

## Result filenames

Modern-env runs write to `logs/` with this pattern:
```
{task}_meta-llama-3-8b-instruct_{method_tag}_results.json
```

Where `{method_tag}` depends on method:
- `fp16`
- `kivi_res128` (KIVI-2 g=32 r=128)
- `pertoken_int4_flat_noresidual` (pertoken g=128 r=0)
- `fp8_g128_noresidual`
- `smoothkv_g128` + calib-path-derived suffix for SmoothKV

## Per-config batch sizes

The bundled launcher uses `--batch_size 16` everywhere. For Llama-3-Instruct at
4k gen on 80GB A100, bs=16 fits all methods (GQA with 8 kv_heads keeps KV cache
manageable). If you hit OOM, halve to bs=8.

## Expected runtime

On 4× A100-80GB at bs=16:
- Calibration: ~30 min
- Variant generation: ~2 min (CPU)
- 9 streams × ~45 min / 4 GPUs in parallel = ~2h wallclock
- **Total: ~3h from clean start**

## Recycling from previous runs

If the target pod already has `logs/*meta-llama-3-8b-instruct*_results.json`
files from a previous experiment, TQA and CoQA results (ran at HFLM default 256)
are reusable — they're the standard setting. Reasoning task results at old
settings need re-run. The launcher will just overwrite.

## Post-run verification

Quick sanity check that reasoning tasks ran with correct gen budget:
```bash
# Should show "max_new_tokens=4096" (or similar) somewhere in the log tail
grep -h "max_new_tokens\|max_gen_toks" logs/run_out/pX_meta_fp16.log | head -5
```

If the log shows `max_new_tokens=256` instead, `--max_gen_toks` CLI flag isn't
being threaded through — check that `run_eval.py` has the `--max_gen_toks` arg
and it's in `gen_kwargs` at `simple_evaluate()` time.

## Known pitfalls

- **Empty prompt error** (`IndexError: index -1 is out of bounds for dimension 1 with size 0`):
  You set `max_gen_toks` ≥ model context. Llama-3-8B has 8k context → max 4096 gen.
- **OOM on gsm8k/math500**: lower `--batch_size` in the launcher.
- **`kivi_gemv not found`** or `undefined symbol` on import: kivi_gemv was built against
  a different torch ABI. Rebuild inside modernenv: `cd quant && rm -f kivi_gemv*.so &&
  /opt/modernenv/bin/pip install -e . --no-build-isolation`. If torch is cu121 but
  nvcc defaults to cu118, prepend `CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH`.
- **`ImportError: FlashAttention2 has been toggled on, but ... flash_attn seems to be not installed`**:
  `run_eval.py` fp16 path hard-codes `attn_implementation="flash_attention_2"`. Install
  `flash-attn==2.8.3` (see Environment section). If your GPU is pre-Ampere (V100 / sm_70),
  you can't use flash-attn — edit `run_eval.py` fp16 branch to drop `attn_implementation`.
- **`cd /workspace/KIVI: No such file or directory`** (in tmux windows): the launcher
  scripts used to hardcode `/workspace/KIVI`. Current launchers default to
  `/home/home-mcl/sunghyuck/kv_cache_compression/KIVI` and respect `REPO_ROOT` env var.
  Set `REPO_ROOT=$(pwd)` before `bash scripts/phaseX_llama3_launch.sh`.
- **No `_32k` tasks**: make sure `tasks/gsm8k/`, `tasks/gpqa/`, `tasks/math500/`,
  `tasks/aime/` all exist (they're in the repo, shouldn't be missing).
- **HF_TOKEN not set** but `huggingface-cli login` already ran: the launcher requires
  `HF_TOKEN` env var explicitly. Load it from the cli token file:
  `export HF_TOKEN=$(cat ~/.cache/huggingface/token)`.

## Minimal test (skip full matrix)

If you just want to verify the pipeline end-to-end on 1 GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
/opt/modernenv/bin/python run_eval.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model fp16 --k_bits 16 --v_bits 16 \
    --task truthfulqa_gen --batch_size 16
```
Should finish in ~3 min, write `logs/truthfulqa_gen_meta-llama-3-8b-instruct_fp16_results.json`.
