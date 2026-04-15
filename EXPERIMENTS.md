# KIVI KV-Cache Quantization Experiments

Experiments evaluating per-token key quantization variants on top of [KIVI](https://github.com/jy-yuan/KIVI), using **Mistral-7B-Instruct-v0.2** with 2-bit KV cache compression.

---

## What This Branch Adds

The original KIVI uses:
- **Keys**: per-channel quantization (groups along the sequence dimension T)
- **Values**: per-token quantization (groups along head_dim D)

This fork adds a **per-token key** variant where both keys and values are quantized per-token (groups along head_dim), enabling a clean ablation across:

| Experiment | Key quant | Group size | Residual buffer |
|---|---|---|---|
| FP16 baseline | none | — | — |
| KIVI default | per-channel | 32 | 32 tokens |
| PerToken, group=32, residual=32 | per-token | 32 | 32 tokens |
| PerToken, group=32, residual=0 | per-token | 32 | none |
| PerToken, group=128 (flat), residual=32 | per-token | 128 (=head\_dim) | 32 tokens |
| PerToken, group=128 (flat), residual=0 | per-token | 128 (=head\_dim) | none |

### Key concepts

**Per-channel vs per-token** — controlled purely by whether the key tensor is transposed before quantization:
```python
# Per-channel (original KIVI): groups along sequence dim T
triton_quantize_and_pack_along_last_dim(key.transpose(2, 3).contiguous(), ...)
# shape: (B, nh, T, D) → (B, nh, D, T) → groups along T

# Per-token (this fork): groups along head_dim D
triton_quantize_and_pack_along_last_dim(key.contiguous(), ...)
# shape: (B, nh, T, D) → groups along D
```

**Group size** — number of elements that share one scale/min within a token's head_dim:
- `group_size=32` → 128/32 = 4 groups per token (finer, better quality)
- `group_size=128` → 128/128 = 1 group per token (flat, coarser)

**Residual buffer** — the most recent `residual_length` tokens kept in FP16:
- `residual_length=32` → last 32 tokens always FP16
- `residual_length=0` → every token quantized immediately

**Quantization scheme**: asymmetric min-max per group, stored as `(scale, min)` per group.

---

## Hardware Requirements

- Tested on **NVIDIA V100 32GB** (compute capability 7.0)
- FlashAttention **disabled** — requires Ampere (sm_80+); set `config.use_flash = False`
- One GPU per experiment; 32GB is sufficient for Mistral-7B fp16 + batch_size=16
- For batch_size=16 with 5-shot prompts, ~18–22GB peak VRAM usage observed

---

## Environment Setup

### 1. Create conda environment

```bash
conda create -n kivi python=3.10 -y
conda activate kivi
```

### 2. Install PyTorch (match your CUDA version)

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install KIVI dependencies

```bash
cd KIVI
pip install -r requirements.txt
pip install -e . --no-build-isolation   # installs the quant CUDA package
```

> **Note**: The `quant` package compiles custom CUDA kernels and requires torch to be installed first. Always use `--no-build-isolation`.

### 4. Install lm-eval

```bash
pip install lm-eval==0.4.2
```

### 5. Authenticate with HuggingFace

Two gated resources require HF auth:
- `mistralai/Mistral-7B-Instruct-v0.2` (model weights)
- `Idavidrein/gpqa` (GPQA benchmark dataset)

```bash
huggingface-cli login   # paste your HF token when prompted
```

Request access at:
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- https://huggingface.co/datasets/Idavidrein/gpqa

---

## New Files

```
models/
  mistral_kivi_pertoken.py      # Per-token key quantization attention + model stack

quant/
  new_pack.py                   # + dequantize_cache_pertoken() added

run_gsm8k.py                    # GSM8K — KIVI default (modified: saves JSON, bs=16)
run_gsm8k_fp16.py               # GSM8K — FP16 baseline
run_gsm8k_pertoken.py           # GSM8K — PerToken, group=32, residual=32
run_gsm8k_pertoken_noresidual.py# GSM8K — PerToken, group=32, residual=0
run_gsm8k_pertoken_flat.py      # GSM8K — PerToken, group=128, residual=32
run_gsm8k_pertoken_flat_noresidual.py # GSM8K — PerToken, group=128, residual=0

run_gpqa_kivi.py                # GPQA Diamond — KIVI default
run_gpqa_fp16.py                # GPQA Diamond — FP16 baseline
run_gpqa_pertoken.py            # GPQA Diamond — PerToken, group=32, residual=32
run_gpqa_pertoken_noresidual.py # GPQA Diamond — PerToken, group=32, residual=0
run_gpqa_pertoken_flat.py       # GPQA Diamond — PerToken, group=128, residual=32
run_gpqa_pertoken_flat_noresidual.py  # GPQA Diamond — PerToken, group=128, residual=0

generate_report.py              # Reads all logs/JSON and prints results table
wait_and_report.sh              # Polls for completion, auto-runs generate_report.py
```

### Modified files (from original KIVI)

**`models/mistral_kivi.py`** — three transformers 4.43 compatibility fixes:
1. `DynamicCache` normalization: empty `DynamicCache` passed as `past_key_values` normalized to `None`
2. Same guard in `prepare_inputs_for_generation()`
3. Rotary embedding API: `rotary_emb(x, seq_len=n)` → `rotary_emb(x, position_ids)`

**`quant/new_pack.py`** — added `dequantize_cache_pertoken()` for unpacking per-token quantized keys back to fp16 during decode.

---

## Running Experiments

### Single experiment

```bash
conda activate kivi
cd KIVI

# FP16 baseline
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_fp16.py

# KIVI default (per-channel keys)
CUDA_VISIBLE_DEVICES=0 python run_gsm8k.py

# Per-token keys, group=32, with residual buffer
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_pertoken.py

# Per-token keys, group=32, no residual buffer
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_pertoken_noresidual.py

# Per-token keys, flat (group=128), with residual
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_pertoken_flat.py

# Per-token keys, flat (group=128), no residual
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_pertoken_flat_noresidual.py
```

Replace `run_gsm8k_*.py` with `run_gpqa_*.py` for GPQA Diamond Generative evaluations.

Results are saved to `logs/<experiment>_results.json`.

### Parallel runs across multiple GPUs (recommended)

Use tmux to keep experiments alive independent of your SSH/VSCode session:

```bash
# GPU 0: FP16 baseline
tmux new-session -d -s gpu0
tmux send-keys -t gpu0 "cd KIVI && CUDA_VISIBLE_DEVICES=0 python run_gsm8k_fp16.py 2>&1 | tee logs/gsm8k_fp16.log && CUDA_VISIBLE_DEVICES=0 python run_gpqa_fp16.py 2>&1 | tee logs/gpqa_fp16.log" Enter

# GPU 1: PerToken flat, residual=32
tmux new-session -d -s gpu1
tmux send-keys -t gpu1 "cd KIVI && CUDA_VISIBLE_DEVICES=1 python run_gsm8k_pertoken_flat.py 2>&1 | tee logs/gsm8k_pertoken_flat.log && CUDA_VISIBLE_DEVICES=1 python run_gpqa_pertoken_flat.py 2>&1 | tee logs/gpqa_pertoken_flat.log" Enter

# GPU 2: PerToken flat, residual=0
tmux new-session -d -s gpu2
tmux send-keys -t gpu2 "cd KIVI && CUDA_VISIBLE_DEVICES=2 python run_gsm8k_pertoken_flat_noresidual.py 2>&1 | tee logs/gsm8k_pertoken_flat_noresidual.log && CUDA_VISIBLE_DEVICES=2 python run_gpqa_pertoken_flat_noresidual.py 2>&1 | tee logs/gpqa_pertoken_flat_noresidual.log" Enter

# GPU 3: reruns of KIVI default + PerToken variants (6 experiments chained)
tmux new-session -d -s gpu3
tmux send-keys -t gpu3 "cd KIVI && \
  CUDA_VISIBLE_DEVICES=3 python run_gsm8k.py 2>&1 | tee logs/gsm8k_kivi.log && \
  CUDA_VISIBLE_DEVICES=3 python run_gpqa_kivi.py 2>&1 | tee logs/gpqa_kivi.log && \
  CUDA_VISIBLE_DEVICES=3 python run_gsm8k_pertoken.py 2>&1 | tee logs/gsm8k_pertoken.log && \
  CUDA_VISIBLE_DEVICES=3 python run_gpqa_pertoken.py 2>&1 | tee logs/gpqa_pertoken.log && \
  CUDA_VISIBLE_DEVICES=3 python run_gsm8k_pertoken_noresidual.py 2>&1 | tee logs/gsm8k_pertoken_noresidual.log && \
  CUDA_VISIBLE_DEVICES=3 python run_gpqa_pertoken_noresidual.py 2>&1 | tee logs/gpqa_pertoken_noresidual.log" Enter

# Watch for completion and auto-generate report
tmux new-session -d -s report
tmux send-keys -t report "cd KIVI && bash wait_and_report.sh" Enter
```

To check progress:
```bash
tmux attach -t gpu0     # Ctrl+B, D to detach without stopping
tail -f logs/gsm8k_fp16.log
```

### Auto-generate report when all done

```bash
python generate_report.py
# also auto-runs via wait_and_report.sh when all 12 JSON files appear
```

---

## Configuration Reference

All config is set inline at the top of each run script. Key parameters:

| Parameter | Description | Values used |
|---|---|---|
| `k_bits` | Key cache bit-width | 2 |
| `v_bits` | Value cache bit-width | 2 |
| `group_size` | Elements per quantization group (along head\_dim) | 32 or 128 |
| `residual_length` | FP16 buffer for most-recent tokens | 0 or 32 |
| `use_flash` | FlashAttention 2 (requires Ampere GPU) | False |
| `batch_size` | lm-eval batch size | 16 (32GB V100) |

To add a new variant, simply copy any run script and change these values.

---

## Reproducibility Notes

- **Batch size matters**: lm-eval generative tasks may produce slightly different scores at different batch sizes due to padding. Always compare experiments at the same batch size.
- **Greedy decoding**: all runs use greedy (temperature=0) via lm-eval defaults.
- **Seeds**: lm-eval sets `random_seed=0, numpy_seed=1234, torch_seed=1234` by default.
- **GSM8K**: 5-shot, 1319 test questions, `exact_match` with strict-match and flexible-extract filters.
- **GPQA Diamond**: `gpqa_diamond_generative_n_shot` task, 198 questions.
