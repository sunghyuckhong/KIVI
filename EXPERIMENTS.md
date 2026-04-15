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

run_eval.py                     # Unified evaluation script (replaces 12 individual scripts)
                                #   --model  fp16 | kivi | pertoken
                                #   --task   gsm8k | gpqa
                                #   --group_size  32 (default) | 128
                                #   --residual    32 (default) | 0
                                #   --k_bits / --v_bits  (default 2)
                                #   --batch_size         (default 16)

run_all.sh                      # Launches all 12 experiments across 4 GPUs via tmux
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

`run_eval.py` is the unified entry point for all 12 experiment configurations:

```bash
conda activate kivi
cd KIVI

# FP16 baseline
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model fp16     --task gsm8k
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model fp16     --task gpqa

# KIVI default (per-channel keys)
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model kivi     --task gsm8k
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model kivi     --task gpqa

# Per-token keys, group=32, residual=32
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gsm8k --group_size 32  --residual 32
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gpqa  --group_size 32  --residual 32

# Per-token keys, group=32, no residual
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gsm8k --group_size 32  --residual 0
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gpqa  --group_size 32  --residual 0

# Per-token keys, flat (group=128), residual=32
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gsm8k --group_size 128 --residual 32
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gpqa  --group_size 128 --residual 32

# Per-token keys, flat (group=128), no residual
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gsm8k --group_size 128 --residual 0
CUDA_VISIBLE_DEVICES=0 python run_eval.py --model pertoken --task gpqa  --group_size 128 --residual 0
```

Results are saved to `logs/<experiment>_results.json`.

### All experiments in parallel (recommended)

`run_all.sh` distributes all 12 experiments across 4 GPUs using tmux, so they survive SSH/VSCode disconnects:

```bash
conda activate kivi
cd KIVI
bash run_all.sh
```

GPU assignment inside `run_all.sh`:

| Session | GPU | Experiments |
|---------|-----|-------------|
| gpu0 | 0 | FP16 baseline (GSM8K + GPQA) |
| gpu1 | 1 | PerToken flat/group=128, residual=32 and 0 (GSM8K + GPQA each) |
| gpu2 | 2 | PerToken group=32, residual=0 (GSM8K + GPQA) |
| gpu3 | 3 | KIVI default + PerToken group=32, residual=32 (GSM8K + GPQA each) |

Check progress:
```bash
tmux attach -t gpu0          # Ctrl+B, D to detach without stopping
tail -f logs/gsm8k_fp16.log
```

### Generate report when all done

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

To add a new variant, pass the desired values to `run_eval.py`.

---

## Reproducibility Notes

- **Batch size matters**: lm-eval generative tasks may produce slightly different scores at different batch sizes due to padding. Always compare experiments at the same batch size.
- **Greedy decoding**: all runs use greedy (temperature=0) via lm-eval defaults.
- **Seeds**: lm-eval sets `random_seed=0, numpy_seed=1234, torch_seed=1234` by default.
- **GSM8K**: 5-shot, 1319 test questions, `exact_match` with strict-match and flexible-extract filters.
- **GPQA Diamond**: `gpqa_diamond_cot_n_shot` task, 198 questions.
