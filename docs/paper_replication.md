# Replicating KIVI Paper Results

Step-by-step guide to reproduce the exact numbers from [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) (ICML 2024, Table 3).

## Verified Results

Using this setup, we reproduced the following for **Llama-2-7B** (base):

| Task | Metric | Our Result | Paper (Table 3) |
|------|--------|-----------|-----------------|
| CoQA | Exact Match | **63.05** | 63.05 |
| TruthfulQA | BLEU (bleu_max) | **33.95** | 33.95 |
| GSM8K | Exact Match | pending | 12.74 |

FP16 baselines also match:

| Task | Metric | Our Result | Paper (Table 3) |
|------|--------|-----------|-----------------|
| CoQA | Exact Match | **63.88** | 63.88 |
| TruthfulQA | BLEU | **30.83** | 30.76 |
| GSM8K | Exact Match | **13.42** | 13.50 |

---

## Important Notes

- The paper uses **Llama-2-7b-hf** (base model), not the chat variant.
- TruthfulQA is evaluated with `truthfulqa_gen` (BLEU score), **not** `truthfulqa_mc1`/`mc2`.
- CoQA metric is **Exact Match**, not F1.
- The paper's model code requires **transformers 4.36.2**. The updated model code in `models/` requires transformers 4.43+. We provide the original model files in `models_paper/` for paper replication.
- The paper uses a specific lm-eval commit, not a released version.

---

## Environment Setup

### 1. Create conda environment

```bash
conda create -n kivi_paper python=3.10 -y
conda activate kivi_paper
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install \
    transformers==4.36.2 \
    tokenizers==0.15.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    huggingface-hub==0.20.2 \
    datasets==2.16.1 \
    sentencepiece==0.1.99 \
    peft==0.7.1 \
    protobuf==4.25.1 \
    evaluate==0.4.1 \
    rouge-score==0.1.2 \
    sacrebleu==1.5.0 \
    nltk==3.8.1 \
    numpy==1.26.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    einops==0.7.0 \
    pybind11==2.11.1 \
    numexpr==2.8.8 \
    "setuptools<75"
```

### 4. Install lm-eval (paper's exact commit)

```bash
pip install -e "git+https://github.com/EleutherAI/lm-evaluation-harness.git@c9bbec6e7de418b9082379da82797522eb173054#egg=lm_eval"
```

### 5. Install KIVI package and CUDA kernels

```bash
cd KIVI
pip install -e . --no-deps
cd quant && pip install -e . --no-build-isolation
cd ..
```

### 6. (Optional) Install flash-attn for Ampere+ GPUs

```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

> V100 users: skip this step. Flash-attn requires Ampere (sm_80+).

### 7. Authenticate with HuggingFace

```bash
huggingface-cli login
```

Request access to `meta-llama/Llama-2-7b-hf` at https://huggingface.co/meta-llama/Llama-2-7b-hf

---

## Running Experiments

### Paper replication (uses `models_paper/` with transformers 4.36.2)

The script `run_lm_eval_harness.py` uses the original KIVI model files from the paper's lmeval branch.

**FP16 baseline:**

```bash
# Uses --k_bits 16 --v_bits 16 for FP16 mode
CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 16 --v_bits 16 \
    --tasks gsm8k
```

**KIVI 2-bit (paper's default: group_size=32, residual_length=128):**

```bash
CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 2 --v_bits 2 \
    --group_size 32 --residual_length 128 \
    --tasks gsm8k
```

**Run all three tasks at once:**

```bash
CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 2 --v_bits 2 \
    --group_size 32 --residual_length 128 \
    --tasks gsm8k,coqa,truthfulqa_gen
```

**Run multiple tasks in parallel on different GPUs:**

```bash
# GPU 0: GSM8K
CUDA_VISIBLE_DEVICES=0 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 2 --v_bits 2 --group_size 32 --residual_length 128 \
    --tasks gsm8k &

# GPU 1: CoQA
CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 2 --v_bits 2 --group_size 32 --residual_length 128 \
    --tasks coqa &

# GPU 2: TruthfulQA
CUDA_VISIBLE_DEVICES=2 python run_lm_eval_harness.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --k_bits 2 --v_bits 2 --group_size 32 --residual_length 128 \
    --tasks truthfulqa_gen &

wait
```

Results are saved to `logs/` as JSON files.

### FP16 baselines (alternative: simpler script)

For FP16 baselines you can also use `run_eval_paper.py` which doesn't need the KIVI model files:

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval_paper.py \
    --model fp16 --task gsm8k \
    --model_path meta-llama/Llama-2-7b-hf
```

---

## Paper Configuration Reference

From Section 4.1 of the paper:

| Parameter | Value |
|-----------|-------|
| Group size (G) | 32 |
| Residual length (R) | 128 |
| K bits | 2 |
| V bits | 2 |
| Key quantization | per-channel |
| Value quantization | per-token |

Tasks and metrics (from Section 4.1):
- **CoQA**: Exact match accuracy, 0-shot
- **TruthfulQA**: BLEU score (`truthfulqa_gen`), 0-shot
- **GSM8K**: Exact match accuracy, 5-shot (set by lm-eval default)

---

## Troubleshooting

### Dataset cache errors (`TypeError: must be called with a dataclass type`)

If you previously ran with a newer `datasets` version, the cached data is incompatible. Clear the cache:

```bash
rm -rf ~/.cache/huggingface/datasets/truthful_qa*
rm -rf ~/.cache/huggingface/datasets/EleutherAI___coqa
rm -rf ~/.cache/huggingface/datasets/gsm8k
```

### `kivi_gemv` import error (`undefined symbol`)

The CUDA extension must be compiled with the same PyTorch version you're running. Rebuild:

```bash
cd quant
rm -f kivi_gemv*.so
pip install -e . --no-build-isolation
```

### OOM on GSM8K

Reduce batch size (the paper's script defaults to batch_size=1, which is safe):

```bash
python run_lm_eval_harness.py ... --batch_size 1
```

---

## Two Environments

This repo supports two environments:

| | Paper Replication (`kivi_paper`) | Modern (`kivi`) |
|---|---|---|
| Purpose | Reproduce Table 3 exactly | New experiments with updated code |
| torch | 2.1.2 | 2.4.1 |
| transformers | 4.36.2 | 4.43.1 |
| lm-eval | commit c9bbec6e | 0.4.2 |
| Model files | `models_paper/` | `models/` |
| Eval script | `run_lm_eval_harness.py` | `run_eval.py` |
| Flash-attn | Optional (2.5.6) | Optional (2.8.3) |

The modern `kivi` environment produces very similar results but may differ by ~0.1-2pp due to library version differences.
