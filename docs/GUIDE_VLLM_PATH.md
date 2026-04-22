# Guide: vLLM fake-quant path

Hand this to another Claude (or yourself on a fresh pod) to enable the vLLM
inference path for KIVI KV-cache methods. End-to-end long-generation throughput
is ~10× the HFLM path (measured on pertoken int4 + Mistral-Instruct).

## What this is

`vllm_custom/` monkey-patches `vllm.model_executor.models.llama.LlamaAttention.forward`
so K and V go through a quant→dequant round-trip after RoPE, before PagedAttention.
The KV cache itself stays FP16 — only the values written to it are quantization-
lossy, simulating the quant error of a real low-bit cache.

Supported methods (installers in `vllm_custom/patches.py`):

| Method                | Installer                                  |
|-----------------------|--------------------------------------------|
| `fp16`                | no patch                                   |
| `fp8`                 | `install_fp8(group_size=128)`              |
| `pertoken` (int4)     | `install_pertoken_int4(group_size=128)`    |
| `smoothkv`            | `install_smoothkv(calib_path, ...)`        |
| `kivi` (approx KIVI-2)| `install_kivi2(group_size=32)`             |

**Caveats** (also in module docstrings):
- Only one method can be active per process — the patch is global.
- `install_kivi2` is NOT a faithful KIVI-2 port. vLLM's KV cache is fp16 with
  no retroactive-rewrite hook, so the R=128 residual buffer isn't simulated.
  Use `run_eval.py` (HFLM path) for KIVI-2 numbers.
- SmoothKV/fp8/pertoken with `residual=0` ARE faithful — they apply the same
  quant→dequant transform per step that the HFLM path does.

## Prerequisites

```bash
git clone https://github.com/sunghyuckhong/KIVI.git
cd KIVI
git checkout pertoken-experiments    # need the vLLM scaffold commit
export HF_TOKEN=<your_hf_token>
```

## Environment — build `/opt/vllm_env`

Python 3.10 + vLLM 0.6.6 + transformers 4.45.2 (exact pin — newer transformers
break vLLM 0.6.6's tokenizer backend).

```bash
python3.10 -m venv /opt/vllm_env
/opt/vllm_env/bin/pip install --upgrade pip setuptools wheel
/opt/vllm_env/bin/pip install "vllm==0.6.6" "transformers==4.45.2" "lm_eval==0.4.2" \
    accelerate datasets sentencepiece "numpy<2"

# KIVI's triton-based KV quant extension (same one the HFLM path uses)
cd quant && /opt/vllm_env/bin/pip install -e . && cd ..
```

Sanity check:
```bash
/opt/vllm_env/bin/python -c "
import vllm, transformers, lm_eval, torch
print(f'vllm={vllm.__version__}  transformers={transformers.__version__}  torch={torch.__version__}')
import kivi_gemv; print('kivi_gemv OK')
import vllm_custom.patches as p; print('vllm_custom OK')
"
```

Expected:
```
vllm=0.6.6  transformers=4.45.2  torch=2.5.1
kivi_gemv OK
vllm_custom OK
```

## Smoke tests (~2 min each on a single GPU)

Before launching a full matrix, confirm all four methods boot and produce
non-garbage exact_match on a 10-sample gsm8k slice:

```bash
MODEL=mistralai/Mistral-7B-Instruct-v0.2
CALIB=logs/calib/smoothkv_mistral-7b-instruct-v0.2_a0.75.pt   # or any existing .pt

for METHOD_ARGS in \
    "--model fp16" \
    "--model fp8 --group_size 128" \
    "--model pertoken --group_size 128" \
    "--model smoothkv --calib_path $CALIB"; do
  CUDA_VISIBLE_DEVICES=0 /opt/vllm_env/bin/python run_eval_vllm.py \
    --model_path $MODEL --task gsm8k_32k \
    --max_gen_toks 256 --limit 10 $METHOD_ARGS
done
```

Each run prints an lm-eval table and writes `logs/<task>_<model>_<method>_vllm_results.json`.

## Full evaluation

`run_eval_vllm.py` mirrors `run_eval.py`'s CLI — same `--model` choices, same
`--task` / `--max_gen_toks` / `--calib_path` semantics. Output filenames are
suffixed with `_vllm` so HFLM and vLLM results co-exist under `logs/`.

Example — SmoothKV α=0.75 pair on Mistral-Instruct, full 5-task sweep, 16k gen:
```bash
MODEL=mistralai/Mistral-7B-Instruct-v0.2
CALIB=logs/calib/smoothkv_mistral-7b-instruct-v0.2_a0.75_pair.pt

for TASK in truthfulqa_gen coqa gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k; do
  EXTRA=""
  case "$TASK" in
    gsm8k_32k|gpqa_diamond_cot_n_shot_32k|math500_32k) EXTRA="--max_gen_toks 16384" ;;
  esac
  CUDA_VISIBLE_DEVICES=0 /opt/vllm_env/bin/python run_eval_vllm.py \
    --model smoothkv --calib_path "$CALIB" \
    --model_path "$MODEL" --task "$TASK" --batch_size 16 $EXTRA
done
```

vLLM's PagedAttention handles long-gen samples in-batch, so `--batch_size 16`
is fine (HFLM path needed much smaller batches to avoid OOM on 32k-gen tasks).

## Getting calibrations onto the new pod

SmoothKV calibs are large (~5 GB for the base `.pt`, ~40 MB per variant).
Options:
- **Re-calibrate locally** — `/opt/vllm_env/bin/python run_smoothkv_calibrate.py
  --model_path <model> --num_samples 128 --seq_length 2048
  --samples_per_channel 300000 --output logs/calib/smoothkv_<model>_perc.pt`
  then run `scripts/make_alpha_variants.py` / `scripts/make_percentile_variants.py`.
  Takes ~30 min on 1 GPU. Always use `samples_per_channel ≥ num_samples × seq_length`
  to retain every observation (no reservoir eviction).
- **Copy variants only** — the small `_a0.75_pair.pt` / `_pairK{90,95,99,99p9}.pt`
  files are enough to evaluate; you don't need the full `_perc.pt` base.

## How the patch works (if you need to debug)

1. `install_<method>(...)` saves the original `LlamaAttention.forward` and
   replaces it with a new forward that applies quant→dequant to K (and V, and
   sometimes Q-matching scale).
2. For SmoothKV, `install_smoothkv` additionally hooks `LlamaAttention.__init__`
   to stash `self._kivi_layer_idx` — vLLM 0.6.6 passes `prefix="model.layers.N.self_attn"`
   to the constructor but doesn't store it, so we parse it once at build time.
3. The patched forward calls `self.attn(q, k, v, kv_cache, attn_metadata)` with
   the dequantized tensors, so PagedAttention and CUDA graphs still work.

If a new vLLM version changes `LlamaAttention.forward`'s signature (positions,
hidden_states, kv_cache, attn_metadata), update the five installers in
`vllm_custom/patches.py` to match.

## Files touched by this path

- `vllm_custom/__init__.py`
- `vllm_custom/fake_quant_utils.py` — shape-agnostic quant→dequant helpers
- `vllm_custom/patches.py` — `install_*` monkey-patches + `restore()`
- `run_eval_vllm.py` — CLI wrapper, mirrors `run_eval.py`, suffix `_vllm`

Nothing in the HFLM path (`run_eval.py`, `models/`, `quant/`) is modified.
