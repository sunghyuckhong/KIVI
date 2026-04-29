# Setup guide — vLLM 0.20+ environment for KIVI sweeps

This guide documents the exact environment used for the KIVI per-token / SmoothKV sweeps on Qwen3-8B, Qwen3-30B-A3B, and EXAONE-4.5-33B. Reproducing on a separate pod gets you cudagraph-compatible quant runs (including software FP8 emulation on A100/sm_80) plus the patched lm-evaluation-harness flow with `math_verify` for boxed-aware math500 scoring.

## Hardware tested

- 8× NVIDIA A100-SXM4-80GB
- NVIDIA driver 570.133.20 (CUDA 12.8)
- Compute capability **sm_80** (A100). On sm_89+ (Hopper H100/H200) the FP8 hardware path activates automatically — no rebuild needed.

## Python + system deps

```bash
# Python 3.10 (tested with 3.10.12). Other 3.10.x should work; 3.11+ untested.
python3 --version  # 3.10.12

# git is required for the editable forks
which git
```

## Step 1 — Create the env

```bash
python3 -m venv /opt/vllm_v0.20_env
source /opt/vllm_v0.20_env/bin/activate
pip install --upgrade pip
```

## Step 2 — Install the vLLM + transformers forks

EXAONE-4.5 needs both forks pinned together to the `add-exaone4_5` branches; these also work for Qwen3 / Qwen3-MoE since they're vanilla vLLM 0.20+ underneath.

```bash
# vLLM fork — has Exaone4_5_ForConditionalGeneration model registered.
# Wheel build takes ~15-30 min on a fresh box.
pip install "git+https://github.com/lkm2835/vllm.git@add-exaone4_5"

# Transformers fork — provides Exaone4_5_Config, modeling, processing.
# Pin the commit so future fork updates don't break us.
pip install --no-deps \
    "git+https://github.com/nuxlear/transformers.git@31991e758f53bebaed91a066ed9ceb476a3c7777"
```

After install, verify:
```bash
python3 -c "import vllm; print(vllm.__version__)"
# expected: 0.20.1.dev0+g101584af0... (or newer pin from the fork)
python3 -c "import transformers; print(transformers.__version__)"
# expected: 5.6.0.dev0
```

## Step 3 — Install lm-evaluation-harness + math_verify

```bash
pip install lm_eval==0.4.11
# math_verify enables boxed-aware scoring on math500 (the second metric on
# minerva_math500). antlr4-python3-runtime must be 4.11 specifically — newer
# versions break sympy's latex parser.
pip install math_verify "antlr4-python3-runtime==4.11"
```

## Step 4 — Patch the transformers EXAONE-4.5 package

The nuxlear fork's `transformers.models.exaone4_5` package ships the LM modeling code but **omits the image processor** that `vllm.model_executor.models.exaone4_5` expects to import. Without this stub, vLLM crashes at engine init when it tries to instantiate the multimodal pipeline.

For text-only inference (the entire KIVI sweep), drop in this minimal stub:

```bash
SITE_PKG=$(python3 -c "import transformers; print(transformers.__path__[0])")
cat > "${SITE_PKG}/models/exaone4_5/image_processing_exaone4_5.py" << 'EOF'
"""Stub for Exaone4_5_ImageProcessor (text-only inference).

The transformers fork ships configuration/modeling/processing but no image
processor. The vllm fork's exaone4_5.py imports Exaone4_5_ImageProcessor at
module load and probes .size and .merge_size during multimodal-budget init,
so we provide a placeholder. Calling .preprocess raises — image inputs are
not supported by this stub."""
from transformers.image_processing_utils import BaseImageProcessor

__all__ = ["Exaone4_5_ImageProcessor"]


class Exaone4_5_ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, min_pixels=56*56, max_pixels=14*14*4*1280,
                 patch_size=14, temporal_patch_size=2, merge_size=2, **kw):
        super().__init__(**{k: v for k, v in kw.items() if k in ("image_mean", "image_std")})
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def preprocess(self, *a, **kw):
        raise NotImplementedError(
            "Exaone4_5_ImageProcessor stub — text-only inference only.")
EOF
```

You also need to pass `limit_mm_per_prompt={"image": 0, "video": 0}` to `LLM(...)` for EXAONE-4.5 to skip the video-processor probe (`run_eval_vllm.py` already does this when `--model_path` contains `EXAONE-4.5`).

## Step 5 — KIVI repo + supporting deps

```bash
git clone https://github.com/<your-fork>/KIVI.git
cd KIVI
git checkout pertoken-experiments

# Build the KIVI quant CUDA package
pip install -e . --no-build-isolation

# Other deps lm-eval imports
pip install evaluate datasets pandas pytz python-dateutil pytablewriter \
    sacrebleu rouge_score nltk word2number absl-py jsonlines msgpack ray
```

## Step 6 — HuggingFace auth

```bash
huggingface-cli login   # paste HF token
```

Required model access:
- `LGAI-EXAONE/EXAONE-4.5-33B`
- `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B`
- `Idavidrein/gpqa` (for the gpqa_main task dataset)

## Step 7 — Smoke test

```bash
cd KIVI
PATH=/usr/bin:/bin:$PATH \
HF_TOKEN=$(cat ~/.cache/huggingface/token) \
CUDA_VISIBLE_DEVICES=0 \
python3 run_eval_vllm.py \
  --model bf16 \
  --model_path Qwen/Qwen3-8B \
  --task gsm8k_32k \
  --apply_chat_template \
  --max_gen_toks 4096 --max_model_len 6144 \
  --max_num_seqs 64 --batch_size 64 --tp 1 \
  --limit 20 --log_samples
```

Expected: ~20s wall-time on A100, output `logs/gsm8k_32k_qwen3-8b_bf16_chat_vllm_results.json` with strict-match score ≥ 0.85.

## Replication notes

- **Adaptive MG pattern**: chat-templated thinking-mode (`--apply_chat_template`) generates ~2k token responses; size `max_num_seqs` to avoid preempting at long MG. Suggested: `MG=4k` first pass with `MAX_NS=64` (no preemption), then `retry_truncated.py --orig_mg 4096 --retry_mg 32768` for the truncated tail.
- **Cudagraph defaults**: `run_eval_vllm.py` auto-enables cudagraphs for vllm 0.20+ on every method (including FP8 on sm_80 via the software E4M3 emulation in `quant/fp8_quant.py`). On older vllm (<0.20), the script falls back to eager for quant methods. Override via `FORCE_ENFORCE_EAGER=1` or `NO_ENFORCE_EAGER=1`.
- **`/no_think` / `<think></think>` pre-fill is forbidden** in any task `doc_to_text` — Qwen3 and EXAONE-4.5 honor those directives and silently disable thinking mode. Control thinking via `--apply_chat_template` only; the tokenizer's chat template defaults to `enable_thinking=True`.
- **math500 scoring**: use upstream `--task minerva_math500` (lm-eval ≥ 0.4.11) to get both `exact_match` (Minerva regex) and `math_verify` (boxed-aware) metrics inline. Or rescore saved `_samples.json` files offline with `math_verify.parse + verify`.
- **SmoothKV calibration**: `run_smoothkv_calibrate.py --device auto` auto-shards 33B+ models across visible GPUs via `device_map="auto"`. Multimodal wrappers (e.g. EXAONE-4.5) load via `AutoModel` fallback when `AutoModelForCausalLM` doesn't recognize the config.

## Pinned package list (informational)

The exact `pip freeze` from the working v2 env is in `requirements_v0.20.txt`. It includes 220+ transitive deps with cuda 12.8 wheels — use it as a last resort if a fresh install drifts; for fresh installs, the steps above are sufficient.
