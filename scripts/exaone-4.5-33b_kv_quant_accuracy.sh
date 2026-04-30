#!/bin/bash
# =============================================================================
# EXAONE-4.5-33B KV-quant accuracy harness
# =============================================================================
#
# Same 4 KV-cache settings + same 3 tasks as the Qwen3 harnesses, but on
# LGAI-EXAONE/EXAONE-4.5-33B. Uses MG=32k single-pass.
#
# EXAONE-4.5 differs from Qwen3 in two ways that warrant separate scripting:
#
#   1. Architecture is Exaone4_5_ForConditionalGeneration — needs the lkm2835
#      vllm fork that registers it. The nuxlear transformers fork ships
#      modeling but no image processor; vllm imports
#      Exaone4_5_ImageProcessor at module load and crashes without a stub.
#      A one-time stub install is required (see "Step 4" below or SETUP.md).
#
#   2. SmoothKV calib variant differs. EXAONE-4.5's attention does NOT have
#      q_norm/k_norm gamma vectors (unlike Qwen3 / Qwen3-MoE / Olmo2), so the
#      head-uniform constraint doesn't apply. Calib is `_pair` (per-head s_K,
#      folded per-head into the K-projection weights) instead of `_huk_halfpair`.
#      For Qwen3 use _huk_halfpair; for EXAONE-4.5 use _a1b1_pair.
#
# =============================================================================
# Environment
# =============================================================================
# Same env as the Qwen3 harnesses (the v2 env at /opt/vllm_exaone_v2_env was
# built for EXAONE-4.5 from day one). Pinned versions:
#   vllm           : 0.20.1.dev0+g101584af0  (lkm2835/vllm@add-exaone4_5)
#   transformers   : 5.6.0.dev0              (nuxlear/transformers@31991e75)
#   torch          : 2.10.0+cu128
#   lm_eval        : 0.4.11
#   math_verify    : 0.9.0
#   antlr4-python3 : 4.11.0
#
# From-scratch build: see SETUP.md "Step 1" through "Step 4". The image
# processor stub at $SITE_PKG/transformers/models/exaone4_5/image_processing_exaone4_5.py
# must exist before this script runs; the script verifies it.
#
# Override env path via VLLM_ENV.
#
# =============================================================================
# Usage
# =============================================================================
#   bash scripts/exaone-4.5-33b_kv_quant_accuracy.sh
#
#   ETA on 8× A100-80GB: ~6-9 hours wall clock.
#
# =============================================================================
set -euo pipefail
cd "$(/usr/bin/dirname "$0")/.."

VLLM_ENV="${VLLM_ENV:-/opt/vllm_exaone_v2_env}"
PY="$VLLM_ENV/bin/python3"
[ -x "$PY" ] || { /usr/bin/echo "ERROR: $PY not found. See SETUP.md."; exit 1; }

# Verify the image-processor stub exists; without it vllm crashes at engine init.
SITE_PKG=$($PY -c "import transformers; print(transformers.__path__[0])")
STUB="$SITE_PKG/models/exaone4_5/image_processing_exaone4_5.py"
if [ ! -f "$STUB" ]; then
  /usr/bin/echo "ERROR: missing $STUB"
  /usr/bin/echo "Install per SETUP.md \"Step 4 — Patch the transformers EXAONE-4.5 package\""
  exit 1
fi

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

MODEL=LGAI-EXAONE/EXAONE-4.5-33B
MODEL_TAG=exaone-4.5-33b

# EXAONE has no q_norm/k_norm → use the per-head `_pair` calib variant.
CALIB_BASE=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512.pt
CALIB_PAIR=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512_puremax_a1b1_pair.pt

GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
LABELS=("ex45a_bf16" "ex45a_fp8" "ex45a_pert" "ex45a_smk")
MTAGS=("bf16" "fp8_g128" "pertoken_int4_g128"
       "smoothkv_fused_g128_perc_ns512_puremax_a1b1_pair")
METHOD_ARGS=(
  "--model bf16"
  "--model fp8 --group_size 128"
  "--model pertoken --bits 4 --group_size 128"
  "--model smoothkv_fused --calib_path $CALIB_PAIR --group_size 128 --bits 4"
)

/bin/mkdir -p logs/run_out logs/calib

# -----------------------------------------------------------------------------
# SmoothKV calibration
# -----------------------------------------------------------------------------
if [ ! -f "$CALIB_BASE" ]; then
  /usr/bin/echo "[calib] generating base $CALIB_BASE (~60 min on 2× A100, device_map=auto)"
  CUDA_VISIBLE_DEVICES=0,1 $PY run_smoothkv_calibrate.py \
    --model_path "$MODEL" \
    --num_samples 512 --seq_length 2048 \
    --alpha 0.5 --beta 0.5 \
    --samples_per_channel 10000 \
    --output "$CALIB_BASE" --device auto
fi
if [ ! -f "$CALIB_PAIR" ]; then
  /usr/bin/echo "[calib] deriving per-head pair α=β=1 variant (no _huk for EXAONE)"
  $PY scripts/make_alpha_variants.py \
    --base "$CALIB_BASE" \
    --alpha 1.0 --beta 1.0 \
    --pair_max_k --half_pair_max_k --no_head_uniform_k \
    --output "$CALIB_PAIR"
fi

# -----------------------------------------------------------------------------
# Per-variant worker
# -----------------------------------------------------------------------------
WORKER=/tmp/exaone-4.5-33b_kv_worker_$$.sh
/bin/cat > "$WORKER" <<'EOS'
#!/bin/bash
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN=$(/bin/cat ~/.cache/huggingface/token)
export NO_ENFORCE_EAGER=1
export CUDA_VISIBLE_DEVICES=$1
cd /workspace/KIVI
PY=/opt/vllm_exaone_v2_env/bin/python3

LABEL=$2; MTAG=$3; METHOD_ARGS=$4

declare -A MML
MML[gsm8k_32k]=34304
MML[minerva_math500]=34304
MML[gpqa_main_cot_n_shot_32k]=35584

for task in gsm8k_32k minerva_math500 gpqa_main_cot_n_shot_32k; do
  result=logs/${task}_exaone-4.5-33b_${MTAG}_chat_vllm_results.json
  if [ -f "$result" ]; then
    echo "[$LABEL] SKIP $task — exists"
    continue
  fi
  echo "[$LABEL] === START $task at $(/bin/date) ==="
  $PY run_eval_vllm.py $METHOD_ARGS \
      --model_path LGAI-EXAONE/EXAONE-4.5-33B \
      --task "$task" --apply_chat_template \
      --max_gen_toks 32768 --max_model_len ${MML[$task]} \
      --max_num_seqs 8 --batch_size 8 --tp 2 \
      --log_samples \
      2>&1 | /usr/bin/tee -a logs/run_out/${LABEL}_${task}.log
  echo "[$LABEL] === DONE $task at $(/bin/date) ==="
done
echo "[$LABEL] === ALL DONE at $(/bin/date) ==="
EOS
/bin/chmod +x "$WORKER"

for i in 0 1 2 3; do
  /usr/bin/tmux new-session -d -s "${LABELS[$i]}" \
    "$WORKER ${GPU_PAIRS[$i]} ${LABELS[$i]} ${MTAGS[$i]} '${METHOD_ARGS[$i]}'"
done

/usr/bin/echo "launched 4 streams (TP=2 each):"
/usr/bin/tmux ls 2>&1 | /usr/bin/grep -E "^ex45a_"
