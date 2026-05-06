#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for EXAONE-4.5-33B.
#
# EXAONE differs from Qwen3:
#   - No q_norm/k_norm gamma → SmoothKV uses _pair calib (not _huk_halfpair)
#   - Single-pass MG=32k (no adaptive 2-pass) — tested empirically to give the
#     same headline number as Qwen3 adaptive within noise
#   - Needs lkm2835/vllm@add-exaone4_5 fork + nuxlear/transformers fork +
#     image_processor stub at SITE_PKG/transformers/models/exaone4_5/.
#   - HF model card uses text_config.model_type=exaone4_5_text but fork
#     registers only "exaone4" — alias patch is in configuration_exaone4_5.py
#     (see memory note feedback_exaone45_text_config_alias.md).
#
# Usage:
#   bash scripts/run_eval_exaone.sh \
#       --variant bf16|fp8|pertoken|smkv \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot \
#       [--ns 512] [--alpha 1.0] [--beta 1.0] \
#       [--gpus 0,1]                        # TP=2 pair (33B doesn't fit on 80GB single)
#
# Examples:
#   bash scripts/run_eval_exaone.sh --variant bf16 --task gsm8k_cot --gpus 0,1
#   bash scripts/run_eval_exaone.sh --variant smkv --gpus 6,7 --task minerva_math500
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

# ---- args ----
VARIANT=""; TASK=""; GPUS="0,1"; NS=512; ALPHA=1.0; BETA=1.0
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --ns) NS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    *) /usr/bin/echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -z "$VARIANT" ] || [ -z "$TASK" ] && {
  /usr/bin/echo "Required: --variant {bf16|fp8|pertoken|smkv} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot}"
  exit 1
}

case "$TASK" in
  gsm8k_cot|minerva_math500) PROMPT_BUDGET=1536 ;;
  gpqa_main_cot_n_shot)  PROMPT_BUDGET=3072 ;;
  *) /usr/bin/echo "task must be gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot"; exit 1 ;;
esac

MODEL_PATH=LGAI-EXAONE/EXAONE-4.5-33B
MODEL_TAG=exaone-4.5-33b
cd /workspace/KIVI
# PY: python interpreter (defaults to .venv from `make setup`).
PY="${PY:-./.venv/bin/python}"

# ---- derive max-gen-tokens from model's native context length ----
# Rule: mg = 32k if model_max_len >= 32k else model_max_len/2.
MODEL_MAX_LEN=$($PY -c "from transformers import AutoConfig; \
print(AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True).max_position_embeddings)")
if [ "$MODEL_MAX_LEN" -ge 32768 ]; then MG=32768; else MG=$((MODEL_MAX_LEN / 2)); fi
MML=$((MG + PROMPT_BUDGET))
/usr/bin/echo "[mg] model_max_len=$MODEL_MAX_LEN  →  mg=$MG, mml=$MML"

# ---- env sanity (image_processor stub + config alias patch) ----
SITE_PKG=$($PY -c "import transformers; print(transformers.__path__[0])")
STUB="$SITE_PKG/models/exaone4_5/image_processing_exaone4_5.py"
if [ ! -f "$STUB" ]; then
  /usr/bin/echo "ERROR: missing image_processor stub at $STUB"
  /usr/bin/echo "Run setup steps in scripts/exaone-4.5-33b_kv_quant_accuracy.sh"
  exit 1
fi
$PY -c "
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config
src = open('$SITE_PKG/models/exaone4_5/configuration_exaone4_5.py').read()
assert 'exaone4_5_text' in src, 'missing exaone4_5_text→exaone4 alias patch (see memory note)'
"

# ---- variant config ----
calib_path=""
case "$VARIANT" in
  bf16)     METHOD_ARGS="--kv_quant_method bf16";              VARIANT_TAG="bf16" ;;
  fp8)      METHOD_ARGS="--kv_quant_method fp8 --group_size 128";    VARIANT_TAG="fp8_g128" ;;
  pertoken) METHOD_ARGS="--kv_quant_method pertoken --bits 4 --group_size 128"; VARIANT_TAG="pertoken_int4_g128" ;;
  smkv)
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}.pt"
    VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_a${AS}b${BS}_pair.pt"
    [ "$AS" = "1" ] && [ "$BS" = "1" ] && VAR="logs/calib/smoothkv_${MODEL_TAG}_ns${NS}_puremax_a1b1_pair.pt"
    VARIANT_TAG="smoothkv_fused_g128_perc_ns${NS}_puremax_a${AS}b${BS}_pair"
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS)"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 0.5 --beta 0.5 --samples_per_channel 10000 \
        --output "$BASE" --device auto
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving variant $VAR  (α=$ALPHA β=$BETA, pair, no HUK — EXAONE has no q_norm)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA \
        --pair_max_k --no_head_uniform_k
    fi
    calib_path="$VAR"
    METHOD_ARGS="--kv_quant_method smoothkv_fused --calib_path $calib_path --bits 4 --group_size 128"
    ;;
  *) /usr/bin/echo "variant must be bf16|fp8|pertoken|smkv"; exit 1 ;;
esac

LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_${TASK}.log
RESULT=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
/bin/mkdir -p logs/run_out logs/calib

# ---- single-pass run ----
# Set FORCE=1 to redo a cell whose outputs already exist. Verify-graph
# stamp is only checked on a fresh run — SKIP path trusts existing data.
FORCE="${FORCE:-0}"
if [ "$FORCE" != "1" ] && [ -f "$RESULT" ]; then
  /usr/bin/echo "[run] SKIP — results.json exists (set FORCE=1 to override)"
else
  /usr/bin/echo "[run] $TASK  on $MODEL_PATH  (TP=2, MG=$MG, max_num_seqs=8)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY run_eval_vllm.py \
      $METHOD_ARGS --model "$MODEL_PATH" \
      --task "$TASK" --apply_chat_template \
      --max_gen_toks $MG --max_model_len $MML \
      --max_num_seqs 8 --batch_size 8 --tp 2 \
      --log_samples 2>&1 | /usr/bin/tee "$LOG"
  if ! /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$LOG" 2>/dev/null; then
    /usr/bin/echo "ERROR: no explicit verify-graph PASS stamp. Aborting." >&2
    exit 2
  fi
  /usr/bin/echo "[run] verify-graph: PASS"
fi
/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK graph-verified"
/usr/bin/echo "   Output: $RESULT"
/usr/bin/echo "=================================================================="
