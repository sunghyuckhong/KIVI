#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for EXAONE-4.5-33B using the
# isolated .venv-exaone (transformers + vllm forks with KV-cache fake-quant
# code rebased onto lkm2835/vllm@add-exaone4_5).
#
# EXAONE-4.5-33B can't share .venv with the other model families: the
# transformers versions vllm pins don't recognize model_type=exaone4_5,
# and upstream vllm has no EXAONE-4.5 model class. Build the isolated env
# once with `make setup-exaone-4.5`.
#
# EXAONE-4.5 has q_norm/k_norm RMSNorm layers (like Qwen3), so smkv_fused
# uses the head-uniform + half-pair calib (`_huk_halfpair`).
#
# Usage:
#   bash scripts/run_eval_exaone.sh \
#       --variant bf16|fp8|pertoken|smkv_fused|smkv_per_channel \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot \
#       [--ns 512] [--alpha 1.0] [--beta 1.0] \
#       [--gpus 0,1]                        # TP=2 pair (33B doesn't fit on 80GB single)
#
# Examples:
#   bash scripts/run_eval_exaone.sh --variant bf16 --task gsm8k_cot --gpus 0,1
#   bash scripts/run_eval_exaone.sh --variant smkv_fused --gpus 6,7 --task minerva_math500
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

VARIANT=""; TASK=""; GPUS="0,1"; NS=512; ALPHA=1.0; BETA=1.0; GROUP_SIZE=128
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --ns) NS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --group_size) GROUP_SIZE="$2"; shift 2 ;;
    *) /usr/bin/echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -z "$VARIANT" ] || [ -z "$TASK" ] && {
  /usr/bin/echo "Required: --variant {bf16|fp8|pertoken|smkv_fused|smkv_per_channel} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot}"
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
# PY: defaults to the isolated .venv-exaone built by `make setup-exaone-4.5`.
PY="${PY:-./.venv-exaone/bin/python}"

if [ ! -x "$PY" ]; then
  /usr/bin/echo "ERROR: $PY missing — run 'make setup-exaone-4.5' first" >&2
  exit 1
fi

# ---- derive max-gen-tokens from model's native context length ----
# Rule: mg = 32k if model_max_len >= 32k else model_max_len/2.
MODEL_MAX_LEN=$($PY -c "from transformers import AutoConfig; \
print(AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True).max_position_embeddings)")
if [ "$MODEL_MAX_LEN" -ge 32768 ]; then MG=32768; else MG=$((MODEL_MAX_LEN / 2)); fi
MML=$((MG + PROMPT_BUDGET))
/usr/bin/echo "[mg] model_max_len=$MODEL_MAX_LEN  →  mg=$MG, mml=$MML"

# ---- variant config ----
calib_path=""
case "$VARIANT" in
  bf16)     METHOD_ARGS="--kv_quant_method bf16";              VARIANT_TAG="bf16" ;;
  fp8)      METHOD_ARGS="--kv_quant_method fp8 --group_size $GROUP_SIZE";    VARIANT_TAG="fp8_g${GROUP_SIZE}" ;;
  pertoken) METHOD_ARGS="--kv_quant_method pertoken --bits 4 --group_size $GROUP_SIZE"; VARIANT_TAG="pertoken_int4_g${GROUP_SIZE}" ;;
  smkv_fused)
    # EXAONE-4.5 has q_norm/k_norm → use _huk_halfpair (head-uniform + half-pair) like Qwen3
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat.pt"
    VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat_a${AS}_b${BS}_huk_halfpair.pt"
    VARIANT_TAG="smoothkv_fused_g${GROUP_SIZE}_perc_ns${NS}_chat_a${AS}_b${BS}_huk_halfpair"
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS, chat-calib)"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 1.0 --beta 1.0 --apply_chat_template \
        --output "$BASE" --device auto
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving variant $VAR  (α=$ALPHA β=$BETA, huk + half_pair)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA \
        --head_uniform_k --half_pair_max_k
    fi
    calib_path="$VAR"
    METHOD_ARGS="--kv_quant_method smoothkv_fused --calib_path $calib_path --bits 4 --group_size $GROUP_SIZE"
    ;;
  smkv_per_channel)
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat.pt"
    VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat_a${AS}_b${BS}_per_channel.pt"
    VARIANT_TAG="smoothkv_g${GROUP_SIZE}_perc_ns${NS}_chat_a${AS}_b${BS}_per_channel"
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS, chat-calib)"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 1.0 --beta 1.0 --apply_chat_template \
        --output "$BASE" --device auto
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving per-channel variant $VAR  (α=$ALPHA β=$BETA, no_huk + no_halfpair)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA --no_head_uniform_k
      EXPECTED_BETA_OUT=$(/usr/bin/dirname "$BASE")/$(/usr/bin/basename "$BASE" .pt)_a1_b${BS}.pt
      [ -f "$EXPECTED_BETA_OUT" ] && /bin/mv "$EXPECTED_BETA_OUT" "$VAR" || true
    fi
    calib_path="$VAR"
    METHOD_ARGS="--kv_quant_method smoothkv --calib_path $calib_path --bits 4 --group_size $GROUP_SIZE"
    ;;
  *) /usr/bin/echo "variant must be bf16|fp8|pertoken|smkv_fused|smkv_per_channel"; exit 1 ;;
esac

LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_${TASK}.log
RESULT=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
/bin/mkdir -p logs/run_out logs/calib

# ---- single-pass run ----
# Set FORCE=1 to redo a cell whose outputs already exist. Verify-graph
# stamp is checked on both fresh runs and cached SKIP — see below.
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
fi
# Always validate the verify-graph stamp — fresh runs OR cached SKIP. If the
# cached log carries a FAIL or no stamp at all (e.g. legacy run that wrote
# its log under a different filename like `ex45a_${VARIANT}_${TASK}.log`),
# surface it now instead of silently inheriting an untrustworthy result.
# Backported from the May-6 qwen3/llama runner change.
if ! /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$LOG" 2>/dev/null; then
  /usr/bin/echo "ERROR: no explicit verify-graph PASS stamp at $LOG. Aborting." >&2
  /usr/bin/echo "       (cached cell with a legacy log path? rerun with FORCE=1 to refresh," >&2
  /usr/bin/echo "        or investigate the failure.)" >&2
  exit 2
fi
/usr/bin/echo "[run] verify-graph: PASS"
/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK graph-verified (.venv-exaone)"
/usr/bin/echo "   Output: $RESULT"
/usr/bin/echo "=================================================================="
