#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for Llama-3 family.
#
# Llama-3 has NO q_norm/k_norm → SmoothKV uses _pair calib (per-head s_K folded
# into W_K rows; no HUK reduction).
#
# Usage:
#   bash scripts/run_eval_llama.sh \
#       --model meta-llama/Meta-Llama-3-8B-Instruct \
#       --variant bf16|fp8|pertoken|smkv \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot_32k \
#       [--ns 512] [--alpha 1.0] [--beta 1.0] \
#       [--gpus 0]
#
# Llama-3-8B fits on TP=1 (80GB GPU). For larger Llama (70B), use TP=2 or TP=4
# and adjust --gpus accordingly.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

# ---- args ----
MODEL_PATH=""; VARIANT=""; TASK=""; GPUS="0"; NS=512; ALPHA=1.0; BETA=1.0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --ns) NS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    *) /usr/bin/echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -z "$MODEL_PATH" ] || [ -z "$VARIANT" ] || [ -z "$TASK" ] && {
  /usr/bin/echo "Required: --model HF_PATH --variant {bf16|fp8|pertoken|smkv} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot_32k}"
  exit 1
}

# Derive a short tag from model path (e.g. meta-llama/Meta-Llama-3-8B-Instruct → meta-llama-3-8b-instruct)
MODEL_TAG=$(/usr/bin/echo "${MODEL_PATH,,}" | /usr/bin/tr '/' '\n' | /usr/bin/tail -1 | /usr/bin/sed 's/^meta-//')

case "$TASK" in
  gsm8k_cot|minerva_math500) MML4=5632; MML32=34304 ;;
  gpqa_main_cot_n_shot_32k)  MML4=7168; MML32=35584 ;;
  *) /usr/bin/echo "task invalid"; exit 1 ;;
esac

# Default TP=1 (single GPU); user supplies TP=N via --gpus "0,1,..."
TP=$(/usr/bin/echo "$GPUS" | /usr/bin/tr ',' '\n' | /usr/bin/wc -l)
MNS_P1=64; MNS_P2=24
[ "$TP" -gt 1 ] && { MNS_P1=24; MNS_P2=8; }

cd /workspace/KIVI
PY=/opt/vllm_exaone_v2_env/bin/python3

# ---- variant config + calib ----
case "$VARIANT" in
  bf16)     METHOD_ARGS="--model bf16";              VARIANT_TAG="bf16" ;;
  fp8)      METHOD_ARGS="--model fp8 --group_size 128";    VARIANT_TAG="fp8_g128" ;;
  pertoken) METHOD_ARGS="--model pertoken --bits 4 --group_size 128"; VARIANT_TAG="pertoken_int4_g128" ;;
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
        --model_path "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 0.5 --beta 0.5 --samples_per_channel 10000 \
        --output "$BASE" $( [ "$TP" -gt 1 ] && /usr/bin/echo "--device auto" )
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving $VAR  (α=$ALPHA β=$BETA, pair, no HUK — Llama has no q_norm)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA \
        --pair_max_k --no_head_uniform_k
    fi
    METHOD_ARGS="--model smoothkv_fused --calib_path $VAR --bits 4 --group_size 128"
    ;;
  *) /usr/bin/echo "variant invalid"; exit 1 ;;
esac

SAMPLES=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_samples.json
RESULTS=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
ADAPTIVE=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_adaptive_results.json
P1_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass1_${TASK}.log
P2_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass2_${TASK}.log
/bin/mkdir -p logs/run_out logs/calib

# ---- Pass 1 ----
if [ ! -f "$SAMPLES" ] || [ ! -f "$RESULTS" ]; then
  /usr/bin/echo "[pass1] $TASK  on $MODEL_PATH  (TP=$TP, MG=4k)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY run_eval_vllm.py \
      $METHOD_ARGS --model_path "$MODEL_PATH" \
      --task "$TASK" --apply_chat_template \
      --max_gen_toks 4096 --max_model_len $MML4 \
      --max_num_seqs $MNS_P1 --batch_size $MNS_P1 --tp $TP \
      --log_samples 2>&1 | /usr/bin/tee "$P1_LOG"
fi
/usr/bin/grep -q "\[verify-graph\]" "$P1_LOG" || { /usr/bin/echo "ERROR: pass1 NO stamp"; exit 2; }
/usr/bin/grep "\[verify-graph\]" "$P1_LOG" | /usr/bin/grep -qE "FAIL" && { /usr/bin/echo "ERROR: pass1 FAIL"; exit 2; }
/usr/bin/echo "[pass1] verify-graph: PASS"

# ---- Pass 2 ----
if [ ! -f "$ADAPTIVE" ]; then
  /usr/bin/echo "[pass2] retry truncated subset @ MG=32k"
  CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/adaptive_pass2.py \
      --samples "$SAMPLES" --task "$TASK" --model_path "$MODEL_PATH" \
      $METHOD_ARGS \
      --pass1_mg 4096 --pass2_mg 32768 \
      --max_model_len $MML32 --max_num_seqs $MNS_P2 --tp $TP \
      2>&1 | /usr/bin/tee "$P2_LOG"
fi
/usr/bin/grep -q "\[verify-graph\]" "$P2_LOG" || { /usr/bin/echo "ERROR: pass2 NO stamp"; exit 2; }
/usr/bin/grep "\[verify-graph\]" "$P2_LOG" | /usr/bin/grep -qE "FAIL" && { /usr/bin/echo "ERROR: pass2 FAIL"; exit 2; }
/usr/bin/echo "[pass2] verify-graph: PASS"

/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK graph-verified"
/usr/bin/echo "   Output: $ADAPTIVE"
/usr/bin/echo "=================================================================="
