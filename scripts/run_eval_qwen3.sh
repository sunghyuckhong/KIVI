#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for Qwen3-8B / Qwen3-32B.
#
# Two-pass adaptive (pass1 @ MG=4k, pass2 @ MG=32k merged) with verify-graph
# stamps required at both passes. Uses lm-eval-style scoring at the post-hoc
# rescoring step (correct gpqa flex regex, gsm8k upstream flex).
#
# Usage:
#   bash scripts/run_eval_qwen3.sh \
#       --size 8b|32b \
#       --variant bf16|fp8|pertoken|smkv \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot_32k \
#       [--ns 512]                     # SmoothKV calib sample count
#       [--alpha 1.0] [--beta 1.0]     # SmoothKV α/β
#       [--chat_calib]                 # apply chat template at calibration time
#       [--gpus 0]                     # CUDA_VISIBLE_DEVICES (single int for 8B, "a,b" for 32B TP=2)
#
# Examples:
#   # Qwen3-8B BF16 on gsm8k_cot (single GPU)
#   bash scripts/run_eval_qwen3.sh --size 8b --variant bf16 --task gsm8k_cot --gpus 0
#
#   # Qwen3-32B SmoothKV α=1 chat-calib n_s=512 on minerva (TP=2)
#   bash scripts/run_eval_qwen3.sh --size 32b --variant smkv --chat_calib --gpus 0,1 \
#       --task minerva_math500
#
# Output:
#   logs/<task>_qwen3-<size>_<variant_tag>_chat_vllm_adaptive_results.json
#     contains pass2-merged scores. Both pass1 and pass2 logs contain
#     `[verify-graph] ✅ PASS` — the script aborts if either is missing.
#
# Env:
#   /opt/vllm_exaone_v2_env (vllm 0.20.1 + Qwen3 patches)
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

# ---- args ----
SIZE=""; VARIANT=""; TASK=""; GPUS="0"; NS=512; ALPHA=1.0; BETA=1.0; CHAT_CALIB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --size) SIZE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --ns) NS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --chat_calib) CHAT_CALIB=1; shift ;;
    *) /usr/bin/echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -z "$SIZE" ] || [ -z "$VARIANT" ] || [ -z "$TASK" ] && {
  /usr/bin/echo "Required: --size {8b|32b} --variant {bf16|fp8|pertoken|smkv} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot_32k}"
  exit 1
}

case "$SIZE" in 8b|32b) ;; *) /usr/bin/echo "size must be 8b or 32b"; exit 1 ;; esac
MODEL_PATH="Qwen/Qwen3-${SIZE^^}"
MODEL_TAG="qwen3-${SIZE}"

case "$TASK" in
  gsm8k_cot|minerva_math500) PROMPT_BUDGET=1536 ;;
  gpqa_main_cot_n_shot_32k)  PROMPT_BUDGET=3072 ;;
  *) /usr/bin/echo "task must be gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot_32k"; exit 1 ;;
esac

# TP based on size (8B → TP=1, 32B → TP=2). Default max_num_seqs.
if [ "$SIZE" = "8b" ]; then TP=1; MNS_P1=64; MNS_P2=24; else TP=2; MNS_P1=24; MNS_P2=8; fi

cd /workspace/KIVI
PY=/opt/vllm_exaone_v2_env/bin/python3

# ---- derive max-gen-tokens from model's native context length ----
# Rule: pass2_mg = 32k if model_max_len >= 32k else model_max_len/2.
# pass1_mg = min(4096, pass2_mg). pass2 is skipped if pass1 == pass2.
MODEL_MAX_LEN=$($PY -c "from transformers import AutoConfig; \
print(AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True).max_position_embeddings)")
if [ "$MODEL_MAX_LEN" -ge 32768 ]; then PASS2_MG=32768; else PASS2_MG=$((MODEL_MAX_LEN / 2)); fi
if [ "$PASS2_MG" -lt 4096 ]; then PASS1_MG=$PASS2_MG; else PASS1_MG=4096; fi
MML4=$((PASS1_MG + PROMPT_BUDGET))
MML32=$((PASS2_MG + PROMPT_BUDGET))
/usr/bin/echo "[mg] model_max_len=$MODEL_MAX_LEN  →  pass1_mg=$PASS1_MG (mml=$MML4), pass2_mg=$PASS2_MG (mml=$MML32)"

# ---- variant config + calib (SmoothKV only) ----
calib_path=""
case "$VARIANT" in
  bf16)     METHOD_ARGS="--kv_quant_method bf16";              VARIANT_TAG="bf16" ;;
  fp8)      METHOD_ARGS="--kv_quant_method fp8 --group_size 128";    VARIANT_TAG="fp8_g128" ;;
  pertoken) METHOD_ARGS="--kv_quant_method pertoken --bits 4 --group_size 128"; VARIANT_TAG="pertoken_int4_g128" ;;
  smkv)
    # Format alpha/beta into filename (e.g. 1 → "1", 0.75 → "0.75")
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    if [ "$CHAT_CALIB" -eq 1 ]; then
      BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat.pt"
      VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat_a${AS}_b${BS}_huk_halfpair.pt"
      VARIANT_TAG="smoothkv_fused_g128_perc_ns${NS}_chat_a${AS}_b${BS}_huk_halfpair"
      calib_chat_flag="--apply_chat_template"
    else
      BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}.pt"
      VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_a${AS}_b${BS}_halfpair.pt"
      [ "$AS" = "1" ] && [ "$BS" = "1" ] && VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_puremax_a1b1_halfpair.pt"
      VARIANT_TAG=$(/usr/bin/basename "$VAR" .pt | /usr/bin/sed "s/smoothkv_${MODEL_TAG}_perc/smoothkv_fused_g128_perc/")
      calib_chat_flag=""
    fi
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS, $( [ -n "$calib_chat_flag" ] && /usr/bin/echo chat-calib || /usr/bin/echo raw-calib ))"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 1.0 --beta 1.0 \
        $calib_chat_flag \
        --output "$BASE" \
        $( [ "$SIZE" = "32b" ] && /usr/bin/echo "--device auto" )
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving variant $VAR  (α=$ALPHA β=$BETA)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA --half_pair_max_k
    fi
    calib_path="$VAR"
    METHOD_ARGS="--kv_quant_method smoothkv_fused --calib_path $calib_path --bits 4 --group_size 128"
    ;;
  *) /usr/bin/echo "variant must be bf16|fp8|pertoken|smkv"; exit 1 ;;
esac

SAMPLES=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_samples.json
RESULTS=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
ADAPTIVE=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_adaptive_results.json
P1_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass1_${TASK}.log
P2_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass2_${TASK}.log
/bin/mkdir -p logs/run_out logs/calib

# ---- Pass 1 ----
if [ -f "$SAMPLES" ] && [ -f "$RESULTS" ]; then
  /usr/bin/echo "[pass1] SKIP — samples + results exist"
else
  /usr/bin/echo "[pass1] $TASK  on $MODEL_PATH  (TP=$TP, MG=$PASS1_MG, max_num_seqs=$MNS_P1)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY run_eval_vllm.py \
      $METHOD_ARGS \
      --model "$MODEL_PATH" \
      --task "$TASK" --apply_chat_template \
      --max_gen_toks $PASS1_MG --max_model_len $MML4 \
      --max_num_seqs $MNS_P1 --batch_size $MNS_P1 --tp $TP \
      --log_samples 2>&1 | /usr/bin/tee "$P1_LOG"
fi

# Verify pass1 stamp
if ! /usr/bin/grep -q "\[verify-graph\]" "$P1_LOG" 2>/dev/null; then
  /usr/bin/echo "ERROR: pass1 has NO verify-graph stamp. Aborting." >&2
  exit 2
fi
if /usr/bin/grep "\[verify-graph\]" "$P1_LOG" | /usr/bin/grep -qE "FAIL"; then
  /usr/bin/echo "ERROR: pass1 verify-graph FAILED. Aborting." >&2
  exit 2
fi
/usr/bin/echo "[pass1] verify-graph: PASS"

# ---- Pass 2 (skip if pass1_mg == pass2_mg — pass2 would be redundant) ----
if [ "$PASS1_MG" -eq "$PASS2_MG" ]; then
  /usr/bin/echo "[pass2] skipped (pass1_mg == pass2_mg == $PASS1_MG; model_max_len=$MODEL_MAX_LEN doesn't allow longer retry)"
  /bin/cp "$RESULTS" "$ADAPTIVE"
elif [ -f "$ADAPTIVE" ]; then
  /usr/bin/echo "[pass2] SKIP — adaptive_results.json exists"
else
  /usr/bin/echo "[pass2] $TASK  retry truncated subset @ MG=$PASS2_MG  (max_num_seqs=$MNS_P2)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/adaptive_pass2.py \
      --samples "$SAMPLES" --task "$TASK" --model "$MODEL_PATH" \
      $METHOD_ARGS \
      --pass1_mg $PASS1_MG --pass2_mg $PASS2_MG \
      --max_model_len $MML32 --max_num_seqs $MNS_P2 --tp $TP \
      2>&1 | /usr/bin/tee "$P2_LOG"
fi

# Verify pass2 stamp (only if pass2 actually ran)
if [ "$PASS1_MG" -ne "$PASS2_MG" ]; then
  if ! /usr/bin/grep -q "\[verify-graph\]" "$P2_LOG" 2>/dev/null; then
    /usr/bin/echo "ERROR: pass2 has NO verify-graph stamp. Aborting." >&2
    exit 2
  fi
  if /usr/bin/grep "\[verify-graph\]" "$P2_LOG" | /usr/bin/grep -qE "FAIL"; then
    /usr/bin/echo "ERROR: pass2 verify-graph FAILED. Aborting." >&2
    exit 2
  fi
  /usr/bin/echo "[pass2] verify-graph: PASS"
fi

/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK fully graph-verified"
/usr/bin/echo "   Output: $ADAPTIVE"
/usr/bin/echo "=================================================================="
