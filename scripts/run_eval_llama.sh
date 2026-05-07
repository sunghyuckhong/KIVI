#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for Llama-3 family.
#
# Llama-3 has NO q_norm/k_norm → SmoothKV uses _pair calib (per-head s_K folded
# into W_K rows; no HUK reduction).
#
# Usage:
#   bash scripts/run_eval_llama.sh \
#       --model meta-llama/Meta-Llama-3-8B-Instruct \
#       --variant bf16|fp8|pertoken|smkv_fused|smkv_per_channel \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot \
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
  /usr/bin/echo "Required: --model HF_PATH --variant {bf16|fp8|pertoken|smkv_fused|smkv_per_channel} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot}"
  exit 1
}

# Derive a short tag from model path. MUST match what run_eval_vllm.py's
# output_name() produces (model.rstrip('/').split('/')[-1].lower()) so the
# runner-built RESULTS path matches lm-eval's actual output file. Don't
# strip 'meta-' or any other prefix.
MODEL_TAG=$(/usr/bin/echo "${MODEL_PATH,,}" | /usr/bin/tr '/' '\n' | /usr/bin/tail -1)

case "$TASK" in
  gsm8k_cot|minerva_math500) PROMPT_BUDGET=1536 ;;
  gpqa_main_cot_n_shot)  PROMPT_BUDGET=3072 ;;
  *) /usr/bin/echo "task invalid"; exit 1 ;;
esac

# Default TP=1 (single GPU); user supplies TP=N via --gpus "0,1,..."
TP=$(/usr/bin/echo "$GPUS" | /usr/bin/tr ',' '\n' | /usr/bin/wc -l)
MNS_P1=64; MNS_P2=24
[ "$TP" -gt 1 ] && { MNS_P1=24; MNS_P2=8; }

cd /workspace/KIVI
# PY: python interpreter (defaults to .venv from `make setup`).
PY="${PY:-./.venv/bin/python}"

# ---- derive max-gen-tokens from model's native context length ----
# Rule: if model_max_len >= 32k, pass2 generates up to 32k; else pass2 is
# capped at model_max_len/2. pass1 is min(4096, pass2). When pass1 == pass2,
# pass2 is redundant and we skip it (pass1 stands).
MODEL_MAX_LEN=$($PY -c "from transformers import AutoConfig; \
print(AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True).max_position_embeddings)")
# Use MG=32k only when model context is STRICTLY larger than 32k (room for
# the prompt budget). Otherwise PASS2_MG = model_max_len / 2 keeps room for
# both prompt + generation within the model's context window.
# Examples: Qwen3 (40960) → 32768; Mistral-7B-Instruct-v0.2 (32768) → 16384;
# Llama-3-8B (8192) → 4096.
if [ "$MODEL_MAX_LEN" -gt 32768 ]; then PASS2_MG=32768; else PASS2_MG=$((MODEL_MAX_LEN / 2)); fi
if [ "$PASS2_MG" -lt 4096 ]; then PASS1_MG=$PASS2_MG; else PASS1_MG=4096; fi
MML4=$((PASS1_MG + PROMPT_BUDGET))
MML32=$((PASS2_MG + PROMPT_BUDGET))
/usr/bin/echo "[mg] model_max_len=$MODEL_MAX_LEN  →  pass1_mg=$PASS1_MG (mml=$MML4), pass2_mg=$PASS2_MG (mml=$MML32)"

# ---- variant config + calib ----
case "$VARIANT" in
  bf16)     METHOD_ARGS="--kv_quant_method bf16";              VARIANT_TAG="bf16" ;;
  fp8)      METHOD_ARGS="--kv_quant_method fp8 --group_size 128";    VARIANT_TAG="fp8_g128" ;;
  pertoken) METHOD_ARGS="--kv_quant_method pertoken --bits 4 --group_size 128"; VARIANT_TAG="pertoken_int4_g128" ;;
  smkv_fused)
    # Match Qwen3 runner's calib structure (run_eval_qwen3.sh smkv_fused branch):
    # BASE is generated with α=1.0 β=1.0 so make_alpha_variants's betas-block
    # writes a file whose name encodes both alpha and beta (`_a${AS}_b${BS}_`),
    # which is what we then pick up as VAR. No q_norm here, so use plain
    # `--pair_max_k --no_head_uniform_k` (vs Qwen3's halfpair+huk).
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}.pt"
    VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_a${AS}_b${BS}_pair.pt"
    VARIANT_TAG="smoothkv_fused_g128_perc_ns${NS}_a${AS}_b${BS}_pair"
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS, alpha=1.0 beta=1.0)"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 1.0 --beta 1.0 \
        --output "$BASE" $( [ "$TP" -gt 1 ] && /usr/bin/echo "--device auto" )
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving $VAR  (α=$ALPHA β=$BETA, pair, no HUK — Llama/Mistral have no q_norm)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA \
        --pair_max_k --no_head_uniform_k
    fi
    METHOD_ARGS="--kv_quant_method smoothkv_fused --calib_path $VAR --bits 4 --group_size 128"
    ;;
  smkv_per_channel)
    # Per-(layer, kv_head, head_dim) UNIQUE smoothing factors via runtime
    # smoothkv kernel (post-RoPE). Llama-3/Mistral have no q_norm/k_norm,
    # so the head-uniform constraint that fused-Qwen3 needs doesn't apply.
    # Runtime kernel applies post-RoPE so half-pair constraint isn't needed.
    #
    # CHAT_CALIB defaults to 1 — calib generation uses --apply_chat_template
    # to match the eval-time prompt distribution (we always run eval with
    # chat template applied). Use CHAT_CALIB=0 for raw-text calibration.
    CHAT_CALIB="${CHAT_CALIB:-1}"
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    if [ "$CHAT_CALIB" = "1" ]; then
      BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat.pt"
      VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_chat_a${AS}_b${BS}_per_channel.pt"
      VARIANT_TAG="smoothkv_g128_perc_ns${NS}_chat_a${AS}_b${BS}_per_channel"
      calib_chat_flag="--apply_chat_template"
    else
      BASE="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}.pt"
      VAR="logs/calib/smoothkv_${MODEL_TAG}_perc_ns${NS}_a${AS}_b${BS}_per_channel.pt"
      VARIANT_TAG="smoothkv_g128_perc_ns${NS}_a${AS}_b${BS}_per_channel"
      calib_chat_flag=""
    fi
    if [ ! -f "$BASE" ]; then
      /usr/bin/echo "[calib] generating base $BASE  (n_s=$NS, $( [ -n "$calib_chat_flag" ] && /usr/bin/echo chat-calib || /usr/bin/echo raw-calib ))"
      CUDA_VISIBLE_DEVICES=$GPUS $PY run_smoothkv_calibrate.py \
        --model "$MODEL_PATH" \
        --num_samples $NS --seq_length 2048 \
        --alpha 1.0 --beta 1.0 \
        $calib_chat_flag \
        --output "$BASE" $( [ "$TP" -gt 1 ] && /usr/bin/echo "--device auto" )
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving per-channel variant $VAR  (α=$ALPHA β=$BETA, no HUK, no halfpair)"
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA --no_head_uniform_k
      EXPECTED_BETA_OUT=$(/usr/bin/dirname "$BASE")/$(/usr/bin/basename "$BASE" .pt)_a1_b${BS}.pt
      [ -f "$EXPECTED_BETA_OUT" ] && /bin/mv "$EXPECTED_BETA_OUT" "$VAR" || true
    fi
    METHOD_ARGS="--kv_quant_method smoothkv --calib_path $VAR --bits 4 --group_size 128"
    ;;
  *) /usr/bin/echo "variant invalid"; exit 1 ;;
esac

SAMPLES=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_samples.json
RESULTS=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
ADAPTIVE=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_adaptive_results.json
P1_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass1_${TASK}.log
P2_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass2_${TASK}.log
/bin/mkdir -p logs/run_out logs/calib

# ---- Trust gate scenarios (see README.md "Trust gate" table) ----
#   A  pass-1 only: pass1_mg == pass2_mg → pass-2 entirely skipped (cp _results
#      → _adaptive_results). Pass-1 must emit [PASS]. Common for Llama-3-8B
#      (mml=8192).
#   B  both passes ran: pass-1 had ≥1 truncated sample → pass-2 retries.
#      Both passes must emit [PASS].
#   C  both passes ran, pass-2 noop: pass-1 had 0 truncated samples →
#      adaptive_pass2.py emits [PASS] directly without launching vLLM. Common
#      for Mistral on short-output tasks (gsm8k_cot).
#   D  cached skip: FORCE=0 + outputs exist → runner skips both passes,
#      no fresh stamps. Existing _adaptive_results.json carries whatever
#      stamps the prior invocation generated. Use FORCE=1 to re-validate.
#
# ---- Pass 1 ----
# Set FORCE=1 to redo a cell whose outputs already exist.
FORCE="${FORCE:-0}"
if [ "$FORCE" = "1" ] || [ ! -f "$SAMPLES" ] || [ ! -f "$RESULTS" ]; then
  /usr/bin/echo "[pass1] $TASK  on $MODEL_PATH  (TP=$TP, MG=$PASS1_MG)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY run_eval_vllm.py \
      $METHOD_ARGS --model "$MODEL_PATH" \
      --task "$TASK" --apply_chat_template \
      --max_gen_toks $PASS1_MG --max_model_len $MML4 \
      --max_num_seqs $MNS_P1 --batch_size $MNS_P1 --tp $TP \
      --log_samples 2>&1 | /usr/bin/tee "$P1_LOG"
else
  /usr/bin/echo "[pass1] SKIP — samples + results exist (set FORCE=1 to override)"
fi
# Always validate pass-1 stamp — fresh runs OR cached SKIP. Surfaces FAIL on
# stale cached cells instead of silently inheriting them.
/usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$P1_LOG" 2>/dev/null \
  || { /usr/bin/echo "ERROR: pass1 has no PASS stamp at $P1_LOG (cached cell? rerun FORCE=1)"; exit 2; }
/usr/bin/echo "[pass1] verify-graph: PASS"

# ---- Pass 2 (skip if pass1_mg == pass2_mg — pass2 would just re-run identical) ----
if [ "$PASS1_MG" -eq "$PASS2_MG" ]; then
  /usr/bin/echo "[pass2] skipped (pass1_mg == pass2_mg == $PASS1_MG; model_max_len=$MODEL_MAX_LEN doesn't allow longer retry)"
  /bin/cp "$RESULTS" "$ADAPTIVE"
  /usr/bin/echo "[pass2] verify-graph: N/A (Scenario A — pass-2 never invoked, mml=$MODEL_MAX_LEN)"
elif [ "$FORCE" = "1" ] || [ ! -f "$ADAPTIVE" ]; then
  /usr/bin/echo "[pass2] retry truncated subset @ MG=$PASS2_MG"
  CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/adaptive_pass2.py \
      --samples "$SAMPLES" --task "$TASK" --model "$MODEL_PATH" \
      $METHOD_ARGS \
      --pass1_mg $PASS1_MG --pass2_mg $PASS2_MG \
      --max_model_len $MML32 --max_num_seqs $MNS_P2 --tp $TP \
      2>&1 | /usr/bin/tee "$P2_LOG"
  /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$P2_LOG" 2>/dev/null \
    || { /usr/bin/echo "ERROR: pass2 has no PASS stamp at $P2_LOG"; exit 2; }
  /usr/bin/echo "[pass2] verify-graph: PASS"
else
  /usr/bin/echo "[pass2] SKIP — adaptive_results.json exists (set FORCE=1 to override)"
  # Validate cached pass-2 stamp from prior run (Scenario D).
  /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$P2_LOG" 2>/dev/null \
    || { /usr/bin/echo "ERROR: pass2 cached but has no PASS stamp at $P2_LOG (rerun FORCE=1)"; exit 2; }
  /usr/bin/echo "[pass2] verify-graph: PASS (from cached log)"
fi

/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK graph-verified"
/usr/bin/echo "   Output: $ADAPTIVE"
/usr/bin/echo "=================================================================="
