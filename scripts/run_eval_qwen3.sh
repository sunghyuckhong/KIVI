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
#       --variant bf16|fp8|pertoken|smkv|smkv_per_channel \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot \
#       [--ns 512]                     # SmoothKV calib sample count
#       [--alpha 1.0] [--beta 1.0]     # SmoothKV α/β
#       [--no_chat_calib]              # skip chat template at calibration (default: chat-calib ON)
#       [--gpus 0]                     # CUDA_VISIBLE_DEVICES (single int for 8B, "a,b" for 32B TP=2)
#
# Examples:
#   # Qwen3-8B BF16 on gsm8k_cot (single GPU)
#   bash scripts/run_eval_qwen3.sh --size 8b --variant bf16 --task gsm8k_cot --gpus 0
#
#   # Qwen3-32B SmoothKV α=1 chat-calib (default) n_s=512 on minerva (TP=2)
#   bash scripts/run_eval_qwen3.sh --size 32b --variant smkv --gpus 0,1 \
#       --task minerva_math500
#
#   # Same but with raw-text calibration instead of chat-template
#   bash scripts/run_eval_qwen3.sh --size 32b --variant smkv --no_chat_calib --gpus 0,1 \
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
# CHAT_CALIB defaults to 1: chat-template applied at calibration time matches
# the eval-time prompt distribution (we always run with --apply_chat_template
# at eval), which is the SmoothKV setting we ship in headline numbers. Use
# --no_chat_calib to opt into raw-text calibration for ablation.
SIZE=""; VARIANT=""; TASK=""; GPUS="0"; NS=512; ALPHA=1.0; BETA=1.0; CHAT_CALIB=1
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
    --no_chat_calib) CHAT_CALIB=0; shift ;;
    *) /usr/bin/echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[ -z "$SIZE" ] || [ -z "$VARIANT" ] || [ -z "$TASK" ] && {
  /usr/bin/echo "Required: --size {8b|32b|30b-a3b} --variant {bf16|fp8|pertoken|smkv|smkv_per_channel} --task {gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot}"
  exit 1
}

case "$SIZE" in 8b|32b|30b-a3b) ;; *) /usr/bin/echo "size must be 8b|32b|30b-a3b"; exit 1 ;; esac
case "$SIZE" in
  30b-a3b) MODEL_PATH="Qwen/Qwen3-30B-A3B" ;;
  *)       MODEL_PATH="Qwen/Qwen3-${SIZE^^}" ;;
esac
# Match run_eval_vllm.py's output_name(): basename(model).lower(). For 8b/32b
# this is unchanged ("qwen3-8b" / "qwen3-32b"); for 30b-a3b it now correctly
# reflects the full HF id ("qwen3-30b-a3b-instruct-2507"), so pass-2 finds
# the samples pass-1 wrote.
MODEL_TAG=$(/usr/bin/basename "$MODEL_PATH" | /usr/bin/tr '[:upper:]' '[:lower:]')

case "$TASK" in
  gsm8k_cot|minerva_math500) PROMPT_BUDGET=1536 ;;
  gpqa_main_cot_n_shot)  PROMPT_BUDGET=3072 ;;
  *) /usr/bin/echo "task must be gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot"; exit 1 ;;
esac

# TP based on size (8B → TP=1, 32B/30B-A3B → TP=2). Default max_num_seqs.
# Qwen3-30B-A3B is MoE (3B active), so KV memory budget is ~3× the 32B dense
# model — use a higher MNS for throughput.
case "$SIZE" in
  8b)      TP=1; MNS_P1=64; MNS_P2=24 ;;
  32b)     TP=2; MNS_P1=24; MNS_P2=8 ;;
  30b-a3b) TP=2; MNS_P1=32; MNS_P2=20 ;;  # MoE: ~3× KV headroom vs 32B dense
esac

cd /workspace/KIVI
# PY: python interpreter to use. Defaults to the .venv that `make setup`
# builds (which has the rebased fork's `kv_cache_quant_config` support).
# Override with PY=... for a different env.
PY="${PY:-./.venv/bin/python}"

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
  smkv_per_channel)
    # Per-(layer, kv_head, head_dim) UNIQUE smoothing factors. For Qwen3-8B
    # that's 36 layers × 8 kv_heads × 128 = 36864 unique s_K values (and
    # same for s_V) — versus the head-uniform `smkv` variant which shares
    # a single (head_dim,) row across all heads in a layer.
    #
    # Why a separate variant: the fused path (smoothkv_fused) folds s_K
    # into q_norm.γ / k_norm.γ which are shape (head_dim,) shared across
    # heads — that fusion mathematically requires head-uniform s_K. The
    # non-fused runtime path (vllm_kv_quant::smoothkv kernel in the fork)
    # applies s_K per-(head, channel) at attention time AFTER RoPE, so it
    # has no such constraint.
    #
    # Note: s_K is applied symmetrically here only for K-side quantization
    # preconditioning (K /= s_K → quant → K *= s_K). Q is untouched at
    # runtime — there's no Q-side scale. So this is a per-channel quant
    # range balancer, not a SmoothQuant-style activation migration.
    fmt() { /usr/bin/awk -v v="$1" 'BEGIN{ if(v==int(v)) printf "%d", v; else printf "%g", v; }'; }
    AS=$(fmt $ALPHA); BS=$(fmt $BETA)
    if [ "$CHAT_CALIB" -eq 1 ]; then
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
        --output "$BASE" \
        $( [ "$SIZE" = "32b" ] && /usr/bin/echo "--device auto" )
    fi
    if [ ! -f "$VAR" ]; then
      /usr/bin/echo "[calib] deriving per-head variant $VAR  (α=$ALPHA β=$BETA, no_huk + no_halfpair)"
      # --no_head_uniform_k: keep per-head granularity (override Qwen3 auto-HUK).
      # No --half_pair_max_k: smoothkv runtime kernel applies post-RoPE, so the
      # i/i+d/2 pair-equal constraint (needed for fused-pre-RoPE) is unnecessary.
      $PY scripts/make_alpha_variants.py \
        --base "$BASE" --alphas $ALPHA --betas $BETA --no_head_uniform_k
      # make_alpha_variants writes the alpha-loop file as `..._a${AS}.pt` and
      # the betas-loop file as `..._a${base_α}_b${BS}.pt`. Base α=1, so the
      # betas-loop output (which has both α and β in name) is what we want.
      EXPECTED_BETA_OUT=$(/usr/bin/dirname "$BASE")/$(/usr/bin/basename "$BASE" .pt)_a1_b${BS}.pt
      [ -f "$EXPECTED_BETA_OUT" ] && /bin/mv "$EXPECTED_BETA_OUT" "$VAR" || true
    fi
    calib_path="$VAR"
    # Use the runtime smoothkv kernel (NOT smoothkv_fused) so per-head s_K applies.
    METHOD_ARGS="--kv_quant_method smoothkv --calib_path $calib_path --bits 4 --group_size 128"
    ;;
  *) /usr/bin/echo "variant must be bf16|fp8|pertoken|smkv|smkv_per_channel"; exit 1 ;;
esac

SAMPLES=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_samples.json
RESULTS=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_results.json
ADAPTIVE=logs/${TASK}_${MODEL_TAG}_${VARIANT_TAG}_chat_vllm_adaptive_results.json
P1_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass1_${TASK}.log
P2_LOG=logs/run_out/${MODEL_TAG}_${VARIANT_TAG}_pass2_${TASK}.log
/bin/mkdir -p logs/run_out logs/calib

# ---- Pass 1 ----
# Skip if outputs exist AND FORCE != 1. Set FORCE=1 to redo a cell (e.g. to
# pick up a new verify-graph stamp after upgrading the fork).
FORCE="${FORCE:-0}"
if [ "$FORCE" != "1" ] && [ -f "$SAMPLES" ] && [ -f "$RESULTS" ]; then
  /usr/bin/echo "[pass1] SKIP — samples + results exist (set FORCE=1 to override)"
else
  /usr/bin/echo "[pass1] $TASK  on $MODEL_PATH  (TP=$TP, MG=$PASS1_MG, max_num_seqs=$MNS_P1)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY run_eval_vllm.py \
      $METHOD_ARGS \
      --model "$MODEL_PATH" \
      --task "$TASK" --apply_chat_template \
      --max_gen_toks $PASS1_MG --max_model_len $MML4 \
      --max_num_seqs $MNS_P1 --batch_size $MNS_P1 --tp $TP \
      --log_samples 2>&1 | /usr/bin/tee "$P1_LOG"
  # Verify pass1 stamp (only on a fresh run — SKIP path trusts existing data;
  # use FORCE=1 to redo + re-stamp).
  if ! /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$P1_LOG" 2>/dev/null; then
    /usr/bin/echo "ERROR: pass1 has no explicit verify-graph PASS stamp. Aborting." >&2
    exit 2
  fi
  /usr/bin/echo "[pass1] verify-graph: PASS"
fi

# ---- Pass 2 (skip if pass1_mg == pass2_mg — pass2 would be redundant) ----
PASS2_RAN=0
if [ "$PASS1_MG" -eq "$PASS2_MG" ]; then
  /usr/bin/echo "[pass2] skipped (pass1_mg == pass2_mg == $PASS1_MG; model_max_len=$MODEL_MAX_LEN doesn't allow longer retry)"
  /bin/cp "$RESULTS" "$ADAPTIVE"
elif [ "$FORCE" != "1" ] && [ -f "$ADAPTIVE" ]; then
  /usr/bin/echo "[pass2] SKIP — adaptive_results.json exists (set FORCE=1 to override)"
else
  /usr/bin/echo "[pass2] $TASK  retry truncated subset @ MG=$PASS2_MG  (max_num_seqs=$MNS_P2)"
  CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/adaptive_pass2.py \
      --samples "$SAMPLES" --task "$TASK" --model "$MODEL_PATH" \
      $METHOD_ARGS \
      --pass1_mg $PASS1_MG --pass2_mg $PASS2_MG \
      --max_model_len $MML32 --max_num_seqs $MNS_P2 --tp $TP \
      2>&1 | /usr/bin/tee "$P2_LOG"
  PASS2_RAN=1
fi

# Verify pass2 stamp only if pass2 actually ran in this session
# (SKIP path trusts existing data; use FORCE=1 to redo + re-stamp).
if [ "$PASS2_RAN" = "1" ]; then
  if ! /usr/bin/grep -qE "\[verify-graph\] (\[PASS\]|✅[[:space:]]*PASS)" "$P2_LOG" 2>/dev/null; then
    /usr/bin/echo "ERROR: pass2 has no explicit verify-graph PASS stamp. Aborting." >&2
    exit 2
  fi
  /usr/bin/echo "[pass2] verify-graph: PASS"
fi

/usr/bin/echo ""
/usr/bin/echo "=================================================================="
/usr/bin/echo "✅ DONE — $MODEL_TAG / $VARIANT / $TASK fully graph-verified"
/usr/bin/echo "   Output: $ADAPTIVE"
/usr/bin/echo "=================================================================="
