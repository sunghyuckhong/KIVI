#!/bin/bash
# =============================================================================
# Qwen3-32B KV-quant accuracy harness
# =============================================================================
#
# Same shape as scripts/qwen3-8b_kv_quant_accuracy.sh — 4 KV-cache settings on
# Qwen3-32B chat-templated reasoning tasks at MG=32k single-pass:
#
#   1. bf16
#   2. fp8 g=128
#   3. pertoken int4 g=128 (naive)
#   4. smoothkv_fused HUK α=β=1 halfpair (zero-runtime, gamma-folded into k_norm)
#
# Tasks: gsm8k_32k, minerva_math500, gpqa_main_cot_n_shot_32k
#
# Layout: 4 GPU pairs × TP=2 = 8 GPUs. One variant per pair. Each variant
# walks 3 tasks sequentially.
#
# Hyperparameters / scoring: identical to the 8B harness — same chat template,
# greedy seed=1234, max_gen_toks=32768, max_num_seqs=8, math_verify on math500.
# Only differences vs 8B:
#   - MODEL=Qwen/Qwen3-32B
#   - 64-layer model (calib s_K shape (64, 8, 128) instead of (36, 8, 128))
#   - Bigger weights → ~30 GiB / GPU at TP=2 (vs ~15 GiB on 8B)
#
# =============================================================================
# Environment
# =============================================================================
# Same as 8B — see scripts/qwen3-8b_kv_quant_accuracy.sh "Environment" section
# for exact pinned versions, or SETUP.md for from-scratch install.
#
# Canonical env path: /opt/vllm_exaone_v2_env  (override via VLLM_ENV)
#
# =============================================================================
# Usage
# =============================================================================
#   bash scripts/qwen3-32b_kv_quant_accuracy.sh
#
#   ETA on 8× A100-80GB: ~6-8 hours wall clock.
#   gsm8k_32k is the bottleneck (1319 problems × ~5 s/it × 4 streams).
#
# =============================================================================
set -euo pipefail
cd "$(/usr/bin/dirname "$0")/.."

VLLM_ENV="${VLLM_ENV:-/opt/vllm_exaone_v2_env}"
PY="$VLLM_ENV/bin/python3"
[ -x "$PY" ] || { /usr/bin/echo "ERROR: $PY not found. See SETUP.md."; exit 1; }

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1

MODEL=Qwen/Qwen3-32B
MODEL_TAG=qwen3-32b
CALIB_BASE=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512.pt
CALIB_HUK=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512_puremax_a1b1_halfpair.pt

GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
LABELS=("qw32a_bf16" "qw32a_fp8" "qw32a_pert" "qw32a_smk")
MTAGS=("bf16" "fp8_g128" "pertoken_int4_g128"
       "smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair")
METHOD_ARGS=(
  "--model bf16"
  "--model fp8 --group_size 128"
  "--model pertoken --bits 4 --group_size 128"
  "--model smoothkv_fused --calib_path $CALIB_HUK --group_size 128 --bits 4"
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
if [ ! -f "$CALIB_HUK" ]; then
  /usr/bin/echo "[calib] deriving HUK halfpair α=β=1 variant"
  $PY scripts/make_alpha_variants.py \
    --base "$CALIB_BASE" \
    --alpha 1.0 --beta 1.0 \
    --pair_max_k --half_pair_max_k --head_uniform_k \
    --output "$CALIB_HUK"
fi

# -----------------------------------------------------------------------------
# Per-variant worker
# -----------------------------------------------------------------------------
WORKER=/tmp/qwen3-32b_kv_worker_$$.sh
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
  result=logs/${task}_qwen3-32b_${MTAG}_chat_vllm_results.json
  if [ -f "$result" ]; then
    echo "[$LABEL] SKIP $task — exists"
    continue
  fi
  echo "[$LABEL] === START $task at $(/bin/date) ==="
  $PY run_eval_vllm.py $METHOD_ARGS \
      --model_path Qwen/Qwen3-32B \
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

# -----------------------------------------------------------------------------
# Launch 4 parallel streams
# -----------------------------------------------------------------------------
for i in 0 1 2 3; do
  /usr/bin/tmux new-session -d -s "${LABELS[$i]}" \
    "$WORKER ${GPU_PAIRS[$i]} ${LABELS[$i]} ${MTAGS[$i]} '${METHOD_ARGS[$i]}'"
done

/usr/bin/echo "launched 4 streams (TP=2 each):"
/usr/bin/tmux ls 2>&1 | /usr/bin/grep -E "^qw32a_"
/usr/bin/echo
/usr/bin/echo "monitor with:  tmux attach -t qw32a_<bf16|fp8|pert|smk>"
/usr/bin/echo "result files:  logs/<task>_qwen3-32b_<mtag>_chat_vllm_results.json"
