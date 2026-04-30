#!/bin/bash
# =============================================================================
# Qwen3-8B KV-quant accuracy harness
# =============================================================================
#
# Runs four KV-cache settings on Qwen3-8B chat-templated reasoning tasks,
# all at MG=32k single-pass (no adaptive merge, no retry script):
#
#   1. bf16              — no-quant ceiling
#   2. fp8 g=128         — software E4M3 KV (cudagraph-compat on A100 sm_80)
#   3. pertoken int4 g=128 — naive 4-bit per-token KV (KIVI-style, no smoothing)
#   4. smoothkv_fused HUK α=β=1 halfpair — SmoothKV folded into q_norm/k_norm
#                          gamma at model-load time. Zero per-step runtime cost.
#                          On Qwen3 (qk_norm), s_K must be head-uniform — the
#                          HUK halfpair calib enforces that.
#
# Tasks:
#   - gsm8k_32k                  (custom YAML, 5-shot)
#   - minerva_math500            (HuggingFaceH4/MATH-500, 4-shot, math_verify)
#   - gpqa_main_cot_n_shot_32k   (GPQA-main, 0-shot CoT)
#
# Hyperparameters (identical across all 4 variants):
#   chat template       : enabled (--apply_chat_template)
#   thinking mode       : on (Qwen3 default)
#   sampling            : greedy (temp=0, do_sample=False, seed=1234)
#   max_gen_toks        : 32768
#   max_model_len       : 34304 (gsm8k/minerva) / 35584 (gpqa)
#   max_num_seqs / batch: 8 / 8
#   tensor_parallel     : 2 (one variant per GPU pair, 4 variants × 2 GPUs = 8)
#   cudagraphs          : on (vllm 0.20+ handles all quant paths in graph mode)
#
# Scoring:
#   gsm8k         → strict-match (lm-eval default)
#   gpqa          → flexible-extract (lm-eval default)
#   minerva_math500 → math_verify (sympy boxed-aware)
#
# Output: logs/<task>_qwen3-8b_<mtag>_chat_vllm_results.json
#
# =============================================================================
# Environment (exact pinned versions — see SETUP.md for from-scratch install)
# =============================================================================
#   Python           : 3.10 (tested with 3.10.12)
#   vllm             : 0.20.1.dev0+g101584af0
#                      (git+https://github.com/lkm2835/vllm.git@add-exaone4_5)
#   transformers     : 5.6.0.dev0
#                      (git+https://github.com/nuxlear/transformers.git@31991e75)
#   torch            : 2.10.0+cu128
#   lm_eval          : 0.4.11
#   math_verify      : 0.9.0
#   antlr4-python3   : 4.11.0  (newer breaks sympy latex parser)
#   compressed-tensors: 0.15.0.1
#   numpy            : 2.2.6
#   sympy            : 1.14.0
#   datasets         : 4.8.4
#
#   GPUs   : 8× A100-80GB (sm_80) tested. Hopper sm_89+ activates hardware FP8
#            automatically; no rebuild needed.
#   Driver : NVIDIA 570.133.20 / CUDA 12.8
#
# Canonical env path on this pod: /opt/vllm_exaone_v2_env
# Override via VLLM_ENV=/path/to/your/venv
#
# =============================================================================
# Usage
# =============================================================================
#   bash scripts/qwen3-8b_kv_quant_accuracy.sh
#
#   # idempotent — skips any (variant, task) whose results.json already exists.
#   # to redo, move the old result + samples files aside first.
#
# =============================================================================
set -euo pipefail
cd "$(/usr/bin/dirname "$0")/.."

VLLM_ENV="${VLLM_ENV:-/opt/vllm_exaone_v2_env}"
PY="$VLLM_ENV/bin/python3"
[ -x "$PY" ] || { /usr/bin/echo "ERROR: $PY not found. See SETUP.md."; exit 1; }

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN="${HF_TOKEN:-$(/bin/cat ~/.cache/huggingface/token 2>/dev/null || /bin/echo '')}"
export NO_ENFORCE_EAGER=1   # vllm 0.20+ runs all quant paths in graph mode

MODEL=Qwen/Qwen3-8B
MODEL_TAG=qwen3-8b
CALIB_BASE=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512.pt
CALIB_HUK=logs/calib/smoothkv_${MODEL_TAG}_perc_ns512_puremax_a1b1_halfpair.pt

# 4 GPU pairs, one per variant, TP=2
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
LABELS=("qw8a_bf16" "qw8a_fp8" "qw8a_pert" "qw8a_smk")
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
# SmoothKV calibration (skip if already present)
# -----------------------------------------------------------------------------
if [ ! -f "$CALIB_BASE" ]; then
  /usr/bin/echo "[calib] generating base $CALIB_BASE (~30 min on 1 A100)"
  CUDA_VISIBLE_DEVICES=0 $PY run_smoothkv_calibrate.py \
    --model_path "$MODEL" \
    --num_samples 512 --seq_length 2048 \
    --alpha 0.5 --beta 0.5 \
    --samples_per_channel 10000 \
    --output "$CALIB_BASE" --device cuda:0
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
# Per-variant launch script (run inline as a shell function so we can fan out
# 4 tmux sessions in parallel, one per GPU pair).
# -----------------------------------------------------------------------------
WORKER=/tmp/qwen3-8b_kv_worker_$$.sh
/bin/cat > "$WORKER" <<'EOS'
#!/bin/bash
# Args: $1=GPU_PAIR  $2=LABEL  $3=MTAG  $4="<method args>"
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
  result=logs/${task}_qwen3-8b_${MTAG}_chat_vllm_results.json
  if [ -f "$result" ]; then
    echo "[$LABEL] SKIP $task — exists"
    continue
  fi
  echo "[$LABEL] === START $task at $(/bin/date) ==="
  $PY run_eval_vllm.py $METHOD_ARGS \
      --model_path Qwen/Qwen3-8B \
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
/usr/bin/tmux ls 2>&1 | /usr/bin/grep -E "^qw8a_"
/usr/bin/echo
/usr/bin/echo "monitor with:  tmux attach -t qw8a_<bf16|fp8|pert|smk>"
/usr/bin/echo "result files:  logs/<task>_qwen3-8b_<mtag>_chat_vllm_results.json"
/usr/bin/echo "rerun summary: bash $0  (all 12 (variant,task) cells idempotent)"
