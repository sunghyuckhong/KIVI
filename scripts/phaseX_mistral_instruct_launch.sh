#!/bin/bash
# Mistral-7B-Instruct-v0.2 full-matrix eval (9 methods × 5 tasks).
# 1. Calibrate SmoothKV with reservoir (~30 min, 1 GPU) if not present
# 2. Generate 5 pair-mergeable variants (α=0.75 pair, pairK90/95/99/99p9)
# 3. Launch 9 eval streams on 4 GPUs
# max_gen_toks = 16384 for reasoning (half of Mistral's 32k context), 256 default for TQA/CoQA.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
MODERN=/opt/modernenv/bin/python
MODEL=mistralai/Mistral-7B-Instruct-v0.2
MODEL_SHORT=mistral-7b-instruct-v0.2
MAX_GEN_REASONING=16384

is_free() {
  local g=$1 mem pids
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$mem" ] || [ "$mem" -ge 20000 ] && return 1
  pids=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && return 1
  return 0
}
first_free_gpu() { for g in 0 1 2 3; do if is_free $g; then echo "$g"; return 0; fi; done; return 1; }

# ---- Step 1: Calibration (only if missing) ----
CALIB=logs/calib/smoothkv_${MODEL_SHORT}_perc.pt
if [ ! -f "$CALIB" ]; then
  echo "[pX-mistral-instruct] Step 1: calibration"
  while true; do gpu=$(first_free_gpu) && break; sleep 30; done
  bash scripts/launch_wave.sh "pX_mi_calib" "$gpu" \
    "$MODERN run_smoothkv_calibrate.py --model_path $MODEL --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output $CALIB"
  # wait for calib to finish
  while true; do
    idx=$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' | awk '$2=="pX_mi_calib"{print $1}')
    [ -z "$idx" ] && break
    tmux capture-pane -t kivi:$idx -p -S -100 2>/dev/null | grep -qE "^(DONE|FAIL)_pX_mi_calib" && break
    sleep 60
  done
  [ ! -f "$CALIB" ] && { echo "[pX-mistral-instruct] ABORT: $CALIB missing after calib"; exit 1; }
fi

# ---- Step 2: Generate pair-mergeable variants (CPU, ~2 min) ----
echo "[pX-mistral-instruct] Step 2: generating variants"
BASE_A=logs/calib/smoothkv_${MODEL_SHORT}_a0.5.pt
[ ! -f "$BASE_A" ] && BASE_A=$CALIB   # fallback: use perc file (has max_k/q/v stats too)
A075=logs/calib/smoothkv_${MODEL_SHORT}_a0.75_pair.pt
P90=logs/calib/smoothkv_${MODEL_SHORT}_pairK90_pV90.pt
P95=logs/calib/smoothkv_${MODEL_SHORT}_pairK95_pV95.pt
P99=logs/calib/smoothkv_${MODEL_SHORT}_pairK99_pV99.pt
P999=logs/calib/smoothkv_${MODEL_SHORT}_pairK99p9_pV99p9.pt
[ ! -f "$A075" ] && /usr/bin/python scripts/make_alpha_variants.py --base "$BASE_A" --alphas 0.75 --pair_max_k 2>&1 | tail -3
[ ! -f "$P999" ] && /usr/bin/python scripts/make_percentile_variants.py --base "$CALIB" --pk 90 95 99 99.9 --symmetric 2>&1 | tail -6

# ---- Step 3: Launch 9 eval streams ----
BS=16
cmd_task() {
  local method_args=$1 task=$2 extra=""
  case "$task" in
    gsm8k_32k|gpqa_diamond_cot_n_shot_32k|math500_32k)
      extra="--max_gen_toks $MAX_GEN_REASONING"
      ;;
  esac
  echo "$MODERN run_eval.py --model_path $MODEL --batch_size $BS $method_args --task $task $extra"
}
chain_stream() {
  local method_args=$1
  local parts=() t
  for t in truthfulqa_gen coqa gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k; do
    parts+=("$(cmd_task "$method_args" "$t")")
  done
  local out=""
  for c in "${parts[@]}"; do out="$out $c && "; done
  echo "${out% && }"
}

STREAMS=(
  "pX_mi_fp16|--model fp16 --k_bits 16 --v_bits 16"
  "pX_mi_kivi2|--model kivi --k_bits 2 --v_bits 2 --group_size 32 --residual 128"
  "pX_mi_pertoken|--model pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual 0"
  "pX_mi_fp8|--model fp8 --group_size 128"
  "pX_mi_a075pair|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $A075"
  "pX_mi_pairK90|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P90"
  "pX_mi_pairK95|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P95"
  "pX_mi_pairK99|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P99"
  "pX_mi_pairK99p9|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P999"
)

for entry in "${STREAMS[@]}"; do
  IFS='|' read -r name args <<< "$entry"
  while true; do gpu=$(first_free_gpu) && break; sleep 60; done
  cmd=$(chain_stream "$args")
  echo "[pX-mistral-instruct] launching $name on GPU$gpu (gen=$MAX_GEN_REASONING)"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  sleep 20
done
echo "[pX-mistral-instruct] all 9 streams launched."
