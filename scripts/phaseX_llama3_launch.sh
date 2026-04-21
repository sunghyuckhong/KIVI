#!/bin/bash
# Full eval for Llama-3-8B base (meta-llama/Meta-Llama-3-8B):
# 1. SmoothKV calibration (~30 min, 1 GPU, other GPUs idle)
# 2. Generate pair-mergeable variants (α=0.75 pair, pairK90/95/99/99p9)
# 3. Run 9 methods × 5 tasks via run_eval.py (modern_env for transformers 4.43+).
# max_gen_toks: 256 for TQA/CoQA (HFLM default), 4096 for reasoning (half of 8k context).
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
MODERN=/opt/modernenv/bin/python
MODEL=meta-llama/Meta-Llama-3-8B
MODEL_SHORT=meta-llama-3-8b
MAX_GEN_REASONING=4096

is_free() {
  local g=$1 mem pids
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$mem" ] || [ "$mem" -ge 20000 ] && return 1
  pids=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && return 1
  return 0
}
first_free_gpu() { for g in 0 1 2 3; do if is_free $g; then echo "$g"; return 0; fi; done; return 1; }

CALIB=logs/calib/smoothkv_${MODEL_SHORT}_perc.pt
if [ ! -f "$CALIB" ]; then
  echo "[pX-llama3] Step 1: SmoothKV calibration for $MODEL_SHORT"
  while true; do gpu=$(first_free_gpu) && break; sleep 60; done
  bash scripts/launch_wave.sh "pX_l3_calib" "$gpu" \
    "$MODERN run_smoothkv_calibrate.py --model_path $MODEL --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output $CALIB"
  # Wait for calib
  while true; do
    if tmux capture-pane -t kivi:$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' | awk '$2=="pX_l3_calib"{print $1}') -p -S -50 2>/dev/null | grep -qE "^(DONE|FAIL)_pX_l3_calib"; then break; fi
    sleep 60
  done
  if [ ! -f "$CALIB" ]; then
    echo "[pX-llama3] ABORT: calibration did not produce $CALIB"
    exit 1
  fi
fi

# Step 2: Generate 5 pair-mergeable variants
echo "[pX-llama3] Step 2: generating variants"
/usr/bin/python scripts/make_alpha_variants.py --base "$CALIB" --alphas 0.75 --pair_max_k 2>&1 | tail -3
/usr/bin/python scripts/make_percentile_variants.py --base "$CALIB" --pk 90 95 99 99.9 --symmetric 2>&1 | tail -6
A075=logs/calib/smoothkv_${MODEL_SHORT}_a0.75_pair.pt
P90=logs/calib/smoothkv_${MODEL_SHORT}_pairK90_pV90.pt
P95=logs/calib/smoothkv_${MODEL_SHORT}_pairK95_pV95.pt
P99=logs/calib/smoothkv_${MODEL_SHORT}_pairK99_pV99.pt
P999=logs/calib/smoothkv_${MODEL_SHORT}_pairK99p9_pV99p9.pt

# Step 3: Launch 9 evaluation streams
BS=16
cmd_task() {
  local method_args=$1 task=$2 extra=""
  # Only pass --max_gen_toks for reasoning tasks
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
  "pX_l3_fp16|--model fp16 --k_bits 16 --v_bits 16"
  "pX_l3_kivi2|--model kivi --k_bits 2 --v_bits 2 --group_size 32 --residual 128"
  "pX_l3_pertoken|--model pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual 0"
  "pX_l3_fp8|--model fp8 --group_size 128"
  "pX_l3_a075pair|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $A075"
  "pX_l3_pairK90|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P90"
  "pX_l3_pairK95|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P95"
  "pX_l3_pairK99|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P99"
  "pX_l3_pairK99p9|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $P999"
)

for entry in "${STREAMS[@]}"; do
  IFS='|' read -r name args <<< "$entry"
  while true; do gpu=$(first_free_gpu) && break; sleep 60; done
  cmd=$(chain_stream "$args")
  echo "[pX-llama3] launching $name on GPU$gpu"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  sleep 20
done
echo "[pX-llama3] all 9 Llama-3-8B streams launched."
