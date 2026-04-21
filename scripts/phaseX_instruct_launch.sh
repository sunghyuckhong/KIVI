#!/bin/bash
# Mistral-7B-Instruct-v0.2 + Llama-3-8B-Instruct full-matrix eval.
# Step 1: parallel SmoothKV calibration (both models, 2 GPUs)
# Step 2: generate 5 pair-mergeable variants per model (α=0.75 pair, pairK90/95/99/99p9)
# Step 3: 9 methods × 3 reasoning tasks × 2 models = 18 streams
# TQA+CoQA recycled from existing modern-env runs.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
MODERN=/opt/modernenv/bin/python

is_free() {
  local g=$1 mem pids
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$mem" ] || [ "$mem" -ge 20000 ] && return 1
  pids=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && return 1
  return 0
}
first_free_gpu() { for g in 0 1 2 3; do if is_free $g; then echo "$g"; return 0; fi; done; return 1; }
wait_stream() {
  local name=$1
  while true; do
    local idx=$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' | awk -v n="$name" '$2==n{print $1; exit}')
    [ -z "$idx" ] && return 0
    tmux capture-pane -t kivi:$idx -p -S -100 2>/dev/null | grep -qE "^(DONE|FAIL)_$name" && return 0
    sleep 60
  done
}

# ---- Step 1: Parallel SmoothKV calibration ----
MI_MODEL=mistralai/Mistral-7B-Instruct-v0.2
MI_SHORT=mistral-7b-instruct-v0.2
L3I_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
L3I_SHORT=meta-llama-3-8b-instruct
MI_CALIB=logs/calib/smoothkv_${MI_SHORT}_perc.pt
L3I_CALIB=logs/calib/smoothkv_${L3I_SHORT}_perc.pt

if [ ! -f "$MI_CALIB" ] || [ ! -f "$L3I_CALIB" ]; then
  echo "[pX-instruct] Step 1: calibration"
  # Launch two calibs in parallel
  if [ ! -f "$MI_CALIB" ]; then
    while true; do gpu=$(first_free_gpu) && break; sleep 30; done
    bash scripts/launch_wave.sh "pX_calib_mi" "$gpu" \
      "$MODERN run_smoothkv_calibrate.py --model_path $MI_MODEL --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output $MI_CALIB"
    sleep 30
  fi
  if [ ! -f "$L3I_CALIB" ]; then
    while true; do gpu=$(first_free_gpu) && break; sleep 30; done
    bash scripts/launch_wave.sh "pX_calib_l3i" "$gpu" \
      "$MODERN run_smoothkv_calibrate.py --model_path $L3I_MODEL --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output $L3I_CALIB"
  fi
  [ ! -f "$MI_CALIB" ] && wait_stream pX_calib_mi
  [ ! -f "$L3I_CALIB" ] && wait_stream pX_calib_l3i
  [ ! -f "$MI_CALIB" ] && { echo "[pX-instruct] ABORT: $MI_CALIB missing after calib"; exit 1; }
  [ ! -f "$L3I_CALIB" ] && { echo "[pX-instruct] ABORT: $L3I_CALIB missing after calib"; exit 1; }
fi

# ---- Step 2: Generate pair-mergeable variants (CPU) ----
echo "[pX-instruct] Step 2: generating variants"
for short in $MI_SHORT $L3I_SHORT; do
  base_a=logs/calib/smoothkv_${short}_a0.5.pt
  perc=logs/calib/smoothkv_${short}_perc.pt
  if [ ! -f logs/calib/smoothkv_${short}_a0.75_pair.pt ]; then
    /usr/bin/python scripts/make_alpha_variants.py --base "$base_a" --alphas 0.75 --pair_max_k 2>&1 | tail -3
  fi
  if [ ! -f logs/calib/smoothkv_${short}_pairK99p9_pV99p9.pt ]; then
    /usr/bin/python scripts/make_percentile_variants.py --base "$perc" --pk 90 95 99 99.9 --symmetric 2>&1 | tail -6
  fi
done

# ---- Step 3: Launch eval streams ----
BENCH_REASONING="gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k"
BS=16

cmd_task() {
  local model_path=$1 max_gen=$2 method_args=$3 task=$4
  echo "$MODERN run_eval.py --model_path $model_path --batch_size $BS $method_args --task $task --max_gen_toks $max_gen"
}
chain_reasoning() {
  local model_path=$1 max_gen=$2 method_args=$3
  local parts=() t
  for t in $BENCH_REASONING; do parts+=("$(cmd_task "$model_path" "$max_gen" "$method_args" "$t")"); done
  local out=""
  for c in "${parts[@]}"; do out="$out $c && "; done
  echo "${out% && }"
}

# Per model: methods × stream specs
launch_model_streams() {
  local short=$1 model_path=$2 max_gen=$3
  local a075=logs/calib/smoothkv_${short}_a0.75_pair.pt
  local p90=logs/calib/smoothkv_${short}_pairK90_pV90.pt
  local p95=logs/calib/smoothkv_${short}_pairK95_pV95.pt
  local p99=logs/calib/smoothkv_${short}_pairK99_pV99.pt
  local p999=logs/calib/smoothkv_${short}_pairK99p9_pV99p9.pt
  local prefix="pX_${short:0:4}"   # e.g., pX_mist  or pX_meta
  STREAMS=(
    "${prefix}_fp16|--model fp16 --k_bits 16 --v_bits 16"
    "${prefix}_kivi2|--model kivi --k_bits 2 --v_bits 2 --group_size 32 --residual 128"
    "${prefix}_pertoken|--model pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual 0"
    "${prefix}_fp8|--model fp8 --group_size 128"
    "${prefix}_a075pair|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $a075"
    "${prefix}_pairK90|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $p90"
    "${prefix}_pairK95|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $p95"
    "${prefix}_pairK99|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $p99"
    "${prefix}_pairK99p9|--model smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual 0 --calib_path $p999"
  )
  for entry in "${STREAMS[@]}"; do
    IFS='|' read -r name args <<< "$entry"
    while true; do gpu=$(first_free_gpu) && break; sleep 60; done
    cmd=$(chain_reasoning "$model_path" "$max_gen" "$args")
    echo "[pX-instruct] launching $name on GPU$gpu (gen=$max_gen)"
    bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
    sleep 20
  done
}

launch_model_streams "$L3I_SHORT"  "$L3I_MODEL"  4096
launch_model_streams "$MI_SHORT"   "$MI_MODEL"   16384

echo "[pX-instruct] all eval streams launched."
