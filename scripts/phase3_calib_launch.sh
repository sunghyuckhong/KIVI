#!/bin/bash
# Phase 3 calib: parallel SmoothKV calibration for Llama-3-8B base + DeepSeek-R1-Distill-Llama-8B.
# Uses modern_env (transformers 4.51.3) since paper_env_fast's 4.36.2 predates Llama-3.
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

wait_for_two_free() {
  local free=()
  while [ ${#free[@]} -lt 2 ]; do
    free=()
    for g in 0 1 2 3; do
      if is_free $g; then free+=("$g"); fi
    done
    [ ${#free[@]} -ge 2 ] && break
    sleep 60
  done
  echo "${free[0]} ${free[1]}"
}

read gpu_a gpu_b <<< "$(wait_for_two_free)"
echo "[phase3-calib] using GPU $gpu_a for Llama-3-8B base, GPU $gpu_b for R1-Distill-Llama-8B"

CALIB_CMD_L3="$MODERN run_smoothkv_calibrate.py --model_path meta-llama/Meta-Llama-3-8B --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output logs/calib/smoothkv_meta-llama-3-8b_perc.pt"
CALIB_CMD_R1="$MODERN run_smoothkv_calibrate.py --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --alpha 0.5 --beta 0.5 --samples_per_channel 10000 --output logs/calib/smoothkv_deepseek-r1-distill-llama-8b_perc.pt"

bash scripts/launch_wave.sh "calib_llama3_8b"    "$gpu_a" "$CALIB_CMD_L3"
sleep 15
bash scripts/launch_wave.sh "calib_r1d_llama8b"  "$gpu_b" "$CALIB_CMD_R1"
echo "[phase3-calib] both calibration streams launched"
