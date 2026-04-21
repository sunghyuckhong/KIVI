#!/bin/bash
# Phase 2: Mistral-7B-v0.1 pair-mergeable sweep — 5 calibs × 5 tasks at bs=16 paper_env_fast.
# Waits for free GPU (up to 4 at a time), launches remaining on GPUs as they free up.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
FAST=/opt/paper_env_fast/bin/python
MODEL=mistralai/Mistral-7B-v0.1
BENCH="truthfulqa_gen coqa gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k"
BS=16

cmd_for() {
  local calib=$1 task=$2
  echo "$FAST run_lm_eval_harness.py --model_name_or_path $MODEL --method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --tasks $task --batch_size $BS --calib_path $calib"
}
chain_stream() {
  local calib=$1 parts=() t out=""
  for t in $BENCH; do parts+=("$(cmd_for "$calib" "$t")"); done
  for c in "${parts[@]}"; do out="$out $c ; "; done
  echo "${out% ; }"
}

declare -a STREAMS=(
  "pPfpair_m1_pairK90:logs/calib/smoothkv_mistral-7b-v0.1_pairK90_pV90.pt"
  "pPfpair_m1_pairK95:logs/calib/smoothkv_mistral-7b-v0.1_pairK95_pV95.pt"
  "pPfpair_m1_pairK99:logs/calib/smoothkv_mistral-7b-v0.1_pairK99_pV99.pt"
  "pPfpair_m1_pairK99p9:logs/calib/smoothkv_mistral-7b-v0.1_pairK99p9_pV99p9.pt"
  "pPfpair_m1_a075pair:logs/calib/smoothkv_mistral-7b-v0.1_a0.75_pair.pt"
)

is_free() {
  local g=$1 mem pids
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$mem" ] || [ "$mem" -ge 20000 ] && return 1
  pids=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && return 1
  return 0
}
first_free_gpu() {
  for g in 0 1 2 3; do
    if is_free $g; then echo "$g"; return 0; fi
  done
  return 1
}

for entry in "${STREAMS[@]}"; do
  name="${entry%%:*}"; calib="${entry#*:}"
  if [ ! -f "$calib" ]; then
    echo "[pPfpair-m1] MISSING calib: $calib — skipping $name"
    continue
  fi
  while true; do
    gpu=$(first_free_gpu) && break
    sleep 60
  done
  cmd=$(chain_stream "$calib")
  echo "[pPfpair-m1] launching $name on GPU$gpu"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  sleep 20
done
echo "[pPfpair-m1] all 5 Mistral pair-mergeable streams launched."
