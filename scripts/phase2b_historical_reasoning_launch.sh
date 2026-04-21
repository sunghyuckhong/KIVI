#!/bin/bash
# Phase 2b: Historical re-run of Llama-2 + Mistral on 3 reasoning tasks at 32k settings.
# Skips pair-mergeable Mistral (already in Phase 2) and skips TQA+CoQA (non-reasoning, stay at 256).
# Uses paper_env_fast at bs=16; polls for free GPUs and launches streams as they free up.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
FAST=/opt/paper_env_fast/bin/python

# Reasoning-only benchmark set for the re-run; non-reasoning TQA/CoQA stay at their old results
BENCH_REASONING="gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k"
# Llama-2 pairK90 is new (never ran) — give it the full 5-task set
BENCH_FULL="truthfulqa_gen coqa gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k"
BS=16

L2=meta-llama/Llama-2-7b-hf
M1=mistralai/Mistral-7B-v0.1

# build one harness invocation
harness_cmd() {
  # harness_cmd <model> <args_suffix> <task>
  echo "$FAST run_lm_eval_harness.py --model_name_or_path $1 $2 --tasks $3 --batch_size $BS"
}
# chain a list of tasks
chain_stream() {
  local model=$1; local args=$2; local tasks=$3
  local parts=(); local t
  for t in $tasks; do parts+=("$(harness_cmd "$model" "$args" "$t")"); done
  local out=""
  for c in "${parts[@]}"; do out="$out $c ; "; done
  echo "${out% ; }"
}

# Stream specs — one per row: name|model|args|bench
# Bench = reasoning-only (3 tasks) except pairK90 which needs full 5 (never ran before).
STREAMS=(
  # ----- Llama-2 historical (6 streams × 3 reasoning tasks) -----
  "p2b_l2_fp16|$L2|--k_bits 16 --v_bits 16|$BENCH_REASONING"
  "p2b_l2_kivi2_g32r128|$L2|--k_bits 2 --v_bits 2 --group_size 32 --residual_length 128|$BENCH_REASONING"
  "p2b_l2_kivi4_g32r128|$L2|--k_bits 4 --v_bits 4 --group_size 32 --residual_length 128|$BENCH_REASONING"
  "p2b_l2_kivi4_g128r128|$L2|--k_bits 4 --v_bits 4 --group_size 128 --residual_length 128|$BENCH_REASONING"
  "p2b_l2_pertoken|$L2|--method pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BENCH_REASONING"
  "p2b_l2_fp8|$L2|--method fp8 --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BENCH_REASONING"

  # ----- Llama-2 pair-mergeable redo (4 streams × 3 reasoning tasks) + pairK90 new (5 tasks) -----
  "p2b_l2_a075pair|$L2|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_llama-2-7b-hf_a0.75_pair.pt|$BENCH_REASONING"
  "p2b_l2_pairK90|$L2|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_llama-2-7b-hf_pairK90_pV90.pt|$BENCH_FULL"
  "p2b_l2_pairK95|$L2|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_llama-2-7b-hf_pairK95_pV95.pt|$BENCH_REASONING"
  "p2b_l2_pairK99|$L2|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_llama-2-7b-hf_pairK99_pV99.pt|$BENCH_REASONING"
  "p2b_l2_pairK99p9|$L2|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_llama-2-7b-hf_pairK99p9_pV99p9.pt|$BENCH_REASONING"

  # ----- Mistral historical (6 streams × 3 reasoning tasks) — Mistral pair-mergeable already in Phase 2 -----
  "p2b_m1_fp16|$M1|--k_bits 16 --v_bits 16|$BENCH_REASONING"
  "p2b_m1_kivi2_g32r128|$M1|--k_bits 2 --v_bits 2 --group_size 32 --residual_length 128|$BENCH_REASONING"
  "p2b_m1_kivi4_g32r128|$M1|--k_bits 4 --v_bits 4 --group_size 32 --residual_length 128|$BENCH_REASONING"
  "p2b_m1_kivi4_g128r128|$M1|--k_bits 4 --v_bits 4 --group_size 128 --residual_length 128|$BENCH_REASONING"
  "p2b_m1_pertoken|$M1|--method pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BENCH_REASONING"
  "p2b_m1_fp8|$M1|--method fp8 --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BENCH_REASONING"
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
  for g in 0 1 2 3; do if is_free $g; then echo "$g"; return 0; fi; done
  return 1
}

for entry in "${STREAMS[@]}"; do
  IFS='|' read -r name model args tasks <<< "$entry"
  while true; do
    gpu=$(first_free_gpu) && break
    sleep 60
  done
  cmd=$(chain_stream "$model" "$args" "$tasks")
  echo "[p2b] launching $name on GPU$gpu"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  sleep 20
done
echo "[p2b] all 17 Phase-2b streams launched."
