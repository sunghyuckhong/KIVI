#!/bin/bash
# One-shot launcher for 4 Llama-2 pair-max SmoothKV streams at bs=16 in paper_env_fast.
# 3 percentile pair-K variants + α=0.75 pair-K variant. Each stream runs 5 benchmarks.
# Uses GPUs 0..3 (one per stream).
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
FAST=/opt/paper_env_fast/bin/python
MODEL=meta-llama/Llama-2-7b-hf
BENCH="truthfulqa_gen coqa gsm8k gpqa_diamond_cot_n_shot math500"
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

declare -A JOBS=(
  [pPfpair_l2_pairK95]="logs/calib/smoothkv_llama-2-7b-hf_pairK95_pV95.pt"
  [pPfpair_l2_pairK99]="logs/calib/smoothkv_llama-2-7b-hf_pairK99_pV99.pt"
  [pPfpair_l2_pairK99p9]="logs/calib/smoothkv_llama-2-7b-hf_pairK99p9_pV99p9.pt"
  [pPfpair_l2_a075pair]="logs/calib/smoothkv_llama-2-7b-hf_a0.75_pair.pt"
)

gpu=0
for name in pPfpair_l2_pairK95 pPfpair_l2_pairK99 pPfpair_l2_pairK99p9 pPfpair_l2_a075pair; do
  calib=${JOBS[$name]}
  if [ ! -f "$calib" ]; then
    echo "[pPfpair-l2] MISSING calib: $calib — skipping $name"
    continue
  fi
  cmd=$(chain_stream "$calib")
  echo "[pPfpair-l2] launching $name on GPU$gpu  (calib=$calib)"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  gpu=$((gpu+1))
done
echo "[pPfpair-l2] 4 streams launched. tail -f logs/run_out/pPfpair_l2_*.log to watch."
