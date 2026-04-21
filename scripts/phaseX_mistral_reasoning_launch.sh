#!/bin/bash
# Re-run Mistral-7B-v0.1 on 3 reasoning tasks (gsm8k_32k, gpqa_diamond_cot_n_shot_32k,
# math500_32k) at max_gen_toks=16384 (half of Mistral's 32k context).
# TQA+CoQA skipped (already saved from Phase 2 at 256 which is correct).
# Chain uses && so a failure in task A aborts tasks B, C (no fake DONE markers).
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
FAST=/opt/paper_env_fast/bin/python
MODEL=mistralai/Mistral-7B-v0.1
BENCH="gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k"
MAX_GEN_TOKS=16384

# bs tiers: FP16 tighter (bs=8), quantized roomier (bs=16)
BS_FP16=8
BS_QUANT=16

harness() {
  local args=$1 task=$2 bs=$3
  echo "$FAST run_lm_eval_harness.py --model_name_or_path $MODEL $args --tasks $task --batch_size $bs --max_gen_toks $MAX_GEN_TOKS"
}
chain() {
  local args=$1 bs=$2
  local parts=() t
  for t in $BENCH; do parts+=("$(harness "$args" "$t" "$bs")"); done
  local out=""
  for c in "${parts[@]}"; do out="$out $c && "; done
  echo "${out% && }"
}

# 11 streams: 6 historical + 5 pair-mergeable (Mistral pair calibs already generated)
STREAMS=(
  "pX_m1_fp16|--k_bits 16 --v_bits 16|$BS_FP16"
  "pX_m1_kivi2_g32r128|--k_bits 2 --v_bits 2 --group_size 32 --residual_length 128|$BS_QUANT"
  "pX_m1_kivi4_g32r128|--k_bits 4 --v_bits 4 --group_size 32 --residual_length 128|$BS_QUANT"
  "pX_m1_kivi4_g128r128|--k_bits 4 --v_bits 4 --group_size 128 --residual_length 128|$BS_QUANT"
  "pX_m1_pertoken|--method pertoken --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BS_QUANT"
  "pX_m1_fp8|--method fp8 --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0|$BS_QUANT"
  "pX_m1_a075pair|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_mistral-7b-v0.1_a0.75_pair.pt|$BS_QUANT"
  "pX_m1_pairK90|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_mistral-7b-v0.1_pairK90_pV90.pt|$BS_QUANT"
  "pX_m1_pairK95|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_mistral-7b-v0.1_pairK95_pV95.pt|$BS_QUANT"
  "pX_m1_pairK99|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_mistral-7b-v0.1_pairK99_pV99.pt|$BS_QUANT"
  "pX_m1_pairK99p9|--method smoothkv --k_bits 4 --v_bits 4 --group_size 128 --residual_length 0 --calib_path logs/calib/smoothkv_mistral-7b-v0.1_pairK99p9_pV99p9.pt|$BS_QUANT"
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
  IFS='|' read -r name args bs <<< "$entry"
  while true; do gpu=$(first_free_gpu) && break; sleep 60; done
  cmd=$(chain "$args" "$bs")
  echo "[pX-mistral] launching $name on GPU$gpu (bs=$bs, gen=$MAX_GEN_TOKS)"
  bash scripts/launch_wave.sh "$name" "$gpu" "$cmd"
  sleep 20
done
echo "[pX-mistral] all 11 Mistral streams launched."
