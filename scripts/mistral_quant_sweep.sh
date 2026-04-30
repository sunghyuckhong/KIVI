#!/bin/bash
# Mistral-7B-Instruct-v0.2 KV-quant sweep, 4 variants × 3 tasks.
#
# Chat-template policy (per 2026-04-29 user direction):
#   ALL three tasks: NO chat template (matches Llama-3 policy).
#   A/B numbers: chat marginally helps on gpqa (~3pp), neutral or hurts
#   elsewhere. User chose uniform no-chat across both non-thinking instruct
#   models for cross-model comparability.
#
# (ifeval deferred for Mistral — see scripts/llama3_quant_sweep.sh for the
# Llama-3 ifeval policy if you want to mirror it later.)
#
# Cudagraph: NO_ENFORCE_EAGER=1 forces vllm 0.19.x to use cudagraphs for
# fp8/pertoken/smoothkv (~10-20× speedup vs eager). Requires vllm 0.19.1
# in /opt/vllm_qwen3_env.
#
# Generation: MG=4096 (per the user's MG = model_max_length / 2 rule, since
# Mistral-Instruct ctx=32k is large enough that we don't need full 16k MG for
# these tasks; matches Llama-3 setup for cross-model comparability),
# max_model_len=6912, max_num_seqs=64, batch_size=64.
#
# Calib: logs/calib/smoothkv_mistral-7b-instruct-v0.2_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
# (built via run_smoothkv_calibrate.py + make_alpha_variants.py --half_pair_max_k)
#
# Usage:
#   bash scripts/mistral_quant_sweep.sh   # 12 cells, allocates idle GPUs as available
set -u
REPO=/home/home-mcl/sunghyuck/kv_cache_compression/KIVI
cd "$REPO"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

PY=/opt/vllm_qwen3_env/bin/python
MODEL=mistralai/Mistral-7B-Instruct-v0.2
CALIB=logs/calib/smoothkv_mistral-7b-instruct-v0.2_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
MG=4096
MML=6912

mkdir -p logs/run_out

variant_args() {
  case "$1" in
    bf16)     echo "--model bf16" ;;
    fp8)      echo "--model fp8 --bits 4 --group_size 128" ;;
    pertoken) echo "--model pertoken --bits 4 --group_size 128" ;;
    smoothkv) echo "--model smoothkv --bits 4 --group_size 128 --calib_path $CALIB" ;;
  esac
}

wait_for_idle_gpu() {
  while true; do
    while read -r idx mem; do
      if [ "${mem%MiB}" -lt 500 ]; then echo "$idx"; return; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -8)
    sleep 5
  done
}

launch() {
  local gpu=$1 variant=$2 task=$3
  local va; va=$(variant_args "$variant")
  local log=logs/run_out/mistral_${variant}_${task}_g${gpu}.log
  PATH=/usr/bin:/bin:$PATH NO_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=$gpu \
    $PY run_eval_vllm.py $va \
      --model_path $MODEL --task $task \
      --max_gen_toks $MG --max_model_len $MML \
      --max_num_seqs 64 --batch_size 64 --tp 1 --log_samples \
      > "$log" 2>&1 &
  echo "  GPU $gpu  mistral $variant  $task  chat=OFF  pid=$!"
}

for variant in bf16 fp8 pertoken smoothkv; do
  for task in gsm8k_32k math500_32k gpqa_main_cot_n_shot_32k; do
    gpu=$(wait_for_idle_gpu)
    launch "$gpu" "$variant" "$task"
    sleep 8
  done
done

wait
echo "[$(date)] Mistral KV-quant sweep complete"
