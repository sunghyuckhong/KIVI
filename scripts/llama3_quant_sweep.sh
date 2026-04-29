#!/bin/bash
# Llama-3-8B-Instruct KV-quant sweep, 4 variants × 4 tasks.
#
# Chat-template policy (per 2026-04-29 user direction):
#   gsm8k_32k / math500_32k / gpqa_main_cot_n_shot_32k : NO chat template
#     Verified A/B: gsm8k strict no-chat=0.7589 vs chat=0.6513 (+10.8 pp);
#     math500 no-chat=0.288 vs chat=0.036 (Minerva scorer break);
#     gpqa flex no-chat=0.190 vs chat=0.266 (chat slightly better here, but
#     user opted for uniform no-chat-everywhere for the reasoning tasks).
#   ifeval                                              : --apply_chat_template
#     Instruction-following eval; chat template required for instruct models.
#
# Cudagraph: NO_ENFORCE_EAGER=1 forces vllm 0.19.x to use cudagraphs for
# fp8/pertoken/smoothkv (~10-20× speedup vs eager). Requires vllm 0.19.1
# in /opt/vllm_qwen3_env.
#
# Generation: MG=4096 (model_max_length 8k → MG=ctx/2 rule), max_model_len=6912,
# max_num_seqs=64, batch_size=64.
#
# Calib: logs/calib/smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
# (built via run_smoothkv_calibrate.py + make_alpha_variants.py --half_pair_max_k)
#
# Usage:
#   bash scripts/llama3_quant_sweep.sh   # 16 cells (4 variants × 4 tasks),
#                                          allocates idle GPUs as available
set -u
REPO=/home/home-mcl/sunghyuck/kv_cache_compression/KIVI
cd "$REPO"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

PY=/opt/vllm_qwen3_env/bin/python
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
CALIB=logs/calib/smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
MG=4096
MML=6912

mkdir -p logs/run_out

chat_for() {
  case "$1" in
    ifeval) echo "--apply_chat_template" ;;
    *)      echo "" ;;
  esac
}

# Per-task generation budget. ifeval responses are short; others use the rule MG.
mg_for()  { case "$1" in ifeval) echo 1280 ;; *) echo "$MG"  ;; esac; }
mml_for() { case "$1" in ifeval) echo 2048 ;; *) echo "$MML" ;; esac; }

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
  local cf; cf=$(chat_for "$task")
  local mg; mg=$(mg_for "$task")
  local mml; mml=$(mml_for "$task")
  local log=logs/run_out/llama3_${variant}_${task}_g${gpu}.log
  PATH=/usr/bin:/bin:$PATH NO_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=$gpu \
    $PY run_eval_vllm.py $va \
      --model_path $MODEL --task $task $cf \
      --max_gen_toks $mg --max_model_len $mml \
      --max_num_seqs 64 --batch_size 64 --tp 1 --log_samples \
      > "$log" 2>&1 &
  echo "  GPU $gpu  llama3  $variant  $task  chat=${cf:-OFF}  MG=$mg  pid=$!"
}

for variant in bf16 fp8 pertoken smoothkv; do
  for task in gsm8k_32k math500_32k gpqa_main_cot_n_shot_32k ifeval; do
    gpu=$(wait_for_idle_gpu)
    launch "$gpu" "$variant" "$task"
    sleep 8   # stagger so wait_for_idle_gpu sees the new allocation
  done
done

wait
echo "[$(date)] Llama-3 KV-quant sweep complete"
