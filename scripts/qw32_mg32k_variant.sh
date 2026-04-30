#!/bin/bash
# Run one variant of Qwen3-32B at MG=32k single-pass through 3 tasks sequentially.
# Args:
#   $1 = GPU pair (e.g. "0,1")
#   $2 = method args (e.g. "--model bf16" or "--model pertoken --bits 4 --group_size 128")
#   $3 = mtag (output filename method tag, e.g. "bf16" or "pertoken_int4_g128")
#   $4 = label (used for log filename, e.g. "qw32m_bf16")
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_TOKEN=$(/bin/cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=$1
cd /workspace/KIVI
PY=/opt/vllm_exaone_v2_env/bin/python3

METHOD_ARGS=$2
MTAG=$3
LABEL=$4

declare -A MML
MML[gsm8k_32k]=34304
MML[minerva_math500]=34304
MML[gpqa_main_cot_n_shot_32k]=35584

for task in gsm8k_32k minerva_math500 gpqa_main_cot_n_shot_32k; do
  result=logs/${task}_qwen3-32b_${MTAG}_chat_vllm_results.json
  if [ -f "$result" ]; then
    echo "[$LABEL] SKIP $task — $result exists"
    continue
  fi
  echo "[$LABEL] === START $task at $(/bin/date) ==="
  $PY run_eval_vllm.py $METHOD_ARGS \
      --model_path Qwen/Qwen3-32B \
      --task "$task" --apply_chat_template \
      --max_gen_toks 32768 --max_model_len ${MML[$task]} \
      --max_num_seqs 8 --batch_size 8 --tp 2 \
      --log_samples \
      2>&1 | /usr/bin/tee -a logs/run_out/${LABEL}_${task}.log
  echo "[$LABEL] === DONE $task at $(/bin/date) ==="
done
echo "[$LABEL] === ALL TASKS DONE at $(/bin/date) ==="
