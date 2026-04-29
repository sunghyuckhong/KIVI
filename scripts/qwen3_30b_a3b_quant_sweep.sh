#!/bin/bash
# Qwen3-30B-A3B KV-quant sweep, 4 variants × 3 tasks = 12 cells.
# 8 GPUs / TP=2 = 4 pairs, runs 4 cells in parallel; 3 waves.
#
# Path B fusion: Qwen3 has q_norm/k_norm RMSNorm pre-RoPE. SmoothKV must use
# head-uniform s_K (folded into qk_norm.gamma). Calib:
#   logs/calib/smoothkv_qwen3-30b-a3b_perc_ns512_puremax_a1_huk_halfpair.pt
#
# Chat-template policy: ALL three tasks use --apply_chat_template (Qwen3 is
# heavily chat-tuned, runs in thinking mode by default; without chat template
# it would lose its instruction-following entirely).
#
# Script-level mechanism #2 (vllm_custom/patches_qwen3.py via run_eval_vllm_qwen3.py).
# The kivi_vllm_plugin path is silent-no-op in vllm V1 — confirmed earlier.
#
# Usage (overnight, persistent):
#   nohup bash scripts/qwen3_30b_a3b_quant_sweep.sh > logs/run_out/qwen3_30b_a3b_overnight.log 2>&1 &
#   disown
set -u
REPO=/home/home-mcl/sunghyuck/kv_cache_compression/KIVI
cd "$REPO"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

PY=/opt/vllm_qwen3_env/bin/python
SCRIPT=run_eval_vllm_qwen3.py
MODEL=Qwen/Qwen3-30B-A3B
CALIB=logs/calib/smoothkv_qwen3-30b-a3b_perc_ns512_puremax_a1_huk_halfpair.pt
MG=8192
MML=11648
MAX_NS=24
BS=24

mkdir -p logs/run_out

variant_args() {
  case "$1" in
    bf16)     echo "--model bf16" ;;
    fp8)      echo "--model fp8 --bits 4 --group_size 128" ;;
    pertoken) echo "--model pertoken --bits 4 --group_size 128" ;;
    smoothkv) echo "--model smoothkv --bits 4 --group_size 128 --calib_path $CALIB" ;;
  esac
}

launch() {
  local pair=$1 variant=$2 task=$3
  local va; va=$(variant_args "$variant")
  local g1=${pair%,*} g2=${pair#*,}
  local log=logs/run_out/qwen3_30b_a3b_${variant}_${task}_g${g1}${g2}.log
  PATH=/usr/bin:/bin:$PATH NO_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=$pair \
    $PY $SCRIPT $va \
      --model_path $MODEL --tp 2 --task $task --apply_chat_template \
      --max_gen_toks $MG --max_model_len $MML \
      --max_num_seqs $MAX_NS --batch_size $BS --log_samples \
      > "$log" 2>&1
}

PAIRS=("0,1" "2,3" "4,5" "6,7")
declare -a CELLS

# Build the cell list (12 cells = 4 variants × 3 tasks)
i=0
for variant in bf16 fp8 pertoken smoothkv; do
  for task in gsm8k_32k math500_32k gpqa_main_cot_n_shot_32k; do
    CELLS[i]="$variant:$task"
    i=$((i+1))
  done
done

echo "[$(date)] Qwen3-30B-A3B sweep start: 12 cells, 4 GPU pairs, 3 waves"

for wave_start in 0 4 8; do
  echo "[$(date)] === wave $((wave_start/4 + 1)): cells $wave_start-$((wave_start+3)) ==="
  pids=()
  for offset in 0 1 2 3; do
    idx=$((wave_start + offset))
    [ $idx -ge ${#CELLS[@]} ] && break
    cell="${CELLS[$idx]}"
    variant="${cell%:*}"; task="${cell#*:}"
    pair="${PAIRS[$offset]}"
    echo "  GPU $pair  $variant  $task"
    launch "$pair" "$variant" "$task" &
    pids+=($!)
    sleep 8
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "[$(date)] === wave $((wave_start/4 + 1)) complete ==="
done

echo "[$(date)] Qwen3-30B-A3B sweep complete (all 12 cells)"
