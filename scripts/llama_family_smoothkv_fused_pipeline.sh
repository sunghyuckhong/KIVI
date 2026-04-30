#!/bin/bash
# 3-pass adaptive pipeline for zero-runtime SmoothKV (smoothkv_fused) across
# the Llama family (Path A — direct qkv_proj row scaling) and the Qwen3 family
# (Path B — q_norm/k_norm gamma fold, head-uniform calib).
#
#   Pass 1: lm_eval --log_samples at reduced MG via kivi_plugin (smoothkv_fused)
#           → JSONL samples
#   Pass 2: convert JSONL → {task: items} JSON, then adaptive_rerun.py at full
#           MG on items that hit the cap (also goes through the plugin since
#           the per-GPU config persists on /tmp)
#   Pass 3: merge_rerun.py — re-applies lm_eval's filter chain on the merged
#           samples and reports the corrected score
#
# Usage: llama_family_smoothkv_fused_pipeline.sh <model_short> <gpu_or_pair> <calib_path>
#   model_short ∈ {llama3-8b-instruct, mistral-7b-instruct-v0.2,
#                  dsr1-distill-llama-8b, qwen3-32b, qwen3-30b-a3b}
#   gpu_or_pair: single index for TP=1 (e.g. "0"), comma-sep pair for TP=2
#                ("2,3"). The script writes the kivi config to
#                /tmp/kivi_active_${gpu_or_pair}.json.
#
# All three passes run on the SAME GPU(s) sequentially per task. Multiple tasks
# for one model are sequential too — launch separate invocations for parallel
# task streams across GPUs.
set -u
REPO=/home/home-mcl/sunghyuck/kv_cache_compression/KIVI
cd "$REPO"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

MODEL_SHORT=$1
GPU=$2
CALIB=$3

# Common
LM_EVAL=/opt/vllm_qwen3_env/bin/lm_eval
PY=/opt/vllm_qwen3_env/bin/python    # both lm_eval (pass1) and adaptive_rerun (pass2) use vllm_qwen3_env so the kivi plugin fires
SKENV=/home/home-mcl/sunghyuck/kv_cache_compression/SnapKV/snapkv_test_env/bin/python  # merge_rerun (CPU + lm_eval filter chain)
ADAPT=scripts/adaptive_rerun.py
MERGE=scripts/merge_rerun.py
LABEL=smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair

case "$MODEL_SHORT" in
  llama3-8b-instruct)
    MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    MODEL_TAG=llama3_8b_instruct
    OUT_TAG=llama3-8b-instruct
    MG_PASS1=2048
    MG_PASS2=4096       # native ctx 8k → MG = ctx/2 per the user's rule
    MAX_MODEL_LEN1=4864 # 2800 prompt + 2048 gen, rounded
    MAX_MODEL_LEN2=6912 # 2800 prompt + 4096 gen, rounded
    MAX_NS=64           # 128 OOMs during cudagraph capture; smoke worked at 64
    BS=2048
    TP=1
    ;;
  mistral-7b-instruct-v0.2)
    MODEL=mistralai/Mistral-7B-Instruct-v0.2
    MODEL_TAG=mistral_7b_instruct_v02
    OUT_TAG=mistral-7b-instruct-v0.2
    MG_PASS1=8192
    MG_PASS2=16384      # ctx 32k → MG = ctx/2 per the user's rule
    MAX_MODEL_LEN1=11264
    MAX_MODEL_LEN2=19456
    MAX_NS=24
    BS=2048
    TP=1
    ;;
  dsr1-distill-llama-8b)
    MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    MODEL_TAG=dsr1_distill_llama_8b
    OUT_TAG=deepseek-r1-distill-llama-8b
    MG_PASS1=16384
    MG_PASS2=32768      # ctx 131k → cap at 32k per the user's rule
    MAX_MODEL_LEN1=19456
    MAX_MODEL_LEN2=35840
    MAX_NS=8
    BS=2048
    TP=1
    ;;
  qwen3-32b)
    # Path B: q_norm/k_norm fusion. Calib must be `_huk_halfpair`.
    MODEL=Qwen/Qwen3-32B
    MODEL_TAG=qwen3_32b
    OUT_TAG=qwen3-32b
    MG_PASS1=8192
    MG_PASS2=32768      # ctx 40k > 32k → cap at 32k
    MAX_MODEL_LEN1=11648
    MAX_MODEL_LEN2=35840
    MAX_NS=24
    BS=2048
    TP=2
    ;;
  qwen3-30b-a3b)
    # Path B: q_norm/k_norm fusion. Calib must be `_huk_halfpair`.
    # MoE: 30B total, 3B activated — KV per request is small, more concurrency.
    MODEL=Qwen/Qwen3-30B-A3B
    MODEL_TAG=qwen3_30b_a3b
    OUT_TAG=qwen3-30b-a3b
    MG_PASS1=8192
    MG_PASS2=32768
    MAX_MODEL_LEN1=11648
    MAX_MODEL_LEN2=35840
    MAX_NS=48
    BS=2048
    TP=2
    ;;
  *) echo "unknown model_short: $MODEL_SHORT"; exit 2 ;;
esac

[[ -f "$CALIB" ]] || { echo "calib not found: $CALIB"; exit 2; }

CFG_FILE=/tmp/kivi_active_${GPU}.json
cat > "$CFG_FILE" <<EOF
{"method": "smoothkv_fused", "group_size": 128, "bits": 4, "calib_path": "$CALIB"}
EOF
echo "kivi cfg $CFG_FILE: $(cat $CFG_FILE)"

mkdir -p logs/run_out logs/llama_family_results logs/llama_family_results/mg32k

run_task_pipeline() {
  local task=$1
  local stream=pX_${MODEL_TAG}_${LABEL}_${task}
  local base_log=logs/run_out/${stream}_pipeline.log
  local outdir=logs/llama_family_results/${OUT_TAG}_${LABEL}_${task}_pass1
  local model_subdir
  local samples_jsonl
  local samples_json=logs/llama_family_results/mg32k/${OUT_TAG}_${LABEL}_${task}_pass1_samples.json
  local rerun_json=logs/llama_family_results/mg32k/${OUT_TAG}_${LABEL}_${task}_samples_rerun.json
  local merged_json=logs/llama_family_results/mg32k/${OUT_TAG}_${LABEL}_${task}_pass1_samples_merged.json

  # Chat-template policy: default ON. Disable only for Llama-3 + math500_32k.
  #
  # math500_32k uses a 4-shot Minerva self-completing prompt
  # (`Problem: ... Solution: ... Final Answer: X. I hope it is correct.`)
  # that the model is meant to *continue*. Llama-3-8B-Instruct, when given
  # this prompt wrapped in chat template, switches to "assistant" mode and
  # emits `\boxed{...}` instead of the Minerva format — which the default
  # `process_results` (Minerva-only regex) scores 0 on. With chat template
  # bf16 math500 dropped 0.284 -> 0.036, while WITHOUT chat template it
  # holds at 0.284. Mistral / Qwen3 do NOT exhibit this behavior — they
  # continue the Minerva pattern even when chat-wrapped, so we leave
  # chat-template ON for them across all tasks.
  local APPLY_CHAT="--apply_chat_template"
  if [ "$MODEL_SHORT" = "llama3-8b-instruct" ] && [ "$task" = "math500_32k" ]; then
    APPLY_CHAT=""
  fi

  mkdir -p "$outdir"
  {
    echo "============================================="
    echo "[$(date)] $stream pipeline start  GPU=$GPU"
    echo "  pass1 MG=$MG_PASS1, pass2 MG=$MG_PASS2"
    echo "============================================="

    # ── PASS 1 ─────────────────────────────────────────────────────────────
    if [ -f "$samples_json" ]; then
      echo "PASS1 SKIP: $samples_json already exists"
    else
      echo "--- PASS1: lm_eval at MG=$MG_PASS1 with --log_samples ---"
      CUDA_VISIBLE_DEVICES=$GPU "$LM_EVAL" \
          --model vllm \
          --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=${TP},gpu_memory_utilization=0.85,max_model_len=${MAX_MODEL_LEN1},max_num_seqs=${MAX_NS},enable_prefix_caching=True,enforce_eager=False" \
          --tasks "$task" \
          $APPLY_CHAT --batch_size $BS --gen_kwargs "max_gen_toks=${MG_PASS1}" \
          --log_samples --output_path "$outdir" \
          --include_path tasks
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAIL_${stream} (pass1 rc=$rc)"; return 1
      fi

      # Find the JSONL lm_eval wrote, then convert to {task: items} JSON
      samples_jsonl=$(find "$outdir" -maxdepth 3 -name "samples_${task}_*.jsonl" -print -quit 2>/dev/null)
      if [ -z "$samples_jsonl" ]; then
        echo "FAIL_${stream} (pass1: could not locate samples jsonl in $outdir)"; return 1
      fi
      echo "  found samples jsonl: $samples_jsonl"
      "$PY" -c "
import json, sys
items = []
with open('$samples_jsonl') as f:
    for line in f:
        items.append(json.loads(line))
with open('$samples_json', 'w') as f:
    json.dump({'$task': items}, f, default=str)
print(f'wrote {len(items)} items to $samples_json')
"
    fi

    # ── PASS 2 ─────────────────────────────────────────────────────────────
    if [ -f "$rerun_json" ]; then
      echo "PASS2 SKIP: $rerun_json already exists"
    else
      echo "--- PASS2: adaptive_rerun at MG=$MG_PASS2 (truncated only) ---"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ADAPT" \
          --samples "$samples_json" \
          --model_path "$MODEL" \
          --new_max_gen_toks $MG_PASS2 \
          --max_num_seqs 8 \
          --tp $TP \
          --out "$rerun_json"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAIL_${stream} (pass2 rc=$rc)"; return 1
      fi
    fi

    # ── PASS 3 ─────────────────────────────────────────────────────────────
    echo "--- PASS3: merge_rerun + filter rescore ---"
    "$SKENV" "$MERGE" \
        --original "$samples_json" \
        --rerun    "$rerun_json" \
        --task     "$task" \
        --out      "$merged_json"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "FAIL_${stream} (pass3 rc=$rc)"; return 1
    fi

    echo "============================================="
    echo "[$(date)] $stream pipeline DONE"
    echo "============================================="
    echo "DONE_${stream}"
  } > "$base_log" 2>&1
  return 0
}

overall_rc=0
for task in gsm8k_32k math500_32k gpqa_main_cot_n_shot_32k; do
  echo "[$(date)] launching task pipeline: $task"
  run_task_pipeline "$task"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] task $task FAILED (rc=$rc) — continuing to next task"
    overall_rc=$rc
  fi
done

echo "[$(date)] ${MODEL_TAG} ${LABEL} pipeline ALL TASKS DONE  overall_rc=$overall_rc"
exit $overall_rc
