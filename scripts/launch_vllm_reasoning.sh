#!/bin/bash
# Optimized vLLM launcher for reasoning-task evaluation of KV-cache-quantized
# Llama-family models (gsm8k / gpqa / math500).
#
# USAGE:
#   launch_vllm_reasoning.sh <gpu> <preset> <stream_name> "<method_args>" "<out_stem>"
#
# PRESETS (see PRESET table below): llama3-instruct, dsr1-llama-8b, qwen25-7b (etc.)
#
# ─────────────────────────────────────────────────────────────────────────────
# Why each setting matters
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. max_gen_toks rule (user-defined):
#      if model_max_length > 32768:  max_new_tokens = 32768
#      else:                         max_new_tokens = model_max_length / 2
#
# 2. max_model_len = max_prompt_len + max_gen_toks
#    Do NOT use the model's native context (e.g. 131072 for Llama-3.1 /
#    DSR1-Distill). That wastes KV-cache budget and forces preemption. Measure
#    actual prompt lengths with scripts/measure_prompt_lens.py and size
#    max_model_len exactly (plus a small buffer and block-alignment round-up).
#
# 3. max_num_seqs ceiling = total_kv_tokens / max_model_len
#    Where total_kv_tokens = (gpu_mem * gpu_util - model_weights - graphs) / 128KB.
#    For 80 GB GPU, gpu_util=0.70, Llama-8B GQA (128 KB per cached token):
#       KV budget ≈ 55 GB - 16 GB (weights) - 3 GB (graphs) = 36 GB
#       total_kv_tokens ≈ 36 GB / 128 KB ≈ 280,000
#    Exceed the ceiling → vLLM preempts, swaps, recomputes → 3-5× throughput loss.
#
# 4. batch_size MUST equal max_num_seqs. lm_eval's vLLM wrapper submits
#    `batch_size` prompts per generate() call — with a small batch_size, the
#    vLLM concurrency slots sit empty, and effective throughput collapses.
#
# 5. enable_prefix_caching=True — fewshot tasks share a long prefix (1-3k tokens
#    of examples before the test question). Caching it gives 1.5-2× speedup.
#
# ─────────────────────────────────────────────────────────────────────────────
# PRESET TABLE (measured on 80 GB GPU with vLLM 0.6.6, gpu_util=0.70)
# ─────────────────────────────────────────────────────────────────────────────
# Preset            | model                                 | ctx native | MG    | max_model_len | max_num_seqs | batch_size
# llama3-instruct   | meta-llama/Meta-Llama-3-8B-Instruct   | 8192       | 4096  | 8192          | 32           | 32
# mistral-instruct  | mistralai/Mistral-7B-Instruct-v0.2    | 32768      | 16384 | 19456         | 14           | 14
# dsr1-llama-8b     | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 131072  | 32768 | 36864         | 7            | 7
#
# Prompt-length source-of-truth: scripts/measure_prompt_lens.py
# (gsm8k max 1404, gpqa max 2798, math500 max 1373 — all under 3k).
# max_model_len = max_prompt (2798) + max_gen_toks, rounded up to a 256-multiple.

set -u
cd "$(dirname "$0")/.."
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

GPU=$1
PRESET=$2
STREAM=$3
ARGS=$4
OUT_STEM=$5
TASKS="${TASKS:-gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k}"

# Preset → (MODEL, MG, MAX_LEN, MAX_NS)
case "$PRESET" in
  llama3-instruct)
    MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    MG=4096
    MAX_LEN=8192
    MAX_NS=32
    ;;
  dsr1-llama-8b)
    MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    MG=32768
    MAX_LEN=36864   # max prompt 2798 + 32768 gen + buffer, rounded to 144*256
    MAX_NS=7        # 280k token slots / 36864 ≈ 7.6 → 7 safe
    ;;
  mistral-instruct)
    MODEL=mistralai/Mistral-7B-Instruct-v0.2
    MG=16384        # ctx 32768, ≤ 32768 rule → MG = ctx/2
    MAX_LEN=19456   # max prompt 2798 + 16384 gen + buffer, rounded to 76*256
    MAX_NS=14       # 280k / 19456 ≈ 14.4 → 14 safe
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Available: llama3-instruct, mistral-instruct, dsr1-llama-8b" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=$GPU
VLLM=/opt/vllm_env/bin/python
LOG=logs/run_out/${STREAM}_vllm.log
BS=$MAX_NS  # batch_size = max_num_seqs (see note 4 above)

{
  echo "=== $STREAM @ $(date) on GPU$GPU (preset=$PRESET) ==="
  echo "  MODEL=$MODEL"
  echo "  max_model_len=$MAX_LEN  max_num_seqs=$MAX_NS  batch_size=$BS  max_gen_toks=$MG"
  echo "  method_args: $ARGS"
  rc=0
  for t in $TASKS; do
    out="logs/${t}_${OUT_STEM}_vllm_results.json"
    if [ -f "$out" ]; then
      echo "--- SKIP $t: $out exists ---"
      continue
    fi
    echo "--- RUN $t ---"
    $VLLM run_eval_vllm.py --model_path "$MODEL" $ARGS \
        --task $t --max_gen_toks $MG --batch_size $BS \
        --max_model_len $MAX_LEN --max_num_seqs $MAX_NS
    rc=$?
    [ $rc -ne 0 ] && break
  done
  echo "=== $STREAM @ $(date) exit=$rc ==="
  if [ $rc -eq 0 ]; then echo "DONE_${STREAM}_vllm"; else echo "FAIL_${STREAM}_vllm (rc=$rc)"; fi
} > "$LOG" 2>&1
