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
# Preset            | model                                 | ctx native | MG    | max_num_seqs | batch_size
# llama3-instruct   | meta-llama/Meta-Llama-3-8B-Instruct   | 8192       | 4096  | 128          | 128
# mistral-instruct  | mistralai/Mistral-7B-Instruct-v0.2    | 32768      | 16384 | 8            | 128
# dsr1-llama-8b     | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 131072  | 32768 | 2            | 128
#
# max_model_len is NOT set — vLLM uses native ctx. Output length is controlled via --max_gen_toks.
# max_num_seqs sized to real KV-cache ceiling at native ctx (280k token slots / native ctx).
# batch_size must be ≥ task item count so lm_eval submits the full task in one generate()
# call — under-sized batch_size drains vLLM's continuous batching between submits (20× slower).
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

# Preset → (MODEL, MG, MAX_NS)
case "$PRESET" in
  llama3-instruct)
    MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    MG=4096
    MAX_NS=128      # Llama3 ctx 8k — per-task max_len savings are tiny, stay at native
    USE_TASK_LEN=0
    ;;
  dsr1-llama-8b)
    MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    MG=32768
    MAX_NS=8        # at per-task max_model_len ≈ 34-36k: 280k / 35k ≈ 8
    USE_TASK_LEN=1
    ;;
  mistral-instruct)
    MODEL=mistralai/Mistral-7B-Instruct-v0.2
    MG=16384        # ctx 32768, rule: ≤ 32768 → MG = ctx/2
    MAX_NS=15       # at per-task max_model_len ≈ 18-19k: 280k / 19k ≈ 14.7
    USE_TASK_LEN=1
    ;;
  qwen3-8b)
    MODEL=Qwen/Qwen3-8B
    MG=32768        # ctx 40960, rule: > 32768 → cap at 32768
    MAX_NS=64       # nominal ceiling; vLLM self-throttles to KV-budget fit (~15 effective
                    # at 17920 tokens). Combined with BS=64, continuous batching refills
                    # slots from a 64-prompt pool → no tail drain.
    USE_TASK_LEN=1
    ;;
  exaone-4.5-33b)
    MODEL=LGAI-EXAONE/EXAONE-4.5-33B
    MG=32768        # ctx 262144 (256k), rule: > 32768 → cap at 32768
    MAX_NS=16       # 33B BF16 takes ~32GB per TP rank, hybrid attn (sliding=4096 on 48/64
                    # layers) keeps KV per-token small. At max_model_len=36k, KV per request
                    # ≈ 3GB → 16 concurrent slots fit comfortably in 2x80GB.
    USE_TASK_LEN=1
    TP_SIZE=2       # 33B doesn't fit on a single 80GB once we add KV + activations
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Available: llama3-instruct, mistral-instruct, dsr1-llama-8b, qwen3-8b, exaone-4.5-33b" >&2
    exit 2
    ;;
esac

# TP_SIZE may have been set by the preset; default to 1 for single-GPU presets.
TP_SIZE="${TP_SIZE:-1}"

export CUDA_VISIBLE_DEVICES=$GPU
case "$PRESET" in
  qwen3-8b)        VLLM=/opt/vllm_qwen3_env/bin/python ;;
  exaone-4.5-33b)  VLLM=/opt/vllm_exaone_v2_env/bin/python ;;
  *)               VLLM=/opt/vllm_env/bin/python ;;
esac
LOG=logs/run_out/${STREAM}_vllm.log

# Optional overrides via env vars:
#   MG_OVERRIDE=16384  ./launch_vllm_reasoning.sh …  # truncate max_gen_toks (adaptive pass)
#   MAX_NS_OVERRIDE=14 ./launch_vllm_reasoning.sh …  # bump max_num_seqs (useful when MG lowered)
#   LOG_SAMPLES=1      ./launch_vllm_reasoning.sh …  # save per-item generations
if [ -n "${MG_OVERRIDE:-}" ]; then
  echo "MG_OVERRIDE: $MG -> $MG_OVERRIDE"
  MG=$MG_OVERRIDE
fi
if [ -n "${MAX_NS_OVERRIDE:-}" ]; then
  echo "MAX_NS_OVERRIDE: $MAX_NS -> $MAX_NS_OVERRIDE"
  MAX_NS=$MAX_NS_OVERRIDE
fi
# BS must be set AFTER overrides so MAX_NS_OVERRIDE actually takes effect on
# lm_eval's batch_size (otherwise vLLM has more concurrency slots than lm_eval
# fills and the bump is a no-op).
BS=$MAX_NS  # lm_eval batch_size = vLLM max_num_seqs. Every generate() chunk
            # matches the concurrency slots exactly → tqdm advances every MAX_NS items.
EXTRA_EVAL_ARGS=""
[ "${LOG_SAMPLES:-0}" = "1" ] && EXTRA_EVAL_ARGS="--log_samples"

# Per-task max_model_len (measured via measure_prompt_lens.py):
#   Llama3 tok: gsm8k=1404, gpqa=2798, math500=1373.
#   Qwen3  tok: gsm8k=1597, gpqa=2800, math500=1433.
# Use max across tokenizers (slack is within 256-rounding anyway).
# max_model_len = max_prompt + max_gen_toks, rounded up to 256-multiple.
task_max_len() {
  local t=$1 mg=$2
  local prompt
  case "$t" in
    gsm8k_32k)                    prompt=1600 ;;
    gpqa_diamond_cot_n_shot_32k)  prompt=2800 ;;
    gpqa_main_cot_n_shot_32k)     prompt=2800 ;;
    math500_32k)                  prompt=1450 ;;
    *)                            prompt=4096 ;;  # conservative default for unknown task
  esac
  local total=$((prompt + mg))
  # round up to 256-multiple
  echo $(( (total + 255) / 256 * 256 ))
}

{
  echo "=== $STREAM @ $(date) on GPU$GPU (preset=$PRESET) ==="
  echo "  MODEL=$MODEL"
  echo "  max_num_seqs=$MAX_NS  batch_size=$BS  max_gen_toks=$MG  per-task-max-len=$USE_TASK_LEN"
  echo "  method_args: $ARGS"
  rc=0
  for t in $TASKS; do
    out="logs/${t}_${OUT_STEM}_vllm_results.json"
    if [ -f "$out" ]; then
      echo "--- SKIP $t: $out exists ---"
      continue
    fi
    echo "--- RUN $t ---"
    extra_args=()
    if [ "$USE_TASK_LEN" = "1" ]; then
      maxlen=$(task_max_len "$t" "$MG")
      echo "   max_model_len=$maxlen (per-task)"
      extra_args+=(--max_model_len "$maxlen")
    fi
    $VLLM run_eval_vllm.py --model_path "$MODEL" $ARGS \
        --task $t --max_gen_toks $MG --batch_size $BS \
        --max_num_seqs $MAX_NS --tp $TP_SIZE $EXTRA_EVAL_ARGS "${extra_args[@]}"
    rc=$?
    [ $rc -ne 0 ] && break
  done
  echo "=== $STREAM @ $(date) exit=$rc ==="
  if [ $rc -eq 0 ]; then echo "DONE_${STREAM}_vllm"; else echo "FAIL_${STREAM}_vllm (rc=$rc)"; fi
} > "$LOG" 2>&1
