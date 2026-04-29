#!/bin/bash
# Qwen3-30B-A3B KV-quant sweep, 4 variants × 3 tasks = 12 cells, ADAPTIVE 3-PASS.
#
# Per cell:
#   pass1: run_eval_vllm_qwen3.py at MG=8192 with --log_samples
#          (catches ~90% of items; thinking-mode answers usually < 8k tokens).
#   pass2: scripts/adaptive_rerun_qwen3.py at MG=32768 on the truncated tail
#          (Qwen3 native ctx 40k → MG=32k cap per the user's rule).
#   pass3: merge_rerun.py — re-applies lm_eval filter chain on the merged
#          samples and writes the final scored result file.
#
# 8 GPUs / TP=2 = 4 pairs, 4 cells in parallel; 3 waves of 4 cells.
# Single full-MG=32k pass would be 4-8h per cell ("crazy long"); adaptive
# 3-pass is bounded by pass1 wall-time + a small pass2 tail.
#
# Path B fusion: Qwen3 has q_norm/k_norm RMSNorm pre-RoPE. SmoothKV must
# use head-uniform s_K (folded into qk_norm.gamma). Calib:
#   logs/calib/smoothkv_qwen3-30b-a3b_perc_ns512_puremax_a1_huk_halfpair.pt
#
# Chat-template policy: ALL three tasks use --apply_chat_template — Qwen3 is
# heavily chat-tuned, runs in thinking mode by default; without chat it
# loses instruction-following entirely.
#
# Mechanism #2: vllm_custom/patches_qwen3.py via run_eval_vllm_qwen3.py +
# adaptive_rerun_qwen3.py. The kivi_vllm_plugin path is silent-no-op in
# vLLM V1, confirmed earlier in this session.
#
# Usage (overnight, persistent):
#   nohup bash scripts/qwen3_30b_a3b_quant_sweep.sh \
#     > logs/run_out/qwen3_30b_a3b_overnight.log 2>&1 &
#   disown
set -u
REPO=/home/home-mcl/sunghyuck/kv_cache_compression/KIVI
cd "$REPO"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

PY=/opt/vllm_qwen3_env/bin/python
SKENV=/home/home-mcl/sunghyuck/kv_cache_compression/SnapKV/snapkv_test_env/bin/python
MODEL=Qwen/Qwen3-30B-A3B
CALIB=logs/calib/smoothkv_qwen3-30b-a3b_perc_ns512_puremax_a1_huk_halfpair.pt
MG_PASS1=8192
MG_PASS2=32768
MML_PASS1=11648         # ~2800 prompt + 8192 gen + slack, 256-rounded
MML_PASS2=35840         # ~2800 prompt + 32768 gen + slack, 256-rounded
MAX_NS_PASS1=24
MAX_NS_PASS2=8          # smaller batch in pass2 since long generations dominate
BS=24

mkdir -p logs/run_out logs/qwen3_30b_a3b_results/mg32k

variant_args_p1() {
  case "$1" in
    bf16)     echo "--model bf16" ;;
    fp8)      echo "--model fp8 --bits 4 --group_size 128" ;;
    pertoken) echo "--model pertoken --bits 4 --group_size 128" ;;
    smoothkv) echo "--model smoothkv --bits 4 --group_size 128 --calib_path $CALIB" ;;
  esac
}

# Single-cell runner: 3 passes serially on the same GPU pair.
run_cell() {
  local pair=$1 variant=$2 task=$3
  local g1=${pair%,*} g2=${pair#*,}
  local stem=qwen3-30b-a3b_${variant}_${task}
  local outdir=logs/qwen3_30b_a3b_results/${stem}_pass1
  local samples_jsonl
  local samples_json=logs/qwen3_30b_a3b_results/mg32k/${stem}_pass1_samples.json
  local rerun_json=logs/qwen3_30b_a3b_results/mg32k/${stem}_samples_rerun32k.json
  local merged_json=logs/qwen3_30b_a3b_results/mg32k/${stem}_pass1_samples_merged.json
  local cell_log=logs/run_out/qwen3_30b_a3b_${variant}_${task}_g${g1}${g2}.log

  mkdir -p "$outdir"
  {
    echo "============================================="
    echo "[$(date)] $stem  pair=$pair  start"
    echo "============================================="

    # ── PASS 1 ────────────────────────────────────────────────────────────
    # run_eval_vllm_qwen3.py auto-saves to logs/<task>_<modelshort>_<variant>[_g128]_chat_vllm_samples.json
    # We compute that path then copy/rename into our deterministic layout.
    local model_short=qwen3-30b-a3b
    local variant_tag
    case "$variant" in
      bf16)     variant_tag="bf16" ;;
      fp8)      variant_tag="fp8_g128" ;;
      pertoken) variant_tag="pertoken_int4_g128" ;;
      smoothkv) variant_tag="smoothkv_g128_perc_ns512_puremax_a1_huk_halfpair" ;;
    esac
    local autosaved_samples=logs/${task}_${model_short}_${variant_tag}_chat_vllm_samples.json

    if [ -f "$samples_json" ]; then
      echo "PASS1 SKIP: $samples_json already present"
    else
      echo "--- PASS1: MG=$MG_PASS1 with --log_samples ---"
      local va; va=$(variant_args_p1 "$variant")
      PATH=/usr/bin:/bin:$PATH NO_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=$pair \
        $PY run_eval_vllm_qwen3.py $va \
          --model_path $MODEL --tp 2 --task $task --apply_chat_template \
          --max_gen_toks $MG_PASS1 --max_model_len $MML_PASS1 \
          --max_num_seqs $MAX_NS_PASS1 --batch_size $BS \
          --log_samples
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAIL_${stem} (pass1 rc=$rc)"; return 1
      fi
      if [ ! -f "$autosaved_samples" ]; then
        echo "FAIL_${stem} (pass1: expected $autosaved_samples, not found)"
        ls logs/${task}_${model_short}_*.json 2>/dev/null | head -5
        return 1
      fi
      cp "$autosaved_samples" "$samples_json"
      echo "  copied $autosaved_samples -> $samples_json"
    fi

    # ── PASS 2 ────────────────────────────────────────────────────────────
    if [ -f "$rerun_json" ]; then
      echo "PASS2 SKIP: $rerun_json already present"
    else
      echo "--- PASS2: adaptive_rerun_qwen3.py at MG=$MG_PASS2 (truncated only) ---"
      local rerun_args="--model_path $MODEL --tp 2"
      rerun_args="$rerun_args --quant_method $variant"
      [ "$variant" = "smoothkv" ] && rerun_args="$rerun_args --calib_path $CALIB"
      [ "$variant" = "fp8" ] || [ "$variant" = "pertoken" ] && rerun_args="$rerun_args --bits 4 --group_size 128"
      [ "$variant" = "smoothkv" ] && rerun_args="$rerun_args --bits 4 --group_size 128"
      PATH=/usr/bin:/bin:$PATH NO_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=$pair \
        $PY scripts/adaptive_rerun_qwen3.py \
          --samples "$samples_json" $rerun_args \
          --new_max_gen_toks $MG_PASS2 \
          --max_num_seqs $MAX_NS_PASS2 \
          --out "$rerun_json"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAIL_${stem} (pass2 rc=$rc)"; return 1
      fi
    fi

    # ── PASS 3 ────────────────────────────────────────────────────────────
    echo "--- PASS3: merge_rerun ---"
    "$SKENV" scripts/merge_rerun.py \
        --original "$samples_json" \
        --rerun    "$rerun_json" \
        --task     "$task" \
        --out      "$merged_json"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "FAIL_${stem} (pass3 rc=$rc)"; return 1
    fi

    echo "============================================="
    echo "[$(date)] $stem  DONE"
    echo "DONE_${stem}"
    echo "============================================="
  } > "$cell_log" 2>&1
  return 0
}

PAIRS=("0,1" "2,3" "4,5" "6,7")

# Build the cell list (12 cells = 4 variants × 3 tasks)
declare -a CELLS
i=0
for variant in bf16 fp8 pertoken smoothkv; do
  for task in gsm8k_32k math500_32k gpqa_main_cot_n_shot_32k; do
    CELLS[i]="$variant:$task"
    i=$((i+1))
  done
done

echo "[$(date)] Qwen3-30B-A3B adaptive 3-pass sweep start: 12 cells, 4 GPU pairs, 3 waves"

for wave_start in 0 4 8; do
  echo "[$(date)] === wave $((wave_start/4 + 1)): cells $wave_start-$((wave_start+3)) ==="
  pids=()
  for offset in 0 1 2 3; do
    idx=$((wave_start + offset))
    [ $idx -ge ${#CELLS[@]} ] && break
    cell="${CELLS[$idx]}"
    variant="${cell%:*}"; task="${cell#*:}"
    pair="${PAIRS[$offset]}"
    echo "  GPU $pair  $variant  $task  (3-pass)"
    run_cell "$pair" "$variant" "$task" &
    pids+=($!)
    sleep 8
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "[$(date)] === wave $((wave_start/4 + 1)) complete ==="
done

echo "[$(date)] Qwen3-30B-A3B adaptive 3-pass sweep complete (all 12 cells)"
