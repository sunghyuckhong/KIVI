#!/bin/bash
# Wait for the Qwen3-32B MG=32k sweep (12 result files) to finish,
# then launch Qwen3-8B MG=32k sweep on 4 variants × 3 tasks at TP=1.
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /workspace/KIVI
LOG=/workspace/KIVI/logs/run_out/qw8_after_qw32_watcher.log

echo "[watcher] start at $(/bin/date)" > "$LOG"

QW32_FILES=(
  logs/gsm8k_32k_qwen3-32b_bf16_chat_vllm_results.json
  logs/minerva_math500_qwen3-32b_bf16_chat_vllm_results.json
  logs/gpqa_main_cot_n_shot_32k_qwen3-32b_bf16_chat_vllm_results.json
  logs/gsm8k_32k_qwen3-32b_fp8_g128_chat_vllm_results.json
  logs/minerva_math500_qwen3-32b_fp8_g128_chat_vllm_results.json
  logs/gpqa_main_cot_n_shot_32k_qwen3-32b_fp8_g128_chat_vllm_results.json
  logs/gsm8k_32k_qwen3-32b_pertoken_int4_g128_chat_vllm_results.json
  logs/minerva_math500_qwen3-32b_pertoken_int4_g128_chat_vllm_results.json
  logs/gpqa_main_cot_n_shot_32k_qwen3-32b_pertoken_int4_g128_chat_vllm_results.json
  logs/gsm8k_32k_qwen3-32b_smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_results.json
  logs/minerva_math500_qwen3-32b_smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_results.json
  logs/gpqa_main_cot_n_shot_32k_qwen3-32b_smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair_chat_vllm_results.json
)

while true; do
  missing=0
  for f in "${QW32_FILES[@]}"; do
    if [ ! -f "$f" ]; then missing=$((missing+1)); fi
  done
  echo "[watcher] $(/bin/date) — qwen3-32b missing=$missing/${#QW32_FILES[@]}" >> "$LOG"
  if [ $missing -eq 0 ]; then
    echo "[watcher] all qwen3-32b sentinels present at $(/bin/date) — launching qwen3-8b sweep" >> "$LOG"
    break
  fi
  /usr/bin/sleep 120
done

# Wait an extra 30s for any pending writes / vllm cleanup
/usr/bin/sleep 30

CALIB=/workspace/KIVI/logs/calib/smoothkv_qwen3-8b_perc_ns512_puremax_a1b1_halfpair.pt
SCRIPT_DIR="$(/usr/bin/dirname "$(/usr/bin/realpath "$0")")"
VARIANT="$SCRIPT_DIR/qw8_mg32k_variant.sh"

# Launch 4 variants in parallel, each pinned to one GPU at TP=1
/usr/bin/tmux new-session -d -s qw8m_bf16 "$VARIANT 0 '--model bf16' bf16 qw8m_bf16"
/usr/bin/tmux new-session -d -s qw8m_fp8  "$VARIANT 1 '--model fp8 --group_size 128' fp8_g128 qw8m_fp8"
/usr/bin/tmux new-session -d -s qw8m_pert "$VARIANT 2 '--model pertoken --bits 4 --group_size 128' pertoken_int4_g128 qw8m_pert"
/usr/bin/tmux new-session -d -s qw8m_smk  "$VARIANT 3 '--model smoothkv_fused --calib_path $CALIB --group_size 128 --bits 4' smoothkv_fused_g128_perc_ns512_puremax_a1b1_halfpair qw8m_smk"

echo "[watcher] launched 4 qwen3-8b tmux sessions at $(/bin/date)" >> "$LOG"
/usr/bin/tmux ls >> "$LOG" 2>&1
