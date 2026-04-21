#!/bin/bash
# Overnight orchestrator — runs in its own tmux window, polls for Phase 1 completion,
# then fires Phase 2 (Mistral pair-mergeable), then Phase 3 (Llama-3-8B base + R1-Distill-Llama-8B).
# Writes /workspace/KIVI/logs/run_out/overnight_watcher.log so we can inspect progress.
#
# Independent of any Claude Code session — only needs a bash + tmux running.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
LOG=logs/run_out/overnight_watcher.log
: > "$LOG"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# ===== helpers =====
stream_done() {
  local name=$1
  # Window gone => done (also terminal)
  if ! tmux list-windows -t kivi -F '#{window_name}' 2>/dev/null | grep -qx "$name"; then
    return 0
  fi
  local idx
  idx=$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' 2>/dev/null | awk -v n="$name" '$2==n {print $1; exit}')
  [ -z "$idx" ] && return 0
  # DONE_ = clean success; FAIL_ = python non-zero exit. Both are terminal states we stop waiting on.
  tmux capture-pane -t kivi:$idx -p -S -100 2>/dev/null | grep -qE "^(DONE|FAIL)_$name"
}

wait_streams_done() {
  # wait_streams_done <interval_sec> name1 name2 ...
  local interval=$1; shift
  local pending=("$@")
  while true; do
    local new_pending=()
    for n in "${pending[@]}"; do
      if ! stream_done "$n"; then new_pending+=("$n"); fi
    done
    pending=("${new_pending[@]}")
    if [ ${#pending[@]} -eq 0 ]; then return 0; fi
    log "waiting on: ${pending[*]}"
    sleep "$interval"
  done
}

is_gpu_free_once() {
  local g=$1 mem pids
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$mem" ] || [ "$mem" -ge 20000 ] && return 1
  pids=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && return 1
  return 0
}

# Regenerate reports
regen_report() {
  log "regenerating HTML reports"
  /opt/modernenv/bin/python generate_summary.py >> "$LOG" 2>&1
  /opt/modernenv/bin/python generate_simplified_report.py >> "$LOG" 2>&1
}

# ===== Phase 1 wait =====
log "=== waiting for Phase 1 (Llama-2 pair) to finish ==="
wait_streams_done 60 pPfpair_l2_pairK95 pPfpair_l2_pairK99 pPfpair_l2_pairK99p9 pPfpair_l2_a075pair
log "=== Phase 1 done ==="
regen_report

# ===== Phase 2: Mistral pair-mergeable =====
log "=== launching Phase 2 (Mistral pair-mergeable) ==="
bash scripts/phaseP2_mistral_launch.sh >> "$LOG" 2>&1
log "=== Phase 2 streams launched ==="

# Wait for all 5 Mistral streams (4 in wave 1 + 1 in wave 2)
wait_streams_done 120 \
  pPfpair_m1_pairK90 pPfpair_m1_pairK95 pPfpair_m1_pairK99 pPfpair_m1_pairK99p9 pPfpair_m1_a075pair
log "=== Phase 2 done ==="
regen_report

# ===== Phase 2b: historical reasoning re-run for Llama-2 + Mistral at 32k =====
log "=== launching Phase 2b (historical reasoning re-run) ==="
bash scripts/phase2b_historical_reasoning_launch.sh >> "$LOG" 2>&1
log "=== Phase 2b streams launched ==="
wait_streams_done 120 \
  p2b_l2_fp16 p2b_l2_kivi2_g32r128 p2b_l2_kivi4_g32r128 p2b_l2_kivi4_g128r128 p2b_l2_pertoken p2b_l2_fp8 \
  p2b_l2_a075pair p2b_l2_pairK90 p2b_l2_pairK95 p2b_l2_pairK99 p2b_l2_pairK99p9 \
  p2b_m1_fp16 p2b_m1_kivi2_g32r128 p2b_m1_kivi4_g32r128 p2b_m1_kivi4_g128r128 p2b_m1_pertoken p2b_m1_fp8
log "=== Phase 2b done ==="
regen_report

# ===== Phase 3: Llama-3-8B base + R1-Distill-Llama-8B =====
log "=== launching Phase 3 (Llama-3-8B + R1-Distill) calibration ==="
bash scripts/phase3_calib_launch.sh >> "$LOG" 2>&1
wait_streams_done 120 calib_llama3_8b calib_r1d_llama8b
log "=== Phase 3 calibration done ==="

log "=== launching Phase 3 eval ==="
bash scripts/phase3_eval_launch.sh >> "$LOG" 2>&1
log "=== Phase 3 eval streams launched — monitor logs/run_out/phase3_*.log ==="
# Let cron/generate_summary handle the final regeneration; overnight_watcher exits here.

log "=== overnight_watcher done, exiting ==="
