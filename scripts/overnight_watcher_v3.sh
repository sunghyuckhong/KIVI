#!/bin/bash
# Instruct-model chain: Llama-3-8B-Instruct first (faster, 4k gen), then Mistral-7B-Instruct-v0.2 (16k gen).
# Single launcher handles both: parallel calibration, variant generation, then eval streams.
set -u
cd /workspace/KIVI
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running}
LOG=logs/run_out/overnight_watcher.log
: > "$LOG"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

stream_done() {
  local name=$1
  if ! tmux list-windows -t kivi -F '#{window_name}' 2>/dev/null | grep -qx "$name"; then
    return 0
  fi
  local idx=$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' 2>/dev/null | awk -v n="$name" '$2==n{print $1; exit}')
  [ -z "$idx" ] && return 0
  tmux capture-pane -t kivi:$idx -p -S -100 2>/dev/null | grep -qE "^(DONE|FAIL)_$name"
}
wait_streams_done() {
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
regen_report() {
  log "regenerating HTML reports"
  /opt/modernenv/bin/python generate_summary.py >> "$LOG" 2>&1
  /opt/modernenv/bin/python generate_simplified_report.py >> "$LOG" 2>&1
}

log "=== launching phaseX Instruct (Llama-3-8B-Instruct first, then Mistral-7B-Instruct-v0.2) ==="
bash scripts/phaseX_instruct_launch.sh >> "$LOG" 2>&1
log "=== all streams launched ==="

# Wait for all 18 eval streams (and final Mistral streams which come after Llama-3 in the order)
wait_streams_done 120 \
  pX_meta_fp16 pX_meta_kivi2 pX_meta_pertoken pX_meta_fp8 \
  pX_meta_a075pair pX_meta_pairK90 pX_meta_pairK95 pX_meta_pairK99 pX_meta_pairK99p9 \
  pX_mist_fp16 pX_mist_kivi2 pX_mist_pertoken pX_mist_fp8 \
  pX_mist_a075pair pX_mist_pairK90 pX_mist_pairK95 pX_mist_pairK99 pX_mist_pairK99p9
log "=== phaseX Instruct done ==="
regen_report
log "=== overnight_watcher v3 finished ==="
