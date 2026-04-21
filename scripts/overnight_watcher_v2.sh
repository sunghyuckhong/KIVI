#!/bin/bash
# Narrower overnight chain — Mistral reasoning re-run → Llama-3-8B full eval.
# Drops the broader Llama-2 re-run + R1-Distill-Llama-8B from last night's failed plan.
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
  local idx
  idx=$(tmux list-windows -t kivi -F '#{window_index} #{window_name}' 2>/dev/null | awk -v n="$name" '$2==n {print $1; exit}')
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

# ===== Phase X-Mistral: Mistral reasoning re-run at 16384 max_gen_toks =====
log "=== launching phaseX Mistral (reasoning @ 16384) ==="
bash scripts/phaseX_mistral_reasoning_launch.sh >> "$LOG" 2>&1
wait_streams_done 120 \
  pX_m1_fp16 pX_m1_kivi2_g32r128 pX_m1_kivi4_g32r128 pX_m1_kivi4_g128r128 \
  pX_m1_pertoken pX_m1_fp8 \
  pX_m1_a075pair pX_m1_pairK90 pX_m1_pairK95 pX_m1_pairK99 pX_m1_pairK99p9
log "=== phaseX Mistral done ==="
regen_report

# ===== Phase X-Llama3: Llama-3-8B base full matrix =====
log "=== launching phaseX Llama-3-8B base (calib + 9 methods, reasoning @ 4096) ==="
bash scripts/phaseX_llama3_launch.sh >> "$LOG" 2>&1
wait_streams_done 120 \
  pX_l3_fp16 pX_l3_kivi2 pX_l3_pertoken pX_l3_fp8 \
  pX_l3_a075pair pX_l3_pairK90 pX_l3_pairK95 pX_l3_pairK99 pX_l3_pairK99p9
log "=== phaseX Llama-3-8B done ==="
regen_report
log "=== overnight_watcher v2 finished ==="
