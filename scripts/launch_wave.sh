#!/bin/bash
# Generic wave launcher. Each call creates a tmux window that runs a chain of
# harness invocations with bs=1 (paper-exact) and tees to logs/run_out/<name>.log.
#
# Usage:
#   launch_wave.sh <window_name> <gpu_id> <cmd...>
# Example:
#   launch_wave.sh llama_kivi2 1 'python run_lm_eval_harness.py ... ; python ...'

set -eu
name=$1
gpu=$2
shift 2
cmd="$*"

# Derive repo root from this script's location (pod-portable); override with REPO_ROOT env var.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/run_out

tmux new-window -t kivi: -n "$name" \
  "cd $REPO_ROOT; \
   export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN before running} CUDA_VISIBLE_DEVICES=$gpu; \
   ( $cmd ) 2>&1 | tee logs/run_out/$name.log; \
   rc=\${PIPESTATUS[0]}; \
   if [ \"\$rc\" -eq 0 ]; then echo DONE_$name; else echo \"FAIL_$name rc=\$rc\"; fi; \
   read"
echo "launched $name on GPU$gpu (tmux window)"
