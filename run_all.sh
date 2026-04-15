#!/usr/bin/env bash
# run_all.sh — Launch all 12 KIVI experiments in parallel across 4 GPUs via tmux.
#
# Usage:
#   bash run_all.sh           # launch all sessions
#   bash run_all.sh --dry-run # print commands without running
#
# GPU assignment:
#   GPU 0 : FP16 baseline (GSM8K + GPQA)
#   GPU 1 : PerToken flat/group=128 variants (GSM8K + GPQA, residual=32 and residual=0)
#   GPU 2 : PerToken group=32, no-residual (GSM8K + GPQA)
#   GPU 3 : KIVI default + PerToken group=32 with residual (GSM8K + GPQA)
#
# Results land in logs/<experiment>_results.json.
# Run generate_report.py once all 12 are done.

set -euo pipefail
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

launch() {
    # launch SESSION GPU "cmd1 && cmd2 && ..."
    local session=$1 gpu=$2 cmds=$3
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[skip] session '$session' already exists — kill it first to re-run"
        return
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] tmux $session (GPU $gpu):"
        echo "  $cmds"
        return
    fi
    tmux new-session -d -s "$session"
    tmux send-keys -t "$session" \
        "cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$gpu $cmds" Enter
    echo "[started] $session  GPU=$gpu"
}

PY="python run_eval.py"

# ── GPU 0: FP16 baseline ─────────────────────────────────────────────────────
launch gpu0 0 \
    "$PY --model fp16 --task gsm8k 2>&1 | tee logs/gsm8k_fp16.log && \
     $PY --model fp16 --task gpqa  2>&1 | tee logs/gpqa_fp16.log"

# ── GPU 1: PerToken flat (group=128), residual=32 then residual=0 ─────────────
launch gpu1 1 \
    "$PY --model pertoken --task gsm8k --group_size 128 --residual 32 2>&1 | tee logs/gsm8k_pertoken_flat.log && \
     $PY --model pertoken --task gpqa  --group_size 128 --residual 32 2>&1 | tee logs/gpqa_pertoken_flat.log && \
     $PY --model pertoken --task gsm8k --group_size 128 --residual 0  2>&1 | tee logs/gsm8k_pertoken_flat_noresidual.log && \
     $PY --model pertoken --task gpqa  --group_size 128 --residual 0  2>&1 | tee logs/gpqa_pertoken_flat_noresidual.log"

# ── GPU 2: PerToken group=32, no residual ─────────────────────────────────────
launch gpu2 2 \
    "$PY --model pertoken --task gsm8k --group_size 32 --residual 0 2>&1 | tee logs/gsm8k_pertoken_noresidual.log && \
     $PY --model pertoken --task gpqa  --group_size 32 --residual 0 2>&1 | tee logs/gpqa_pertoken_noresidual.log"

# ── GPU 3: KIVI default + PerToken group=32, residual=32 ─────────────────────
launch gpu3 3 \
    "$PY --model kivi     --task gsm8k 2>&1 | tee logs/gsm8k_kivi.log && \
     $PY --model kivi     --task gpqa  2>&1 | tee logs/gpqa_kivi.log  && \
     $PY --model pertoken --task gsm8k --group_size 32 --residual 32  2>&1 | tee logs/gsm8k_pertoken.log && \
     $PY --model pertoken --task gpqa  --group_size 32 --residual 32  2>&1 | tee logs/gpqa_pertoken.log"

echo ""
echo "Sessions launched. Monitor progress:"
echo "  tmux attach -t gpu0          # Ctrl+B, D to detach"
echo "  tail -f logs/gsm8k_fp16.log"
echo ""
echo "When all 12 experiments are done:"
echo "  python generate_report.py"
