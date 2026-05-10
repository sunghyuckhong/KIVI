#!/usr/bin/env bash
# Check that kv_fake_quant/ is byte-identical between the two vllm forks
# we edit (vllm-compression-part for .venv, vllm-exaone-fork for .venv-exaone).
#
# Drift here silently breaks NVFP4 sweeps for one venv but not the other.
# See CLAUDE.md "Two vllm forks — keep kv_fake_quant/ in sync".
#
# Exit 0 if in sync, 1 if drift detected.
set -euo pipefail

A=/workspace/sunghyuck/vllm-compression-part/vllm/model_executor/layers/quantization/kv_fake_quant
B=/workspace/sunghyuck/vllm-exaone-fork/vllm/model_executor/layers/quantization/kv_fake_quant

if [ ! -d "$A" ] || [ ! -d "$B" ]; then
    echo "[sync-check] one of the kv_fake_quant trees is missing:"
    echo "  A=$A  exists=$( [ -d "$A" ] && echo yes || echo no )"
    echo "  B=$B  exists=$( [ -d "$B" ] && echo yes || echo no )"
    exit 1
fi

# diff -r without recursing into __pycache__ (those auto-regenerate per venv).
if diff -r --exclude=__pycache__ "$A" "$B" >/dev/null; then
    echo "[sync-check] OK — kv_fake_quant trees are identical"
    exit 0
fi

echo "[sync-check] FAIL — kv_fake_quant trees differ:"
diff -r --exclude=__pycache__ --brief "$A" "$B" | sed 's/^/  /'
echo ""
echo "Both forks must carry the same kv_fake_quant/ code, or sweeps that hit"
echo "one venv will run the other venv's kernel. Mirror the changes and commit"
echo "to BOTH repos:"
echo "  $A"
echo "  $B"
exit 1
