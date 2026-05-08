#!/bin/bash
# One-time setup of /workspace/KIVI/.venv-exaone with:
#   - PyTorch (driver-detected cu128/cu130 wheels)
#   - nuxlear/transformers @ add-exaone4_5-v5.3.0.dev0 (recognizes exaone4_5)
#   - sunghyuckhong/vllm-compression-part @ kv_cache_quant_exaone4_5
#     (KV-cache fake-quant code rebased onto lkm2835/vllm@add-exaone4_5)
#   - KIVI eval-harness deps (lm-eval etc.)
#
# Clones the fork from VLLM_EXAONE_FORK_URL into VLLM_EXAONE_FORK_PATH if
# missing, then checks out VLLM_EXAONE_FORK_COMMIT. Variables are exported by
# the Makefile (target `make setup-exaone-4.5`); override on the command line
# if running this script directly.

set -euo pipefail
cd /workspace/KIVI

VENV=.venv-exaone
VLLM_URL="${VLLM_EXAONE_FORK_URL:-https://github.com/sunghyuckhong/vllm-compression-part.git}"
VLLM_PATH="${VLLM_EXAONE_FORK_PATH:-/workspace/sunghyuck/vllm-exaone-fork}"
VLLM_BRANCH="${VLLM_EXAONE_FORK_BRANCH:-kv_cache_quant_exaone4_5}"
VLLM_COMMIT="${VLLM_EXAONE_FORK_COMMIT:-545dcdf69}"

LOG=logs/run_out/setup_venv_exaone.log
exec >>"$LOG" 2>&1
ts() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

echo ""
echo "================================================================"
echo "[$(ts)] Setup .venv-exaone"
echo "================================================================"

if [ -d "$VENV" ]; then
  echo "[$(ts)] $VENV already exists; will reuse"
else
  python3 -m venv "$VENV"
  echo "[$(ts)] created $VENV"
fi

PIP="$VENV/bin/pip"

# Driver-aware PyTorch install (mirrors Makefile setup logic)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
DRIVER_MAJOR=${DRIVER%%.*}
if [ "$DRIVER_MAJOR" -ge 575 ]; then
  PT_INDEX=https://download.pytorch.org/whl/cu130; PT_LABEL=cu130
elif [ "$DRIVER_MAJOR" -ge 555 ]; then
  PT_INDEX=https://download.pytorch.org/whl/cu128; PT_LABEL=cu128
else
  echo "[$(ts)] ERROR: driver $DRIVER too old"; exit 1
fi
echo "[$(ts)] driver=$DRIVER  →  $PT_LABEL wheels"

$PIP install -U pip wheel
$PIP install torch==2.11.0 torchvision==0.26.0 --index-url "$PT_INDEX"

# Step: clone (or reuse) the vllm fork, then checkout the pinned commit.
# Mirrors the regular `make setup` flow for the .venv vllm-compression-part fork.
if [ ! -d "$VLLM_PATH" ]; then
  echo "[$(ts)] cloning $VLLM_URL @ $VLLM_BRANCH → $VLLM_PATH"
  git clone --branch "$VLLM_BRANCH" "$VLLM_URL" "$VLLM_PATH"
else
  echo "[$(ts)] $VLLM_PATH already exists; reusing"
fi
echo "[$(ts)] checkout pinned commit $VLLM_COMMIT"
(cd "$VLLM_PATH" && git fetch --all && git checkout "$VLLM_COMMIT")

# Step: install the vllm fork from local path.
# Our KV-cache fake-quant code is pure-Python (kv_fake_quant package +
# Attention/Worker integration); no C/C++ changes, so VLLM_USE_PRECOMPILED=1
# downloads precompiled binaries from the vllm release that matches the
# fork's base commit. This skips the cmake/nvcc build (the system has
# CUDA 11.8, but PyTorch 2.11 + vllm need CUDA 12.x).
#
# We must pin VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 (the vllm wheel index
# serves only cu129 / cu130; cu130 needs driver >= 575, but the typical
# runpod box runs driver 570 → only cu129 is loadable).
echo "[$(ts)] installing vllm fork from $VLLM_PATH (VLLM_USE_PRECOMPILED=1, cu129)"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 $PIP install -e "$VLLM_PATH"

# Step: install nuxlear/transformers AFTER vllm. The vllm install hard-pins
# transformers (4.57.x as of writing) and would uninstall the nuxlear fork
# if installed earlier. We use --no-deps to keep the rest of the env intact.
# We also force huggingface_hub>=1.3.0 because the nuxlear fork's hub.py
# imports `is_offline_mode` which exists in hub 1.x but not in 0.36 (the
# vllm-pinned version).
echo "[$(ts)] installing nuxlear/transformers @ add-exaone4_5-v5.3.0.dev0 + huggingface_hub 1.x (--no-deps)"
$PIP install --no-deps --force-reinstall \
  "git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0" \
  "huggingface_hub>=1.3.0,<2.0"

# Step: KIVI requirements (lm-eval, math-verify, ray, etc.)
echo "[$(ts)] installing KIVI requirements"
[ -f requirements.txt ] && $PIP install -r requirements.txt

echo "[$(ts)] DONE — verify with .venv-exaone/bin/python -c 'import vllm; from transformers import AutoConfig; AutoConfig.from_pretrained(\"LGAI-EXAONE/EXAONE-4.5-33B\", trust_remote_code=True)'"
