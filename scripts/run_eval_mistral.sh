#!/bin/bash
# Reproduce graph-verified KV-quant accuracy for Mistral family.
#
# Mistral has NO q_norm/k_norm → SmoothKV uses _pair calib.
# Same recipe as run_eval_llama.sh; this is just a shim that selects a
# Mistral default model_path.
#
# Usage:
#   bash scripts/run_eval_mistral.sh \
#       [--model mistralai/Mistral-7B-Instruct-v0.2] \
#       --variant bf16|fp8|pertoken|smkv \
#       --task gsm8k_cot|minerva_math500|gpqa_main_cot_n_shot \
#       [--ns 512] [--alpha 1.0] [--beta 1.0] \
#       [--gpus 0]
set -euo pipefail
MODEL_DEFAULT="mistralai/Mistral-7B-Instruct-v0.2"

# Pass through to run_eval_llama.sh (same recipe — no q_norm)
SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && /bin/pwd)"

# Inject default --model if not provided
HAS_MODEL=0
for arg in "$@"; do
  [ "$arg" = "--model" ] && HAS_MODEL=1
done

if [ "$HAS_MODEL" -eq 0 ]; then
  /usr/bin/echo "[mistral] using default model $MODEL_DEFAULT"
  exec "$SCRIPT_DIR/run_eval_llama.sh" --model "$MODEL_DEFAULT" "$@"
else
  exec "$SCRIPT_DIR/run_eval_llama.sh" "$@"
fi
