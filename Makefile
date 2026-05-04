# =============================================================================
# KIVI evaluation sweeps
# =============================================================================
#
# Per-family entry points that run a 4-variant × 3-task sweep against a fixed
# model. Each cell is one (variant, task) invocation of the underlying
# scripts/run_eval_<family>.sh runner — pass1@MG=4k + pass2@MG=32k with the
# verify-graph trust gate at both passes.
#
# Quick start:
#   make setup                            # one-time: build vllm fork + venv
#   make test                             # unit tests after setup
#   make run-qwen3-8b                     # full sweep on Qwen3-8B (single GPU)
#   make run-qwen3-32b GPUS=0,1           # Qwen3-32B (TP=2)
#   make run-llama                        # Llama-3-8B-Instruct
#   make run-mistral                      # Mistral-7B-Instruct-v0.2
#   make run-exaone GPUS=0,1              # EXAONE-4.5-33B (TP=2)
#   make run-all                          # qwen3-8b + llama + mistral
#
# Common overrides:
#   make run-qwen3-8b VARIANTS="bf16 smkv"
#   make run-llama TASKS=gsm8k_cot
#   make run-llama LLAMA_MODEL=meta-llama/Meta-Llama-3-70B-Instruct GPUS="0,1"
#   make run-qwen3-8b ALPHA=0.5 BETA=0.5  # SmoothKV variant sweep
# =============================================================================

SHELL := /bin/bash

# --- vllm-compression-part fork (KV fake-quant lives here) -------------------
# Pinned commit on kv_cache_quant; bump after re-verifying graph capture.
VLLM_FORK_PATH    ?= /workspace/sunghyuck/vllm-compression-part
VLLM_FORK_BRANCH  ?= kv_cache_quant
VLLM_FORK_COMMIT  ?= 675ba44c0a

VENV       ?= .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip

# --- Defaults (override on command line or via env) --------------------------
VARIANTS       ?= bf16 fp8 pertoken smkv
TASKS          ?= gsm8k_cot minerva_math500 gpqa_main_cot_n_shot_32k
NS             ?= 512
ALPHA          ?= 1.0
BETA           ?= 1.0

LLAMA_MODEL    ?= meta-llama/Meta-Llama-3-8B-Instruct
MISTRAL_MODEL  ?= mistralai/Mistral-7B-Instruct-v0.2
GPUS           ?= auto  # auto = scan idle (< 2GB used), chunk by --tp; or "0,1,2,3"

# --- Colors ------------------------------------------------------------------
# Store the actual ESC byte (not the literal "\033" string) so awk/echo/printf
# all emit the sequence directly without further escape interpretation.
GREEN  := $(shell printf '\033[0;32m')
YELLOW := $(shell printf '\033[0;33m')
CYAN   := $(shell printf '\033[0;36m')
RESET  := $(shell printf '\033[0m')

# =============================================================================

.PHONY: help status setup setup-fork test print-pin \
        run-qwen3-8b run-qwen3-32b run-llama run-mistral run-exaone run-all

# --- Environment setup -------------------------------------------------------
setup: $(VENV)/bin/activate ## One-time: build vllm fork + venv + install KIVI deps
	@echo "$(GREEN)>>> Checkout vllm fork at pinned commit $(VLLM_FORK_COMMIT)$(RESET)"
	cd $(VLLM_FORK_PATH) && git fetch --all && git checkout $(VLLM_FORK_COMMIT)
	@echo "$(GREEN)>>> Install vllm-compression-part (editable; vllm._C compile ~15-30min)$(RESET)"
	$(PIP) install -U pip wheel
	$(PIP) install -e $(VLLM_FORK_PATH)
	@echo "$(GREEN)>>> Install KIVI deps$(RESET)"
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; \
	  else echo "$(YELLOW)[setup] no requirements.txt; skipping$(RESET)"; fi
	@if [ -f setup.py ] || [ -f pyproject.toml ]; then $(PIP) install -e .; \
	  else echo "$(YELLOW)[setup] no setup.py/pyproject.toml; skipping editable KIVI install$(RESET)"; fi
	@echo "$(GREEN)[setup] DONE. Verify with 'make test'.$(RESET)"

setup-fork: ## Re-checkout the pinned commit + reinstall the fork only (faster)
	@[ -d $(VENV) ] || (echo "ERROR: run 'make setup' first" && exit 1)
	cd $(VLLM_FORK_PATH) && git fetch --all && git checkout $(VLLM_FORK_COMMIT)
	$(PIP) install -e $(VLLM_FORK_PATH)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

test: ## Run unit tests (requires `make setup`)
	$(PY) -m pytest tests/test_verify_graph.py tests/test_quant_dtype.py -v

print-pin: ## Print the pinned vllm-compression-part commit
	@echo "vllm-compression-part: $(VLLM_FORK_BRANCH) @ $(VLLM_FORK_COMMIT)"
	@echo "venv: $(VENV)"


help: ## Show available targets
	@echo ""
	@echo "  KIVI evaluation sweeps"
	@echo "  ======================"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /:## /' | \
		awk 'BEGIN {FS = ":## "}; {printf "  $(CYAN)%-16s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "  Common overrides: VARIANTS, TASKS, GPUS, NS, ALPHA, BETA,"
	@echo "                    LLAMA_MODEL, MISTRAL_MODEL"
	@echo ""

status: ## Print resolved sweep parameters
	@echo ""
	@echo "  VARIANTS:       $(VARIANTS)"
	@echo "  TASKS:          $(TASKS)"
	@echo "  GPUS:           $(GPUS)"
	@echo "  NS:             $(NS)"
	@echo "  ALPHA:          $(ALPHA)"
	@echo "  BETA:           $(BETA)"
	@echo "  LLAMA_MODEL:    $(LLAMA_MODEL)"
	@echo "  MISTRAL_MODEL:  $(MISTRAL_MODEL)"
	@echo ""

# --- Per-family sweeps: VARIANTS × TASKS via scripts/run_eval_<family>.sh ---
#
# Each target runs through scripts/parallel_sweep.py, which detects idle GPUs
# (or uses an explicit pool) and runs cells in parallel — one cell per
# TP-sized GPU stream, scheduled greedily as streams free up.
#
#   GPUS=auto (default): scan nvidia-smi for GPUs with memory.used < 2GB,
#                        chunk by --tp, run as many concurrent streams as fit
#   GPUS="0,1,2,3":      explicit pool (still chunked by --tp; e.g. for TP=2
#                        this means 2 streams [0,1] and [2,3])
#   GPUS="0,1":          single stream (sequential cells)
#
# Sweep arg: --gpus is only forwarded when GPUS != "auto" (otherwise
# parallel_sweep.py auto-detects).
SWEEP = $(PY) scripts/parallel_sweep.py \
        --variants $(VARIANTS) --tasks $(TASKS) \
        --ns $(NS) --alpha $(ALPHA) --beta $(BETA) \
        $(if $(filter-out auto,$(GPUS)),--gpus $(GPUS))

run-qwen3-8b: ## Sweep Qwen3-8B (TP=1, auto-parallel)
	@$(SWEEP) --tp 1 --runner scripts/run_eval_qwen3.sh --runner_args="--size 8b"

run-qwen3-32b: ## Sweep Qwen3-32B (TP=2, auto-parallel; e.g. 8 idle GPUs → 4 streams)
	@$(SWEEP) --tp 2 --runner scripts/run_eval_qwen3.sh --runner_args="--size 32b"

run-llama: ## Sweep $(LLAMA_MODEL) (TP=1, auto-parallel)
	@$(SWEEP) --tp 1 --runner scripts/run_eval_llama.sh --runner_args="--model $(LLAMA_MODEL)"

run-mistral: ## Sweep $(MISTRAL_MODEL) (TP=1, auto-parallel)
	@$(SWEEP) --tp 1 --runner scripts/run_eval_mistral.sh --runner_args="--model $(MISTRAL_MODEL)"

run-exaone: ## Sweep EXAONE-4.5-33B (TP=2, auto-parallel)
	@$(SWEEP) --tp 2 --runner scripts/run_eval_exaone.sh --runner_args=""

run-all: run-qwen3-8b run-llama run-mistral ## Run qwen3-8b + llama + mistral (skips 32b/exaone)
