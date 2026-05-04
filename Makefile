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
#   make test                             # run unit tests after setup
#   make run-llama
#   make run-mistral
#   make run-qwen3                        # 8b
#   make run-qwen3 QWEN3_SIZE=32b         # 32b on TP=2 (auto GPUS=0,1)
#   make run-all                          # all three families
#
# Common overrides:
#   make run-qwen3 VARIANTS="bf16 smkv"
#   make run-llama TASKS=gsm8k_cot
#   make run-llama LLAMA_MODEL=meta-llama/Meta-Llama-3-70B-Instruct GPUS="0,1"
#   make run-qwen3 ALPHA=0.5 BETA=0.5     # SmoothKV variant sweep
# =============================================================================

SHELL := /bin/bash

# --- vllm-compression-part fork (KV fake-quant lives here) -------------------
# Pinned commit on smoothkv_exp; bump after re-verifying graph capture.
VLLM_FORK_PATH    ?= /workspace/vllm-compression-part
VLLM_FORK_BRANCH  ?= smoothkv_exp
VLLM_FORK_COMMIT  ?= e8931c812b

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
QWEN3_SIZE     ?= 8b

# Qwen3-32B requires TP=2 → default to two GPUs unless user overrides.
ifeq ($(QWEN3_SIZE),32b)
  GPUS ?= 0,1
else
  GPUS ?= 0
endif

SWEEP_DIR      := scripts

# --- Colors ------------------------------------------------------------------
# Store the actual ESC byte (not the literal "\033" string) so awk/echo/printf
# all emit the sequence directly without further escape interpretation.
GREEN  := $(shell printf '\033[0;32m')
YELLOW := $(shell printf '\033[0;33m')
CYAN   := $(shell printf '\033[0;36m')
RESET  := $(shell printf '\033[0m')

# =============================================================================

.PHONY: help status setup setup-fork test print-pin run-llama run-mistral run-qwen3 run-all clean-stamps

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
	@echo "                    LLAMA_MODEL, MISTRAL_MODEL, QWEN3_SIZE"
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
	@echo "  QWEN3_SIZE:     $(QWEN3_SIZE)"
	@echo ""

run-llama: ## Sweep Llama-family on $(LLAMA_MODEL)
	@echo "$(GREEN)>>> Llama sweep — $(LLAMA_MODEL)$(RESET)"
	@bash $(SWEEP_DIR)/sweep_llama.sh \
		--model "$(LLAMA_MODEL)" \
		--variants "$(VARIANTS)" \
		--tasks "$(TASKS)" \
		--gpus "$(GPUS)" \
		--ns $(NS) --alpha $(ALPHA) --beta $(BETA)

run-mistral: ## Sweep Mistral-family on $(MISTRAL_MODEL)
	@echo "$(GREEN)>>> Mistral sweep — $(MISTRAL_MODEL)$(RESET)"
	@bash $(SWEEP_DIR)/sweep_mistral.sh \
		--model "$(MISTRAL_MODEL)" \
		--variants "$(VARIANTS)" \
		--tasks "$(TASKS)" \
		--gpus "$(GPUS)" \
		--ns $(NS) --alpha $(ALPHA) --beta $(BETA)

run-qwen3: ## Sweep Qwen3 (size via QWEN3_SIZE=8b|32b)
	@echo "$(GREEN)>>> Qwen3-$(QWEN3_SIZE) sweep$(RESET)"
	@bash $(SWEEP_DIR)/sweep_qwen3.sh \
		--size $(QWEN3_SIZE) \
		--variants "$(VARIANTS)" \
		--tasks "$(TASKS)" \
		--gpus "$(GPUS)" \
		--ns $(NS) --alpha $(ALPHA) --beta $(BETA)

run-all: run-llama run-mistral run-qwen3 ## Run all three family sweeps sequentially

clean-stamps: ## Remove stale kivi_vllm_plugin marker files (/tmp/kivi_active_*.json)
	@rm -f /tmp/kivi_active_*.json
	@echo "$(GREEN)Cleaned /tmp/kivi_active_*.json$(RESET)"
