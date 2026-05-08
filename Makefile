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
#   make run-qwen3-8b VARIANTS="bf16 smkv_fused"
#   make run-llama TASKS=gsm8k_cot
#   make run-llama LLAMA_MODEL=meta-llama/Meta-Llama-3-70B-Instruct GPUS="0,1"
#   make run-qwen3-8b ALPHA=0.5 BETA=0.5  # SmoothKV variant sweep
# =============================================================================

SHELL := /bin/bash

# --- vllm-compression-part fork (KV fake-quant lives here) -------------------
# Pinned commit on kv_cache_quant; bump after re-verifying graph capture.
# `make setup` will clone the fork into VLLM_FORK_PATH if missing.
VLLM_FORK_URL     ?= https://github.com/sunghyuckhong/vllm-compression-part.git
VLLM_FORK_PATH    ?= /workspace/sunghyuck/vllm-compression-part
VLLM_FORK_BRANCH  ?= kv_cache_quant
VLLM_FORK_COMMIT  ?= d4f2eeb3b

# Torch version pinned by the vllm fork's pyproject.toml. We install torch
# from the PyTorch CUDA-12.8 / CUDA-13.0 wheel index *before* the vllm fork
# install so pip doesn't fall back to whatever PyPI ships by default
# (which mismatches drivers older than the cu130 minimum). The index URL is
# picked at runtime based on `nvidia-smi` driver major version (see setup).
TORCH_VERSION       ?= 2.11.0
TORCHVISION_VERSION ?= 0.26.0
# Min driver major version compatible with cu128. cu130 needs >= 575.
MIN_DRIVER_MAJOR    ?= 555

VENV       ?= .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip

# Export PY so subprocesses (parallel_sweep.py → run_eval_*.sh) inherit the
# venv built by `make setup`. The runners default to ./.venv/bin/python if
# PY is unset, but that only works when KIVI is run from /workspace/KIVI.
# Exporting makes any cwd work and lets `PY=... make run-...` override it.
export PY

# --- Defaults (override on command line or via env) --------------------------
VARIANTS       ?= bf16 fp8 pertoken smkv_fused
TASKS          ?= gsm8k_cot minerva_math500 gpqa_main_cot_n_shot
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

.PHONY: help status setup setup-fork setup-exaone-4.5 test print-pin \
        run-qwen3-8b run-qwen3-32b run-qwen3-30b-a3b run-llama run-mistral \
        run-exaone run-all

# --- Environment setup -------------------------------------------------------
setup: $(VENV)/bin/activate ## One-time: build vllm fork + venv + install KIVI deps
	@if [ ! -d $(VLLM_FORK_PATH) ]; then \
	  echo "$(GREEN)>>> Clone vllm fork → $(VLLM_FORK_PATH) (from $(VLLM_FORK_URL), branch $(VLLM_FORK_BRANCH))$(RESET)"; \
	  git clone --branch $(VLLM_FORK_BRANCH) $(VLLM_FORK_URL) $(VLLM_FORK_PATH); \
	else \
	  echo "$(YELLOW)[setup] vllm fork already at $(VLLM_FORK_PATH); skipping clone$(RESET)"; \
	fi
	@echo "$(GREEN)>>> Checkout vllm fork at pinned commit $(VLLM_FORK_COMMIT)$(RESET)"
	cd $(VLLM_FORK_PATH) && git fetch --all && git checkout $(VLLM_FORK_COMMIT)
	@# --- Driver-aware PyTorch install -----------------------------------
	@# nvidia-smi must succeed; otherwise this is the wrong machine.
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
	  echo "ERROR: nvidia-smi not found; cannot detect GPU driver. KIVI eval needs CUDA-capable hardware." >&2; exit 1; \
	fi
	@DRIVER=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1); \
	  DRIVER_MAJOR=$${DRIVER%%.*}; \
	  if [ "$$DRIVER_MAJOR" -ge 575 ]; then \
	    PT_INDEX=https://download.pytorch.org/whl/cu130; \
	    PT_LABEL=cu130; \
	  elif [ "$$DRIVER_MAJOR" -ge $(MIN_DRIVER_MAJOR) ]; then \
	    PT_INDEX=https://download.pytorch.org/whl/cu128; \
	    PT_LABEL=cu128; \
	  else \
	    echo "ERROR: NVIDIA driver $$DRIVER (major=$$DRIVER_MAJOR) is too old."  >&2; \
	    echo "       The vllm-compression-part fork pins torch==$(TORCH_VERSION) which needs CUDA 12.8+ (driver major >= $(MIN_DRIVER_MAJOR))." >&2; \
	    echo "       Upgrade your NVIDIA driver, or pin a different torch version via TORCH_VERSION=..." >&2; \
	    exit 1; \
	  fi; \
	  echo "$(GREEN)>>> driver=$$DRIVER → installing torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION) ($$PT_LABEL wheels)$(RESET)"; \
	  $(PIP) install -U pip wheel && \
	  $(PIP) install torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION) --index-url $$PT_INDEX
	@echo "$(GREEN)>>> Install vllm-compression-part (editable; vllm._C compile ~15-30min)$(RESET)"
	$(PIP) install -e $(VLLM_FORK_PATH)
	@echo "$(GREEN)>>> Install KIVI eval-harness deps (lm-eval, math-verify, ray)$(RESET)"
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; \
	  else echo "$(YELLOW)[setup] no requirements.txt; skipping$(RESET)"; fi
	@if [ -f setup.py ] || [ -f pyproject.toml ]; then $(PIP) install -e .; \
	  else echo "$(YELLOW)[setup] no setup.py/pyproject.toml; skipping editable KIVI install$(RESET)"; fi
	@echo "$(GREEN)[setup] DONE. Verify with 'make test'.$(RESET)"

setup-fork: ## Re-checkout the pinned commit + reinstall the fork only (faster)
	@[ -d $(VENV) ] || (echo "ERROR: run 'make setup' first" && exit 1)
	@[ -d $(VLLM_FORK_PATH) ] || (echo "ERROR: $(VLLM_FORK_PATH) missing; run 'make setup' first" && exit 1)
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
	@grep -E '^[a-zA-Z0-9_.-]+:.*## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /:## /' | \
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

run-qwen3-30b-a3b: ## Sweep Qwen3-30B-A3B MoE (TP=2, auto-parallel; e.g. 8 idle GPUs → 4 streams)
	@$(SWEEP) --tp 2 --runner scripts/run_eval_qwen3.sh --runner_args="--size 30b-a3b"

run-llama: ## Sweep $(LLAMA_MODEL) (TP=1, auto-parallel)
	@$(SWEEP) --tp 1 --runner scripts/run_eval_llama.sh --runner_args="--model $(LLAMA_MODEL)"

run-mistral: ## Sweep $(MISTRAL_MODEL) (TP=1, auto-parallel)
	@$(SWEEP) --tp 1 --runner scripts/run_eval_mistral.sh --runner_args="--model $(MISTRAL_MODEL)"

# --- EXAONE-4.5-33B with isolated .venv-exaone -------------------------------
# Uses lkm2835/vllm@add-exaone4_5 + nuxlear/transformers@add-exaone4_5-v5.3.0.dev0
# with the KV-cache fake-quant code rebased on top. Built once via
# `make setup-exaone-4.5`; the runner defaults PY=./.venv-exaone/bin/python
# but we export it here so subprocesses (parallel_sweep.py) inherit it.
VENV_EXAONE ?= .venv-exaone

setup-exaone-4.5: ## One-time: build .venv-exaone with the EXAONE forks + KV-cache fake-quant code
	bash scripts/_setup_venv_exaone.sh

run-exaone: ## Sweep EXAONE-4.5-33B via .venv-exaone (TP=2, auto-parallel)
	@PY=./$(VENV_EXAONE)/bin/python $(SWEEP) --tp 2 --runner scripts/run_eval_exaone.sh --runner_args=""

run-all: run-qwen3-8b run-llama run-mistral ## Run qwen3-8b + llama + mistral (skips 32b/exaone)
