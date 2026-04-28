"""Shared pytest fixtures + helpers."""
import sys
from pathlib import Path

import pytest

# Make repo root importable so tests can `from vllm_custom...`, `from quant...`, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# Mark tests that require a GPU; they're skipped on CPU-only boxes.
requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
