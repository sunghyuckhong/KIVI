"""Output filename canonicalization — collisions silently overwrite results."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Skip if vllm + the attention modules patches.py imports aren't available.
pytest.importorskip("vllm")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def out_name():
    """Lazy-import output_name(). Skips if patches.py can't import its
    architecture targets (e.g. qwen3 missing on older vLLM)."""
    try:
        run_eval_vllm = importlib.import_module("run_eval_vllm")
    except ImportError as e:
        pytest.skip(f"run_eval_vllm not importable in this env: {e}")
    return run_eval_vllm.output_name


def _args(**kw):
    defaults = dict(
        model_path="test/model", model="bf16",
        group_size=128, bits=4, calib_path=None, task="math500_32k",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_baseline_methods_distinguishable(out_name):
    """bf16, fp16, fp8, pertoken, kivi all produce distinct names."""
    methods = ["bf16", "fp16", "fp8", "pertoken", "kivi"]
    names = {m: out_name(_args(model=m)) for m in methods}
    assert len(set(names.values())) == len(methods), names


def test_group_size_appears_in_name(out_name):
    """Different group sizes must produce different names (otherwise FP8 g=32 overwrites g=128)."""
    n32 = out_name(_args(model="fp8", group_size=32))
    n128 = out_name(_args(model="fp8", group_size=128))
    assert n32 != n128
    assert "g32" in n32 or "g128" in n128


def test_smoothkv_calib_variants_distinguishable(tmp_path, out_name):
    """Different SmoothKV calib paths (puremax / qk / wikitext) → different output names."""
    fnames = [
        "smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_puremax_a1b1_pair.pt",
        "smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_qk_a0.75b1_pair.pt",
        "smoothkv_meta-llama-3-8b-instruct_bf16_wikitext_perc_ns512_puremax_a1b1_pair.pt",
        "smoothkv_meta-llama-3-8b-instruct_bf16_numina_perc_ns512_puremax_a1b1_pair.pt",
    ]
    names = []
    for f in fnames:
        p = tmp_path / f
        p.write_text("")
        names.append(out_name(_args(
            model="smoothkv",
            calib_path=str(p),
            model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        )))
    assert len(set(names)) == len(fnames), \
        f"calib variants collided in output names: {names}"


def test_task_appears_in_name(out_name):
    n_math = out_name(_args(task="math500_32k"))
    n_gsm = out_name(_args(task="gsm8k_32k"))
    assert n_math != n_gsm
    assert "math500" in n_math
    assert "gsm8k" in n_gsm


def test_ends_with_vllm_suffix(out_name):
    """All vLLM outputs must end with `_vllm` to distinguish from HFLM (run_eval.py) outputs."""
    assert out_name(_args(model="bf16")).endswith("_vllm")
    assert out_name(_args(model="fp8")).endswith("_vllm")
