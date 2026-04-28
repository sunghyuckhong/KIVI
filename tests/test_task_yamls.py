"""Task YAML loadability + key invariants.

A YAML can be syntactically valid but break at task-load time (missing
include, wrong dataset_name, etc). Catch those at PR time, not eval time.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_taskmanager():
    """Late-import lm_eval to allow CPU-only environments to skip cleanly."""
    lm_eval = pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager
    return TaskManager(include_path=str(REPO / "tasks"))


def _task_registered(tm, task):
    """Cross-version: tm.match_tasks(['x']) returns matching task names if found."""
    if hasattr(tm, "match_tasks"):
        matched = tm.match_tasks([task])
        return task in matched
    if hasattr(tm, "get_task_dict"):
        return task in tm.get_task_dict([task])
    if hasattr(tm, "all_tasks"):
        return task in tm.all_tasks
    raise RuntimeError("unknown TaskManager API")


@pytest.mark.parametrize("task", [
    "gsm8k_32k",
    "math500_32k",
    "gpqa_diamond_cot_n_shot_32k",
    "gpqa_main_cot_n_shot_32k",
])
def test_core_task_loads(task):
    tm = _load_taskmanager()
    assert _task_registered(tm, task), f"{task} did not register"


@pytest.mark.parametrize("task", [
    "gsm8k_32k_chat",
    "math500_32k_chat",
    "gpqa_main_cot_n_shot_32k_chat",
])
def test_chat_template_task_loads(task):
    """Chat-template variants (added in 3ab3898) must register too."""
    tm = _load_taskmanager()
    assert _task_registered(tm, task), f"chat task {task} did not register"


def test_gpqa_main_uses_main_subset():
    """Sanity: gpqa_main_cot_n_shot_32k should pull dataset_name='gpqa_main'."""
    import yaml
    p = REPO / "tasks/gpqa/gpqa_main_cot_n_shot_32k.yaml"
    if not p.exists():
        pytest.skip(f"{p} not present")
    cfg = yaml.safe_load(p.read_text())
    assert cfg.get("dataset_name") == "gpqa_main"
    assert cfg.get("task") == "gpqa_main_cot_n_shot_32k"


def test_gpqa_diamond_uses_diamond_subset():
    import yaml
    p = REPO / "tasks/gpqa/gpqa_diamond_cot_n_shot_32k.yaml"
    if not p.exists():
        pytest.skip(f"{p} not present")
    cfg = yaml.safe_load(p.read_text())
    assert cfg.get("dataset_name") == "gpqa_diamond"
