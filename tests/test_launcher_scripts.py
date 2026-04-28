"""Bash syntax checks + per-task max_model_len sanity for the launcher.

These don't actually run vLLM — they just verify the launcher doesn't have
syntax errors and the per-task prompt-length table is internally consistent.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "scripts/launch_vllm_reasoning.sh"


def test_launcher_bash_syntax():
    if not LAUNCHER.exists():
        pytest.skip(f"{LAUNCHER} not present")
    r = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed:\n{r.stderr}"


def test_launcher_task_max_len_known_tasks():
    """task_max_len() must have entries for all *_32k tasks we run."""
    if not LAUNCHER.exists():
        pytest.skip()
    text = LAUNCHER.read_text()
    expected = [
        "gsm8k_32k", "math500_32k",
        "gpqa_diamond_cot_n_shot_32k", "gpqa_main_cot_n_shot_32k",
    ]
    for task in expected:
        # match e.g.   gsm8k_32k)                    prompt=1404 ;;
        m = re.search(rf"\b{re.escape(task)}\)\s+prompt=(\d+)", text)
        assert m, f"task_max_len() missing entry for {task}"
        n = int(m.group(1))
        assert 0 < n < 32_000, f"{task} prompt={n} looks wrong"


def test_other_helper_scripts_bash_syntax():
    """All other .sh scripts in scripts/ should at least parse cleanly."""
    sh_dir = REPO / "scripts"
    if not sh_dir.exists():
        pytest.skip()
    for sh in sh_dir.glob("*.sh"):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh.name}: {r.stderr}"
