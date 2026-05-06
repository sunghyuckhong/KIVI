"""Re-score adaptive_pass2 merged samples by replaying lm-eval's own filter
chain. Reads ``filter_list`` from the task YAML, instantiates each component
via ``lm_eval.filters.get_filter``, applies them in sequence to the merged
(resps, docs), and aggregates with ``exact_match_hf_evaluate`` using the
options from ``metric_list``. The result keys (``"exact_match,strict-match"``,
``"exact_match,flexible-extract"`` etc.) match what lm-eval would have written
if it had scored the merged set itself.

Why we re-score at all: pass-1 (``run_eval_vllm.py``) goes through lm-eval's
``simple_evaluate`` and lands a filtered + scored ``_results.json``. Pass-2
re-generates only the truncated subset at MG=32k and merges those new
responses back into the pass-1 sample list — at which point lm-eval has no
public hook to "score this list of pre-generated samples." This module fills
that gap.

Why YAML instead of ``task.apply_filters``: ``ConfigurableTask`` downloads
the dataset upfront (gpqa's is gated on HF), so constructing the task is
heavy and credential-bound. Reading the YAML and instantiating filters via
``lm_eval.filters`` needs no dataset access. The merged sample dicts already
carry post-``process_docs`` ``doc`` fields (with ``choices`` etc.), so the
filters that consult docs (``MultiChoiceRegexFilter``) work as-is.

Past trap: an earlier hand-rolled regex scorer for gpqa diverged from
lm-eval's filter (~9pt loss on Qwen3 thinking-mode). Delegating to lm-eval's
own filter classes prevents that drift.
"""
from pathlib import Path

import yaml
import lm_eval
from lm_eval.filters import get_filter
from lm_eval.api.metrics import exact_match_hf_evaluate


LMEVAL_TASKS_DIR = Path(lm_eval.__file__).parent / "tasks"


# lm-eval YAMLs use a custom `!function` tag (e.g.
# `process_docs: !function utils.process_docs`) to point at Python callables.
# We don't execute those — we only need filter_list / metric_list — so a
# tolerant loader that turns `!function foo` into the string "foo" suffices.
class _LMEvalLoader(yaml.SafeLoader):
    pass


def _construct_function_tag(loader, node):
    return f"!function {node.value}"


_LMEvalLoader.add_constructor("!function", _construct_function_tag)


# Cache resolved task configs: resolving a task name walks every YAML under
# lm_eval/tasks/ (~hundreds of files) so without caching, scoring N cells does
# N full walks.
_CFG_CACHE: dict[str, dict] = {}


def get_raw_text(it):
    """Pull the response string out of an lm-eval sample dict.

    lm-eval stores responses as ``resps``: a single string, a flat list, or a
    nested ``[[str]]`` (chat-task format). Normalize all three into one string.
    Kept as a public helper for ``score_minerva_math500``.
    """
    r = it.get("resps") or [""]
    if isinstance(r, list) and r and isinstance(r[0], list):
        r = r[0]
    if isinstance(r, list):
        return r[0] if r else ""
    return r if isinstance(r, str) else ""


def _find_task_yaml(task_name):
    for p in LMEVAL_TASKS_DIR.rglob("*.yaml"):
        try:
            cfg = yaml.load(open(p), Loader=_LMEvalLoader)
        except Exception:
            continue
        if isinstance(cfg, dict) and cfg.get("task") == task_name:
            return p
    raise FileNotFoundError(f"no lm-eval YAML for task {task_name!r} under {LMEVAL_TASKS_DIR}")


def _load_full_config(yaml_path):
    """Load a task YAML and recursively merge any ``include:`` directives.

    lm-eval allows includes without a .yaml extension (e.g. gpqa's
    ``include: _gpqa_cot_n_shot_yaml``)."""
    cfg = yaml.load(open(yaml_path), Loader=_LMEvalLoader)
    inc = cfg.pop("include", None)
    if not inc:
        return cfg
    parent_dir = Path(yaml_path).parent
    candidates = [parent_dir / inc, parent_dir / f"{inc}.yaml", parent_dir / f"{inc}.yml"]
    for c in candidates:
        if c.exists():
            base = _load_full_config(c)
            base.update(cfg)
            return base
    raise FileNotFoundError(f"include {inc!r} (referenced from {yaml_path}) not found")


def _build_filter_chain(filter_list_entry):
    """Turn one ``filter_list`` entry (``{name, filter: [components]}``) into
    a callable ``f(resps, docs) -> filtered_resps``."""
    components = []
    for component_cfg in filter_list_entry["filter"]:
        kw = dict(component_cfg)
        fn = kw.pop("function")
        components.append(get_filter(fn)(**kw))

    def chain(resps, docs):
        for f in components:
            resps = f.apply(resps, docs)
        return resps

    return chain


def _resps_for(it):
    r = it.get("resps") or [""]
    if isinstance(r, list) and r and isinstance(r[0], list):
        return list(r[0])
    return list(r) if isinstance(r, list) else [r]


def _target_for(it, task_cfg):
    """Pass-1 populates ``target``; fall back to the YAML's doc_to_target only
    for the trivial "answer" form so we don't reimplement Jinja here."""
    t = it.get("target")
    if t:
        return str(t)
    d2t = task_cfg.get("doc_to_target")
    if d2t == "answer":
        return str(it["doc"].get("answer", ""))
    raise ValueError(
        f"sample lacks `target` and doc_to_target={d2t!r} is not a literal field "
        f"this helper handles; pass-1 should have populated `target`."
    )


def score_via_lm_eval(items, task_name):
    """Apply the lm-eval task's filter chain + metric to merged samples.

    Returns a dict shaped like lm-eval's own results: one ``{metric},{filter}``
    entry per filter in ``filter_list``, plus an ``{metric}_n,{filter}`` count.
    """
    if task_name not in _CFG_CACHE:
        _CFG_CACHE[task_name] = _load_full_config(_find_task_yaml(task_name))
    cfg = _CFG_CACHE[task_name]
    metric_cfg = cfg["metric_list"][0]  # the three tasks we care about are single-metric
    metric_name = metric_cfg["metric"]
    em_kwargs = {
        "ignore_case":        metric_cfg.get("ignore_case", False),
        "ignore_punctuation": metric_cfg.get("ignore_punctuation", False),
        "ignore_numbers":     metric_cfg.get("ignore_numbers", False),
        "regexes_to_ignore":  metric_cfg.get("regexes_to_ignore"),
    }

    resps = [_resps_for(it) for it in items]
    docs = [it["doc"] for it in items]
    targets = [_target_for(it, cfg) for it in items]

    out = {}
    for entry in cfg["filter_list"]:
        fname = entry["name"]
        chain = _build_filter_chain(entry)
        # After RegexFilter the shape is List[List[str]]; after TakeFirstFilter
        # it's a `map` object yielding str — materialize and accept either form.
        filtered = list(chain([list(r) for r in resps], docs))
        preds = [(f if isinstance(f, str) else (f[0] if isinstance(f, list) and f else "")) for f in filtered]
        em = exact_match_hf_evaluate(predictions=preds, references=targets, **em_kwargs)
        out[f"{metric_name},{fname}"] = float(em["exact_match"])
        out[f"{metric_name}_n,{fname}"] = len(items)
    return out


def score_minerva_math500(items):
    """math_verify-based exact match for Minerva-MATH500.

    math_verify parses both gold and prediction with sympy and checks
    boxed-aware mathematical equivalence (so "1/2" == "0.5" etc.). lm-eval's
    minerva task uses the same `math_verify` package for the same purpose, so
    this is delegation-equivalent without the dataset-download cost of going
    through ConfigurableTask.
    """
    from math_verify import parse, verify
    n_correct = 0
    for it in items:
        resp = get_raw_text(it)
        sol = it["doc"].get("solution", "")
        try:
            ok = bool(verify(gold=parse(sol), target=parse(resp)))
        except Exception:
            ok = False
        if ok:
            n_correct += 1
    return {"math_verify,none": n_correct / len(items),
            "math_verify_n,none": len(items)}


def score_gsm8k(items):
    return score_via_lm_eval(items, "gsm8k_cot")


def score_gpqa(items):
    return score_via_lm_eval(items, "gpqa_main_cot_n_shot")


SCORERS = {
    "minerva_math500":      score_minerva_math500,
    "gsm8k_32k":            score_gsm8k,
    "gsm8k_cot":            score_gsm8k,
    "gpqa_main_cot_n_shot": score_gpqa,
}
