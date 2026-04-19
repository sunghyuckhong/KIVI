# Paper-env lm-eval patches

The paper-env `lm-eval` (commit `c9bbec6e`) is missing a few things needed for GPQA-Diamond + MATH500. After installing the editable lm-eval via:

```
pip install -e "git+https://github.com/EleutherAI/lm-evaluation-harness.git@c9bbec6e7de418b9082379da82797522eb173054#egg=lm_eval"
```

apply the following patches to `src/lm-eval/lm_eval/` (the install path).

## 1. `filters/extraction.py` — add `MultiChoiceRegexFilter` + accept `group_select` in `RegexFilter`

Copy `patches/lm_eval_extraction.py` over `src/lm-eval/lm_eval/filters/extraction.py`.

## 2. `filters/__init__.py` — register `multi_choice_regex`

Add to `FILTER_REGISTRY`:
```python
"multi_choice_regex": extraction.MultiChoiceRegexFilter,
```

## 3. Copy GPQA task directory

```
cp -r /opt/modernenv/lib/python3.10/site-packages/lm_eval/tasks/gpqa  src/lm-eval/lm_eval/tasks/
```

(or from any lm-eval ≥ 0.4.2 install)

## 4. Install antlr4 dep (MATH500 needs it for latex parsing)

```
pip install antlr4-python3-runtime==4.11.0
```

## 5. Custom MATH500 task

Live in `tasks/math500/` (in this repo, not `src/lm-eval`). Included via `include_path("tasks/math500")` in `run_lm_eval_harness.py` and `run_eval.py`.
