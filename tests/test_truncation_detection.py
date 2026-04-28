"""Truncation detection logic shared by retry_truncated.py and adaptive rerun.

The "is this item truncated?" predicate decides which items get re-generated
at higher max_gen_toks. A bug here would either silently drop items (under-
counting truncation, scores stay biased low) or rerun everything (wasting GPU).
"""
import torch


def detect_truncated_doc_ids(items, tokenizer, first_pass_mg, slack=16):
    """Reference predicate.

    items: list of {"doc_id": int, "resps": [[str]], ...}  (lm_eval samples format)
    Returns sorted list of doc_ids whose generation length >= first_pass_mg - slack.
    """
    out = []
    for it in items:
        gen = it["resps"][0][0]
        n = len(tokenizer(gen, add_special_tokens=False)["input_ids"])
        if n >= first_pass_mg - slack:
            out.append(it["doc_id"])
    return sorted(out)


class _FakeTokenizer:
    """Tokenizer that returns N tokens for a string of length N (one token per char).

    Lets us test truncation logic without loading a real model.
    """
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def _mk_item(doc_id, gen_len):
    return {"doc_id": doc_id, "resps": [["x" * gen_len]]}


def test_short_items_not_flagged():
    items = [_mk_item(i, 100) for i in range(5)]
    flagged = detect_truncated_doc_ids(items, _FakeTokenizer(), first_pass_mg=4096)
    assert flagged == []


def test_items_at_cap_flagged():
    items = [_mk_item(i, 4096) for i in range(3)]
    flagged = detect_truncated_doc_ids(items, _FakeTokenizer(), first_pass_mg=4096)
    assert flagged == [0, 1, 2]


def test_slack_window_catches_near_cap():
    """vLLM sometimes stops 1-2 tokens early; slack=16 should still catch them."""
    items = [_mk_item(0, 4080), _mk_item(1, 4090), _mk_item(2, 100)]
    flagged = detect_truncated_doc_ids(items, _FakeTokenizer(), first_pass_mg=4096, slack=16)
    assert flagged == [0, 1]   # 4080 and 4090 within slack; 100 is short


def test_slack_zero_strict_match():
    items = [_mk_item(0, 4096), _mk_item(1, 4080), _mk_item(2, 100)]
    flagged = detect_truncated_doc_ids(items, _FakeTokenizer(), first_pass_mg=4096, slack=0)
    assert flagged == [0]


def test_dedup_before_count():
    """Some lm_eval tasks (gpqa) emit two sample entries per doc (one per filter).
    Dedup by doc_id should happen BEFORE truncation counting so we don't double-count.
    """
    items = [_mk_item(0, 4096), _mk_item(0, 4096),  # dupe of doc 0 (two filters)
             _mk_item(1, 100),  _mk_item(1, 100)]
    seen = {}
    for it in items:
        seen.setdefault(it["doc_id"], it)
    deduped = list(seen.values())
    flagged = detect_truncated_doc_ids(deduped, _FakeTokenizer(), first_pass_mg=4096)
    assert flagged == [0]   # doc 0 once, not twice
