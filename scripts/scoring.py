"""Task-specific scorers for adaptive_pass2 merged samples.

We re-score in-process (instead of round-tripping through lm-eval's filter
chain) to avoid version-skew bugs where lm-eval upgrades silently change
extraction regexes between releases.

Each scorer takes ``items``: a list of lm-eval sample dicts (with ``doc``
and ``resps``) and returns a flat ``{metric_name: float}`` dict matching
lm-eval's own metric naming so the downstream tables can stay schema-stable.

Available scorers (looked up via ``SCORERS[task_name]``):
  - ``minerva_math500``           → math_verify (sympy boxed-aware)
  - ``gsm8k_cot`` / ``gsm8k_32k`` → strict + flexible exact-match on numbers
  - ``gpqa_main_cot_n_shot``  → flexible-extract on (A)/(B)/(C)/(D)
"""
import re


def get_raw_text(it):
    """Pull the response string out of an lm-eval sample dict.

    lm-eval stores responses as ``resps``: it can be a single string, a
    flat list of strings, or a nested ``[[str]]`` (chat-task format).
    Normalize all three into one string.
    """
    r = it.get("resps") or [""]
    if isinstance(r, list) and r and isinstance(r[0], list):
        r = r[0]
    if isinstance(r, list):
        return r[0] if r else ""
    return r if isinstance(r, str) else ""


def score_minerva_math500(items):
    """math_verify-based exact match for Minerva-MATH500.

    math_verify parses both gold and prediction with sympy and checks
    boxed-aware mathematical equivalence (so "1/2" == "0.5" etc.).
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


_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_strict_gsm8k(text: str):
    """Strict-match extraction: \\boxed{}, then "answer is X", then GSM8K-style "#### X"."""
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        nums = _NUM_RE.findall(m[-1])
        if nums:
            return nums[-1].replace(",", "")
    m = re.search(r"answer is[:\s]*\$?(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def _extract_flex_gsm8k(text: str):
    """Flexible extraction: last number anywhere in the response."""
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def score_gsm8k(items):
    """Strict + flexible exact-match for GSM8K-style numeric answers."""
    n_strict = 0
    n_flex = 0
    for it in items:
        resp = get_raw_text(it)
        gold = it["doc"].get("answer", "")
        # GSM8K gold is "... #### N" — take the last number.
        g = _NUM_RE.findall(gold)
        gold_num = g[-1].replace(",", "") if g else gold.strip()
        s = _extract_strict_gsm8k(resp)
        f = _extract_flex_gsm8k(resp)
        if s is not None and s == gold_num:
            n_strict += 1
        if f is not None and f == gold_num:
            n_flex += 1
    return {"exact_match,strict-match": n_strict / len(items),
            "exact_match,flexible-extract": n_flex / len(items),
            "exact_match_n,strict-match": len(items),
            "exact_match_n,flexible-extract": len(items)}


def score_gpqa(items):
    """Flexible-extract for GPQA: pick the last (A)/(B)/(C)/(D) letter."""
    n_flex = 0
    for it in items:
        resp = get_raw_text(it)
        gold = str(it["doc"].get("answer", "") or it["doc"].get("Correct Answer", "") or "").strip()
        # gpqa_main_cot_n_shot answers look like "(A)" "(B)" "(C)" "(D)".
        m = re.findall(r"\b\(?([A-D])\)?", resp.split("\n")[-1])
        if not m:
            m = re.findall(r"\b\(([A-D])\)", resp)
        pred = m[-1] if m else None
        gold_letter = gold[1] if gold.startswith("(") and len(gold) >= 3 else gold
        if pred is not None and pred.upper() == gold_letter.upper():
            n_flex += 1
    return {"exact_match,flexible-extract": n_flex / len(items),
            "exact_match_n,flexible-extract": len(items)}


SCORERS = {
    "minerva_math500": score_minerva_math500,
    "gsm8k_32k": score_gsm8k,
    "gsm8k_cot": score_gsm8k,
    "gpqa_main_cot_n_shot": score_gpqa,
}
