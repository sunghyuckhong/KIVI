"""Adaptive retry pass.

Reads an lm_eval samples.json (from `run_eval_vllm.py --log_samples`), finds
items whose first-pass generation hit the max_gen_toks cap, regenerates just
those at a higher MG, then re-applies the task's filter chain + process_results
on every item to score the merged set.

The scoring matches lm_eval's first-pass output keys exactly:
    {metric},{filter_name}        (e.g. exact_match,strict-match)
    {metric}_n,{filter_name}      (count for that filter)

For tasks with no filter_list (math500_32k, minerva_math500), filter_name=='none'.

Two modes:
  default — regenerate truncated items via vLLM, then score the merged set
  --from_merged_samples FILE — skip generation; rescore an already-merged file

Output files (next to the input samples, with _retry{retry_mg}_ infix):
    *_retry{retry_mg}_results.json
    *_retry{retry_mg}_merged_samples.json

Usage:
    python retry_truncated.py \
        --samples logs/gpqa_main_cot_n_shot_32k_qwen3-8b_bf16_chat_vllm_samples.json \
        --model_path Qwen/Qwen3-8B --task gpqa_main_cot_n_shot_32k --model bf16 \
        --orig_mg 4096 --retry_mg 32768 \
        --max_num_seqs 8 --max_model_len 35584 --tp 1
"""
import argparse
import copy
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")


# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=None,
                   help="path to *_samples.json from first pass. Required in generation mode; "
                        "optional in --from_merged_samples mode.")
    p.add_argument("--task", required=True)
    p.add_argument("--from_merged_samples", default=None,
                   help="Skip vLLM generation; load this merged_samples file and rescore it. "
                        "Useful when only the scoring step changed.")
    # vLLM model spec (only required when generating)
    p.add_argument("--model_path", default="Qwen/Qwen3-8B")
    p.add_argument("--model", choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv",
                                        "smoothkv_fused"], default="bf16")
    p.add_argument("--orig_mg", type=int, default=4096,
                   help="max_gen_toks used in first pass (truncation threshold)")
    p.add_argument("--retry_mg", type=int, default=32768,
                   help="max_gen_toks for retry pass (must be > orig_mg)")
    p.add_argument("--calib_path", default=None,
                   help="required for --model smoothkv / smoothkv_fused")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    return p.parse_args()


# ---------- KV fake-quant config ----------
def build_kv_quant_config(args):
    """Return a KVCacheQuantConfig for the requested method, or None for
    bf16/fp16. Pass to LLM(...) via kv_cache_quant_config=cfg."""
    if args.model in ("bf16", "fp16"):
        return None
    from vllm.config import KVCacheQuantConfig
    if args.model == "fp8":
        return KVCacheQuantConfig(method="fp8", group_size=args.group_size)
    if args.model == "pertoken":
        return KVCacheQuantConfig(method="pertoken", group_size=args.group_size,
                                  bits=args.bits)
    if args.model == "smoothkv":
        assert args.calib_path, "--calib_path required for smoothkv"
        return KVCacheQuantConfig(method="smoothkv", group_size=args.group_size,
                                  bits=args.bits, calib_path=args.calib_path)
    if args.model == "smoothkv_fused":
        assert args.calib_path, "--calib_path required for smoothkv_fused"
        return KVCacheQuantConfig(method="smoothkv_fused",
                                  group_size=args.group_size, bits=args.bits,
                                  calib_path=args.calib_path)
    raise ValueError(f"unknown --model {args.model}")


# ---------- truncation detection ----------
def _extract_raw_text(it):
    r = it.get("resps") or []
    if isinstance(r, list):
        r = r[0] if r else ""
    if isinstance(r, list):
        r = r[0] if r else ""
    return str(r)


def find_truncated(items, orig_mg, tokenizer):
    """Return indices of items whose raw-resp token count is within 8 of MG cap.
    Use raw `resps`, NOT `filtered_resps` (the latter is the post-filter extracted
    answer like '(A)' for gpqa, unrelated to whether MG cap was hit)."""
    out = []
    for i, it in enumerate(items):
        n = len(tokenizer.encode(_extract_raw_text(it), add_special_tokens=False))
        if n >= orig_mg - 8:
            out.append(i)
    return out


# ---------- generation ----------
def generate_retries(args, items, truncated_idx, kv_quant_cfg=None):
    """Run vLLM on the truncated subset's prompts at retry_mg. Returns list of new texts."""
    import ast
    import vllm
    from vllm import LLM, SamplingParams

    retry_prompts, retry_gen_kwargs = [], []
    for i in truncated_idx:
        it = items[i]
        arg = it.get("arguments")
        if isinstance(arg, str):
            try: arg = ast.literal_eval(arg)
            except Exception: pass
        if isinstance(arg, (list, tuple)) and arg:
            retry_prompts.append(str(arg[0]))
            retry_gen_kwargs.append(arg[1] if len(arg) > 1 and isinstance(arg[1], dict) else {})
        else:
            retry_prompts.append(str(arg))
            retry_gen_kwargs.append({})

    # Cudagraphs default ON for vllm 0.20+; keep eager fallback for older builds.
    is_old_vllm = tuple(map(int, vllm.__version__.split(".")[:2])) < (0, 20)
    if os.environ.get("FORCE_ENFORCE_EAGER"):
        enforce_eager = True
    elif os.environ.get("NO_ENFORCE_EAGER"):
        enforce_eager = False
    else:
        enforce_eager = is_old_vllm and args.model not in ("bf16", "fp16")

    llm_kwargs = dict(
        model=args.model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if "EXAONE-4.5" in args.model_path:
        # Multimodal wrapper crashes mm-budget profiling without the (missing) video processor
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if kv_quant_cfg is not None:
        llm_kwargs["kv_cache_quant_config"] = kv_quant_cfg

    llm = LLM(**llm_kwargs)
    gk = retry_gen_kwargs[0] if retry_gen_kwargs else {}
    sp = SamplingParams(
        temperature=float(gk.get("temperature", 0.0)),
        top_p=1.0,
        max_tokens=args.retry_mg,
        stop=gk.get("until", None),
    )
    print(f"[retry] launching vLLM on {len(retry_prompts)} prompts at MG={args.retry_mg}")
    outputs = llm.generate(retry_prompts, sampling_params=sp, use_tqdm=True)
    assert len(outputs) == len(retry_prompts)
    return [out.outputs[0].text for out in outputs]


# ---------- merge & score ----------
def _set_resp_field(item, key, new_text):
    r = item.get(key)
    if isinstance(r, list):
        if r and isinstance(r[0], list):
            item[key] = [[new_text]]
        else:
            item[key] = [new_text]
    else:
        item[key] = new_text


def merge_retries(items, truncated_idx, new_texts):
    """Patch raw `resps` for truncated items. Leave `filtered_resps` alone — the
    filter chain will be re-applied per-item during scoring."""
    merged = copy.deepcopy(items)
    for idx, txt in zip(truncated_idx, new_texts):
        _set_resp_field(merged[idx], "resps", txt)
        # Drop stale filtered_resps so the scorer is forced to re-derive from resps.
        merged[idx].pop("filtered_resps", None)
    return merged


def _build_filter_chain(steps):
    """Instantiate lm_eval filter classes from a task config's filter_list step list."""
    from lm_eval.filters.extraction import RegexFilter, MultiChoiceRegexFilter
    from lm_eval.filters.selection import TakeFirstFilter
    chain = []
    for step in steps:
        kind = step["function"]
        kwargs = {k: v for k, v in step.items() if k != "function"}
        if kind == "regex":
            chain.append(RegexFilter(**kwargs))
        elif kind == "multi_choice_regex":
            chain.append(MultiChoiceRegexFilter(**kwargs))
        elif kind == "take_first":
            chain.append(TakeFirstFilter(**kwargs))
        else:
            raise RuntimeError(f"unknown filter function: {kind}")
    return chain


def _materialize(x):
    if isinstance(x, str): return x
    try: return [_materialize(e) for e in x]
    except TypeError: return x


def _filter_specs_from_task(task_obj):
    """Return list of (filter_name, list_of_filter_steps). Empty if no filter_list."""
    fl = getattr(task_obj._config, "filter_list", None)
    if not fl:
        return []
    out = []
    for f in fl:
        name = f["name"] if isinstance(f, dict) else f.name
        steps = f["filter"] if isinstance(f, dict) else f.filter
        out.append((name, steps))
    return out


def score(merged, task, task_manager_include_path,
          mg_cap=None, tokenizer=None):
    """Re-apply each filter chain to every item's raw resps, then call
    task.process_results to derive {metric, filter_name} -> per-item scores.

    Output keys mirror lm_eval's first-pass aggregation: e.g.
        exact_match,strict-match     and    exact_match,flexible-extract
    For tasks with no filter_list, the suffix is ',none'.

    If mg_cap is set and tokenizer is provided, items whose raw resps reach
    `mg_cap - 8` tokens count as TRUNCATED → score 0 for every filter (the
    model didn't finish; flex regexes can luckily pick up tokens from
    mid-reasoning, breaking monotonicity in MG).
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    tm = TaskManager(include_path=task_manager_include_path)
    task_obj = get_task_dict([task], task_manager=tm)[task]
    if hasattr(task_obj, "task"):
        task_obj = task_obj.task

    filter_specs = _filter_specs_from_task(task_obj)

    # Build chains once; lm_eval filter ops operate on [Instance-like list], we
    # mimic that with [[raw_text_per_doc]].
    chains = [(name, _build_filter_chain(steps)) for name, steps in filter_specs]
    # Tasks with no filter_list: still produce one (filter_name='none') row,
    # with the raw text passed straight through.
    if not chains:
        chains = [("none", [])]

    # Dedup by doc_id: lm_eval writes one row per (doc, filter), but the raw `resps`
    # is identical across filter rows of the same doc. Score each unique doc once
    # against each filter chain → count per filter equals the dataset size.
    by_doc = {}
    for it in merged:
        did = it.get("doc_id")
        if did not in by_doc:
            by_doc[did] = it

    per = defaultdict(list)   # key: (metric, filter_name) -> list[score]
    n_truncated = 0
    for it in by_doc.values():
        doc = it.get("doc")
        raw = _extract_raw_text(it)
        # Truncation gate
        is_truncated = False
        if mg_cap is not None and tokenizer is not None:
            ntok = len(tokenizer.encode(raw, add_special_tokens=False))
            if ntok >= mg_cap - 8:
                is_truncated = True
                n_truncated += 1
        if is_truncated:
            for fname, _ in chains:
                per[("exact_match", fname)].append(0.0)
            continue
        for fname, chain in chains:
            # Apply filter chain to a single-doc batch [[text]]. Filter ops
            # walk the outer list = docs, inner list = generations per doc.
            x = [[raw]]
            for f in chain:
                x = _materialize(f.apply(x, [doc]))
            # x[0] is the post-filter list for this doc; lm_eval's process_results
            # expects a list of resps (one per filter typically). For per-filter
            # scoring we pass the single filtered string.
            cand = x[0][0] if (x and isinstance(x[0], list) and x[0]) else (
                   x[0] if x and not isinstance(x[0], list) else "")
            try:
                res = task_obj.process_results(doc, [cand])
            except Exception as e:
                print(f"[retry] process_results failed for doc_id={it.get('doc_id')} filter={fname}: {e}")
                continue
            # The task may emit metric keys with or without filter suffix. Use the
            # *bare* metric name; the filter_name comes from our outer loop.
            for mk, mv in res.items():
                bare = mk.split(",")[0]
                per[(bare, fname)].append(float(mv))
    return per


def main():
    args = parse_args()

    if args.from_merged_samples is None:
        if not args.samples:
            raise SystemExit("--samples is required in generation mode (omit only with --from_merged_samples)")
        # Generation path: build the KV-quant config first; LLM(...) below
        # gets it via kv_cache_quant_config=...
        kv_quant_cfg = build_kv_quant_config(args)

        with open(args.samples) as f:
            sdata = json.load(f)
        task_name = list(sdata.keys())[0]
        items = sdata[task_name]
        print(f"[retry] loaded {len(items)} items from {args.samples}  task={task_name}")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        truncated_idx = find_truncated(items, args.orig_mg, tok)
        print(f"[retry] {len(truncated_idx)}/{len(items)} truncated at MG={args.orig_mg} "
              f"({len(truncated_idx)/len(items)*100:.1f}%)")

        if not truncated_idx:
            print("[retry] nothing to rerun — first-pass result stands; only rescoring.")
            merged = copy.deepcopy(items)
        else:
            new_texts = generate_retries(args, items, truncated_idx, kv_quant_cfg)
            merged = merge_retries(items, truncated_idx, new_texts)

        # Save merged samples first so a scoring crash doesn't lose generation work
        merged_path = args.samples.replace("_samples.json", f"_retry{args.retry_mg}_merged_samples.json")
        with open(merged_path, "w") as f:
            json.dump({task_name: merged}, f, indent=2, default=str)
        print(f"[retry] wrote {merged_path}")
    else:
        # Rescore-only path
        with open(args.from_merged_samples) as f:
            sdata = json.load(f)
        task_name = list(sdata.keys())[0]
        merged = sdata[task_name]
        merged_path = args.from_merged_samples
        print(f"[retry] rescore-only: loaded {len(merged)} merged items from {merged_path}")

    # Score: per-(metric, filter) breakout matching lm_eval's first-pass keys.
    # Apply truncation gate at retry_mg: items still hitting the cap count as 0
    # for every filter (the model didn't finish; flex regex luck shouldn't count).
    include_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    per = score(merged, args.task, include_path, mg_cap=args.retry_mg, tokenizer=tok)

    final = {task_name: {}}
    for (metric, fname), vs in sorted(per.items()):
        key_score = f"{metric},{fname}"
        key_count = f"{metric}_n,{fname}"
        final[task_name][key_score] = sum(vs) / max(len(vs), 1)
        final[task_name][key_count] = len(vs)

    print(f"[retry] RETRY-MG={args.retry_mg} RESULTS:")
    for k, v in final[task_name].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Output paths derived from input samples (or from merged file in rescore-only mode)
    base = args.samples if args.from_merged_samples is None else args.from_merged_samples
    # Normalize: turn either '*_samples.json' or '*_retryN_merged_samples.json' into
    # the bare first-pass stem so the output is just '*_retryN_results.json' (no doubling).
    base = base.replace(f"_retry{args.retry_mg}_merged_samples.json", "_samples.json")
    base = base.replace("_merged_samples.json", "_samples.json")
    out_path = base.replace("_samples.json", f"_retry{args.retry_mg}_results.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"[retry] wrote {out_path}")


if __name__ == "__main__":
    main()
