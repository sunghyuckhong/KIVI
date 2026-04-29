"""Adaptive retry pass.

Reads a completed lm_eval samples.json (produced by run_eval_vllm.py with
--log_samples), identifies items whose generation hit the max_gen_toks cap,
re-runs just those items at a larger max_gen_toks, re-applies the task's
process_results/filters, and writes a merged *_results.json.

Usage:
    python retry_truncated.py \
        --samples logs/math500_32k_qwen3-8b_bf16_vllm_samples.json \
        --model_path Qwen/Qwen3-8B \
        --task math500_32k \
        --model bf16 \
        --orig_mg 4096 --retry_mg 16384 \
        --max_num_seqs 64 --max_model_len 17920

Writes `<samples_stem>_retry{retry_mg}_results.json` next to the input samples.
"""
import argparse
import copy
import json
import os
import warnings
warnings.filterwarnings("ignore")

import vllm  # noqa: F401
# Eagerly import attention modules so vllm_custom.patches can monkey-patch them.
# qwen3 is the canonical target; exaone4 only ships in the lkm2835 vllm fork.
try:
    import vllm.model_executor.models.qwen3  # noqa: F401
except ImportError:
    pass
try:
    import vllm.model_executor.models.exaone4  # noqa: F401
except ImportError:
    pass
from vllm_custom import patches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="path to *_samples.json from first pass")
    p.add_argument("--model_path", default="Qwen/Qwen3-8B")
    p.add_argument("--model", choices=["bf16", "fp16", "fp8", "pertoken", "smoothkv", "kivi"],
                   required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--orig_mg", type=int, required=True,
                   help="max_gen_toks used in first pass (truncation threshold)")
    p.add_argument("--retry_mg", type=int, required=True,
                   help="max_gen_toks for retry pass (should be > orig_mg)")
    p.add_argument("--calib_path", default=None, help="required for --model smoothkv")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    return p.parse_args()


def install_method(args):
    if args.model in ("bf16", "fp16"):
        return
    if args.model == "fp8":
        patches.install_fp8(group_size=args.group_size)
    elif args.model == "pertoken":
        patches.install_pertoken_int4(group_size=args.group_size)
    elif args.model == "smoothkv":
        assert args.calib_path
        patches.install_smoothkv(args.calib_path, group_size=args.group_size, bits=args.bits)
    elif args.model == "kivi":
        patches.install_kivi2(group_size=32, residual=128)


def main():
    args = parse_args()
    install_method(args)

    # lm_eval imports after patch so the engine picks up the hook
    from lm_eval.tasks import TaskManager, get_task_dict
    from transformers import AutoTokenizer

    with open(args.samples) as f:
        sdata = json.load(f)
    task_name = list(sdata.keys())[0]
    items = sdata[task_name]
    print(f"[retry] loaded {len(items)} items from {args.samples} task={task_name}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Find truncated items (within 8 tokens of cap). Must look at `resps`
    # (raw generation) NOT `filtered_resps` — the latter is the post-filter
    # extracted answer (e.g., "(A)" for gpqa) and is unrelated to whether
    # the model hit the MG cap. Looking at filtered_resps under-counts
    # truncations by ~360x on gpqa.
    def _extract_raw_gen_text(it):
        r = it.get("resps") or []
        if isinstance(r, list):
            r = r[0] if r else ""
        if isinstance(r, list):
            r = r[0] if r else ""
        return str(r)

    truncated_idx = []
    for i, it in enumerate(items):
        txt = _extract_raw_gen_text(it)
        n_tok = len(tok.encode(txt, add_special_tokens=False))
        if n_tok >= args.orig_mg - 8:
            truncated_idx.append(i)
    print(f"[retry] {len(truncated_idx)}/{len(items)} truncated at MG={args.orig_mg} ({len(truncated_idx)/len(items)*100:.1f}%)")
    if not truncated_idx:
        print("[retry] nothing to rerun — first-pass result is final")
        return

    # Recover each truncated item's prompt + generation kwargs.
    # lm_eval writes samples.json via json.dump(..., default=str). The
    # `arguments` field is a tuple (prompt_str, gen_kwargs_dict) that gets
    # stringified, so we parse it back with ast.literal_eval.
    import ast
    retry_prompts = []
    retry_gen_kwargs = []
    for i in truncated_idx:
        it = items[i]
        arg = it.get("arguments")
        if isinstance(arg, str):
            try:
                arg = ast.literal_eval(arg)
            except Exception:
                pass
        if isinstance(arg, (list, tuple)) and arg:
            prompt = arg[0]
            gkw = arg[1] if len(arg) > 1 else {}
        else:
            prompt = arg
            gkw = {}
        retry_prompts.append(str(prompt))
        retry_gen_kwargs.append(gkw if isinstance(gkw, dict) else {})

    # Run vLLM on just those prompts at retry_mg
    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=args.model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=(
            True if os.environ.get("FORCE_ENFORCE_EAGER")
            else False if os.environ.get("NO_ENFORCE_EAGER")
            else (tuple(map(int, vllm.__version__.split(".")[:2])) < (0, 20)
                  and args.model not in ("bf16", "fp16"))
        ),
        enable_prefix_caching=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    # EXAONE-4.5 multimodal wrapper crashes mm-budget profiling at engine init
    # (the nuxlear/transformers fork ships no Exaone4_5_VideoProcessor). We only
    # do text inference here, so disable mm.
    if "EXAONE-4.5" in args.model_path:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**llm_kwargs)

    # All items in our sample set have the same gen kwargs (same task). Use the first.
    sample_gk = retry_gen_kwargs[0] if retry_gen_kwargs else {}
    sp = SamplingParams(
        temperature=float(sample_gk.get("temperature", 0.0)),
        top_p=1.0,
        max_tokens=args.retry_mg,
        stop=sample_gk.get("until", None),
    )
    print(f"[retry] launching vLLM on {len(retry_prompts)} prompts at MG={args.retry_mg}")
    outputs = llm.generate(retry_prompts, sampling_params=sp, use_tqdm=True)
    assert len(outputs) == len(retry_prompts)

    # Patch the samples with the new generations
    merged = copy.deepcopy(items)
    for idx, out in zip(truncated_idx, outputs):
        new_text = out.outputs[0].text
        # Preserve structure: resps and filtered_resps are both lists of [str]
        for key in ("resps", "filtered_resps"):
            r = merged[idx].get(key)
            if isinstance(r, list):
                if r and isinstance(r[0], list):
                    merged[idx][key] = [[new_text]]
                else:
                    merged[idx][key] = [new_text]
            else:
                merged[idx][key] = new_text

    # Re-apply task filters + process_results to re-score the merged items
    tm = TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"))
    task_dict = get_task_dict([args.task], task_manager=tm)
    task_obj = task_dict[args.task]
    if hasattr(task_obj, "task"):  # group wrapper
        task_obj = task_obj.task

    # Re-run filters (the `filter_list` from the task config produces
    # filtered_resps per item). For simplicity we invoke the task's
    # apply_filters if available, else trust the existing filter output of the
    # raw model text (which is what lm_eval applied previously).
    # Recompute per-item metric by calling task.process_results(doc, [filtered_resp])
    from collections import defaultdict
    per_metric_sums = defaultdict(list)
    for it in merged:
        doc = it.get("doc")
        # filtered_resps is a list of per-filter results; for multi-filter tasks
        # we iterate each filter separately
        fr = it.get("filtered_resps") or it.get("resps") or []
        if fr and isinstance(fr[0], list):
            # grouped by filter: [[flt1_resp], [flt2_resp], ...] — but samples
            # use one filter per sample-entry (we iterate them separately)
            resp_list = [x[0] if isinstance(x, list) else x for x in fr]
        else:
            resp_list = fr
        try:
            res = task_obj.process_results(doc, resp_list)
        except Exception as e:
            print(f"[retry] process_results failed for doc_id={it.get('doc_id')}: {e}")
            continue
        for k, v in res.items():
            per_metric_sums[k].append(float(v))

    # Aggregate
    out_path = args.samples.replace("_samples.json", f"_retry{args.retry_mg}_results.json")
    final = {task_name: {}}
    for k, vs in per_metric_sums.items():
        m = sum(vs) / max(len(vs), 1)
        final[task_name][f"{k},retry_merge"] = m
        final[task_name][f"{k}_n,retry_merge"] = len(vs)
    print(f"[retry] MERGED RESULTS:")
    for k, v in final[task_name].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"[retry] wrote {out_path}")

    # Also save the merged samples for traceability
    merged_path = args.samples.replace("_samples.json", f"_retry{args.retry_mg}_merged_samples.json")
    with open(merged_path, "w") as f:
        json.dump({task_name: merged}, f, indent=2, default=str)
    print(f"[retry] wrote {merged_path}")


if __name__ == "__main__":
    main()
