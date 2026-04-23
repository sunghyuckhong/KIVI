"""
Measure max prompt token length for each eval task, using lm_eval's task
manager to build the actual prompts (fewshot-included) and a Llama3 tokenizer.

Output: suggested max_model_len per task = max_prompt_len + max_gen_toks.
"""
import argparse
import os

import numpy as np
from transformers import AutoTokenizer

from lm_eval.tasks import TaskManager
from lm_eval.evaluator import get_task_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--tasks", nargs="+", default=[
        "gsm8k_32k",
        "gpqa_diamond_cot_n_shot_32k",
        "math500_32k",
    ])
    ap.add_argument("--task_dir", default="tasks")
    ap.add_argument("--max_gen_toks", type=int, default=32768,
                    help="To derive suggested max_model_len = max_prompt + max_gen_toks")
    args = ap.parse_args()

    tm = TaskManager(include_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.task_dir))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    task_dict = get_task_dict(args.tasks, tm)

    print(f"{'task':36s} | {'n_docs':>7s} | {'max':>6s} | {'p99':>6s} | {'mean':>6s} | {'sugg. max_model_len':>20s}")
    print("-" * 100)
    for name, task in task_dict.items():
        # Build prompts for every test doc
        docs = list(task.test_docs()) if task.has_test_docs() else list(task.validation_docs())
        prompts = []
        for d in docs:
            try:
                ctx = task.fewshot_context(doc=d, num_fewshot=task.config.num_fewshot or 0)
                prompts.append(ctx)
            except Exception as e:
                # Some tasks build fewshot differently; fall back to doc text
                pass
        if not prompts:
            print(f"{name:36s} | {'—':>7s} | (no prompts built)")
            continue
        lengths = np.array([len(tok.encode(p)) for p in prompts])
        suggested = int(lengths.max() + args.max_gen_toks)
        # Round up to next multiple of 256 for clean block alignment
        suggested = ((suggested + 255) // 256) * 256
        print(f"{name:36s} | {len(lengths):>7d} | {lengths.max():>6d} | "
              f"{int(np.percentile(lengths, 99)):>6d} | {int(lengths.mean()):>6d} | "
              f"{suggested:>20d}")


if __name__ == "__main__":
    main()
