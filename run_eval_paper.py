"""
Evaluation script using the old lm-eval (commit c9bbec6e) to replicate KIVI paper results.

This version uses the old lm-eval API where HFLM only accepts string model paths.
For FP16 baselines, we use simple_evaluate directly.
For KIVI models, we subclass HFLM to load our custom model classes.

Usage:
  python run_eval_paper.py --model fp16     --task gsm8k    --model_path meta-llama/Llama-2-7b-hf
  python run_eval_paper.py --model fp16     --task coqa     --model_path meta-llama/Llama-2-7b-hf
  python run_eval_paper.py --model kivi     --task gsm8k    --model_path meta-llama/Llama-2-7b-hf
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks
initialize_tasks()

DEFAULT_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

TASK_CFG = {
    "gsm8k":           dict(tasks="gsm8k",           num_fewshot=5),
    "gsm8k_cot":       dict(tasks="gsm8k_cot",       num_fewshot=None),
    "coqa":            dict(tasks="coqa",             num_fewshot=0),
    "truthfulqa_mc1":  dict(tasks="truthfulqa_mc1",   num_fewshot=0),
    "truthfulqa_mc2":  dict(tasks="truthfulqa_mc2",   num_fewshot=0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=["fp16", "kivi"], required=True)
    p.add_argument("--task",       choices=list(TASK_CFG.keys()), required=True)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--residual",   type=int, default=128)
    p.add_argument("--k_bits",     type=int, default=2)
    p.add_argument("--v_bits",     type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Model dtype. 'auto' preserves model-native (bf16 for Llama-3/Mistral/DSR1/Qwen3).")
    return p.parse_args()


def resolve_dtype(arg_dtype, model_path):
    if arg_dtype == "auto":
        from transformers import AutoConfig
        td = getattr(AutoConfig.from_pretrained(model_path), "torch_dtype", None)
        return td if isinstance(td, torch.dtype) else torch.bfloat16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[arg_dtype]


def is_llama(model_path):
    return "llama" in model_path.lower()


def model_short_name(model_path):
    return model_path.rstrip("/").split("/")[-1].lower()


def output_name(args):
    t = args.task
    m = model_short_name(args.model_path)
    if args.model == "fp16":
        return f"{t}_{m}_fp16_paper"
    bits_tag = f"_int{args.k_bits}" if args.k_bits != 2 else ""
    res_tag = f"_res{args.residual}" if args.residual != 32 else ""
    return f"{t}_{m}_kivi{bits_tag}{res_tag}_paper"


def load_kivi_model(args):
    """Load a KIVI quantized model."""
    mp = args.model_path
    _dt = resolve_dtype(args.dtype, mp)
    print(f"Model dtype: {_dt}")
    if is_llama(mp):
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(mp)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual
        config.use_flash = True

        from models.llama_kivi import LlamaForCausalLM_KIVI
        print(f"Loading KIVI {mp} (k={args.k_bits}, v={args.v_bits}, "
              f"g={args.group_size}, res={args.residual})...")
        return LlamaForCausalLM_KIVI.from_pretrained(
            mp, config=config, low_cpu_mem_usage=True, torch_dtype=_dt
        ).cuda()
    else:
        from transformers import MistralConfig
        config = MistralConfig.from_pretrained(mp)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual
        config.use_flash = False

        from models.mistral_kivi import MistralForCausalLM_KIVI
        print(f"Loading KIVI {mp} (k={args.k_bits}, v={args.v_bits}, "
              f"g={args.group_size}, res={args.residual})...")
        return MistralForCausalLM_KIVI.from_pretrained(
            mp, config=config, low_cpu_mem_usage=True, torch_dtype=_dt
        ).cuda()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    name = output_name(args)
    out_path = f"logs/{name}_results.json"

    print(f"\n{'='*60}")
    print(f"  [paper env] model={args.model}  task={args.task}  path={args.model_path}")
    print(f"  output -> {out_path}")
    print(f"{'='*60}\n")

    task_cfg = TASK_CFG[args.task]

    if args.model == "fp16":
        results = simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.model_path},dtype={args.dtype}",
            tasks=task_cfg["tasks"],
            num_fewshot=task_cfg["num_fewshot"],
            batch_size=args.batch_size,
            log_samples=False,
        )
    else:
        # Load KIVI model, then create HFLM wrapper with swapped model
        from transformers import AutoTokenizer
        kivi_model = load_kivi_model(args)
        kivi_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

        # Create HFLM with the original model path (it loads the standard model)
        lm = HFLM(pretrained=args.model_path, dtype=args.dtype, batch_size=args.batch_size)
        # Swap in the KIVI model
        lm._model = kivi_model
        lm.tokenizer = tokenizer

        results = simple_evaluate(
            model=lm,
            tasks=task_cfg["tasks"],
            num_fewshot=task_cfg["num_fewshot"],
            batch_size=args.batch_size,
            log_samples=False,
        )

    # Print results table
    from lm_eval.utils import make_table
    print(make_table(results))

    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
