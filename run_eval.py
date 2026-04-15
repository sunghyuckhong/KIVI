"""
Unified evaluation script for KIVI KV-cache quantization experiments.

Usage:
  python run_eval.py --model fp16     --task gsm8k
  python run_eval.py --model kivi     --task gsm8k
  python run_eval.py --model pertoken --task gpqa  --group_size 32  --residual 32
  python run_eval.py --model pertoken --task gsm8k --group_size 128 --residual 0

Arguments:
  --model      : fp16 | kivi | pertoken
  --task       : gsm8k | gpqa
  --group_size : quantization group size along head_dim (default: 32)
                 32  → 4 groups per token
                 128 → 1 group per token (flat, = head_dim)
  --residual   : FP16 residual buffer length in tokens (default: 32)
                 32  → keep last 32 tokens in FP16
                 0   → quantize every token immediately
  --k_bits     : key cache bit-width (default: 2)
  --v_bits     : value cache bit-width (default: 2)
  --batch_size : lm-eval batch size (default: 16, safe for 32GB VRAM)
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate, utils

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

TASK_CFG = {
    "gsm8k": dict(tasks=["gsm8k"], num_fewshot=5),
    "gpqa":  dict(tasks=["gpqa_diamond_generative_n_shot"]),
}


def parse_args():
    p = argparse.ArgumentParser(description="KIVI KV-cache quantization evaluation")
    p.add_argument("--model",      choices=["fp16", "kivi", "pertoken"], required=True)
    p.add_argument("--task",       choices=["gsm8k", "gpqa"], required=True)
    p.add_argument("--group_size", type=int, default=32,
                   help="Quantization group size along head_dim (32 or 128)")
    p.add_argument("--residual",   type=int, default=32,
                   help="FP16 residual buffer length (0 = no buffer)")
    p.add_argument("--k_bits",     type=int, default=2)
    p.add_argument("--v_bits",     type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def output_name(args):
    """Derive a canonical output filename from args (matches legacy script names)."""
    t = args.task
    if args.model == "fp16":
        return f"{t}_fp16"
    elif args.model == "kivi":
        return f"{t}_kivi"
    else:  # pertoken
        flat    = "_flat"       if args.group_size == 128 else ""
        nores   = "_noresidual" if args.residual   == 0   else ""
        return f"{t}_pertoken{flat}{nores}"


def load_model(args):
    if args.model == "fp16":
        from transformers import AutoModelForCausalLM
        print(f"Loading FP16 {MODEL_PATH} (no quantization)...")
        return AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).cuda()

    from transformers import MistralConfig
    config = MistralConfig.from_pretrained(MODEL_PATH)
    config.k_bits         = args.k_bits
    config.v_bits         = args.v_bits
    config.group_size     = args.group_size
    config.residual_length = args.residual
    config.use_flash      = False  # V100 (sm_70) does not support FlashAttention 2

    if args.model == "kivi":
        from models.mistral_kivi import MistralForCausalLM_KIVI
        print(f"Loading KIVI {MODEL_PATH} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
              f"group={args.group_size}, residual={args.residual})...")
        return MistralForCausalLM_KIVI.from_pretrained(
            MODEL_PATH, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()

    # pertoken
    from models.mistral_kivi_pertoken import MistralForCausalLM_KIVI_PerToken
    print(f"Loading KIVI-PerToken {MODEL_PATH} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
          f"group={args.group_size}, residual={args.residual})...")
    return MistralForCausalLM_KIVI_PerToken.from_pretrained(
        MODEL_PATH, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
    ).cuda()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    name     = output_name(args)
    out_path = f"logs/{name}_results.json"

    print(f"\n{'='*60}")
    print(f"  model={args.model}  task={args.task}  "
          f"group_size={args.group_size}  residual={args.residual}")
    print(f"  output → {out_path}")
    print(f"{'='*60}\n")

    model = load_model(args)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = simple_evaluate(
        model=lm,
        batch_size=args.batch_size,
        log_samples=False,
        **TASK_CFG[args.task],
    )

    print(utils.make_table(results))
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
