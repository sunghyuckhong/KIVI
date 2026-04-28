"""
Architecture-agnostic FP8 KV cache sanity test.

Instead of a custom model class, we monkey-patch transformers.DynamicCache.update()
to apply FP8 (e4m3fn) per-token quantization round-trip after each cache update.

This works with ANY model (Mistral, Llama, Qwen2, ...) since DynamicCache is
the default cache used by model.generate().

Usage:
  python run_fp8_cachepatch.py --model_path Qwen/Qwen2.5-1.5B-Instruct --task gsm8k
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate, utils

from quant.fp8_quant import quantize_fp8, dequantize_fp8


TASK_CFG = {
    "gsm8k":          dict(tasks=["gsm8k"], num_fewshot=5),
    "coqa":           dict(tasks=["coqa"], num_fewshot=0),
    "truthfulqa_gen": dict(tasks=["truthfulqa_gen"], num_fewshot=0),
}


def patch_cache_with_fp8(group_size=128):
    """Monkey-patch DynamicCache.update to apply FP8 round-trip quantization.

    Every call to update(key_states, value_states, layer_idx, ...) will:
      1. Quantize new KV to FP8 (per-token)
      2. Dequantize back to fp16
      3. Store in the cache as usual
    This simulates FP8 storage accuracy while keeping HF's cache mechanics.
    """
    original_update = DynamicCache.update

    def update_fp8(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Apply FP8 round-trip on the new states only
        # key_states, value_states shape: (B, num_kv_heads, T_new, head_dim)
        k_q, k_s = quantize_fp8(key_states.contiguous(), group_size)
        v_q, v_s = quantize_fp8(value_states.contiguous(), group_size)
        k_deq = dequantize_fp8(k_q, k_s, group_size)
        v_deq = dequantize_fp8(v_q, v_s, group_size)
        return original_update(self, k_deq, v_deq, layer_idx, cache_kwargs)

    DynamicCache.update = update_fp8
    print(f"Patched DynamicCache.update with FP8 per-token group_size={group_size}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--task", choices=list(TASK_CFG.keys()), required=True)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--fp16", action="store_true", help="Skip patching (pure fp16 baseline)")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Model dtype. 'auto' preserves model-native (bf16 for Llama-3/Mistral/DSR1/Qwen3).")
    args = p.parse_args()

    if args.dtype == "auto":
        from transformers import AutoConfig
        td = getattr(AutoConfig.from_pretrained(args.model_path), "torch_dtype", None)
        _dt = td if isinstance(td, torch.dtype) else torch.bfloat16
    else:
        _dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}[args.dtype]
    print(f"Model dtype: {_dt}")

    os.makedirs("logs", exist_ok=True)
    m_short = args.model_path.rstrip("/").split("/")[-1].lower()
    tag = "fp16" if args.fp16 else f"fp8_cachepatch_g{args.group_size}"
    out_path = f"logs/{args.task}_{m_short}_{tag}_results.json"

    print(f"\n{'='*60}")
    print(f"  model={args.model_path}  task={args.task}  mode={tag}")
    print(f"  output -> {out_path}")
    print(f"{'='*60}\n")

    if not args.fp16:
        patch_cache_with_fp8(args.group_size)

    print(f"Loading {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=_dt, low_cpu_mem_usage=True,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = simple_evaluate(
        model=lm, batch_size=args.batch_size, log_samples=False,
        **TASK_CFG[args.task],
    )

    print(utils.make_table(results))
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
