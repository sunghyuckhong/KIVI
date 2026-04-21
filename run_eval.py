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

DEFAULT_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

TASK_CFG = {
    "gsm8k_32k": dict(tasks=["gsm8k_32k"], num_fewshot=5),
    "gsm8k_zeroshot": dict(tasks=["gsm8k"], num_fewshot=0),
    "gsm8k_cot_zeroshot": dict(tasks=["gsm8k_cot_zeroshot"]),
    "gsm8k_cot": dict(tasks=["gsm8k_cot"]),
    "coqa":  dict(tasks=["coqa"], num_fewshot=0),
    "truthfulqa_mc1": dict(tasks=["truthfulqa_mc1"], num_fewshot=0),
    "truthfulqa_gen": dict(tasks=["truthfulqa_gen"], num_fewshot=0),
    "gpqa_diamond_cot_n_shot_32k": dict(tasks=["gpqa_diamond_cot_n_shot_32k"]),
    "gpqa_diamond_cot_zeroshot": dict(tasks=["gpqa_diamond_cot_zeroshot"]),
    "math500_32k":  dict(tasks=["math500_32k"]),
    "aime":    dict(tasks=["aime"]),
    "aime24":  dict(tasks=["aime24"]),
    "aime25":  dict(tasks=["aime25"]),
}


def parse_args():
    p = argparse.ArgumentParser(description="KIVI KV-cache quantization evaluation")
    p.add_argument("--model",      choices=["fp16", "kivi", "pertoken", "fp8", "smoothkv"], required=True)
    p.add_argument("--calib_path", type=str, default=None,
                   help="Path to SmoothKV calibration .pt file (required when --model=smoothkv)")
    p.add_argument("--task",       choices=["gsm8k_32k", "gsm8k_zeroshot", "gsm8k_cot", "gsm8k_cot_zeroshot", "coqa", "truthfulqa_mc1", "truthfulqa_gen", "gpqa_diamond_cot_n_shot_32k", "gpqa_diamond_cot_zeroshot", "math500_32k", "aime", "aime24", "aime25"], required=True)
    p.add_argument("--group_size", type=int, default=32,
                   help="Quantization group size along head_dim (32 or 128)")
    p.add_argument("--residual",   type=int, default=32,
                   help="FP16 residual buffer length (0 = no buffer)")
    p.add_argument("--k_bits",     type=int, default=2)
    p.add_argument("--v_bits",     type=int, default=2)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                   help="HuggingFace model path (default: Mistral-7B-Instruct-v0.2)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_gen_toks", type=int, default=None,
                   help="Per-run override for task generation_kwargs.max_gen_toks.")
    return p.parse_args()


def model_short_name(model_path):
    """Extract a short name from the HF model path for filenames."""
    return model_path.rstrip("/").split("/")[-1].lower()


def output_name(args):
    """Derive a canonical output filename from args."""
    t = args.task
    m = model_short_name(args.model_path)
    if args.model == "fp16":
        return f"{t}_{m}_fp16"

    bits_tag = f"_int{args.k_bits}" if args.k_bits != 2 else ""

    if args.residual == 0:
        res_tag = "_noresidual"
    elif args.residual != 32:
        res_tag = f"_res{args.residual}"
    else:
        res_tag = ""

    grp_tag = f"_g{args.group_size}" if args.group_size != 32 else ""

    if args.model == "kivi":
        return f"{t}_{m}_kivi{bits_tag}{grp_tag}{res_tag}"
    elif args.model == "fp8":
        return f"{t}_{m}_fp8{grp_tag}{res_tag}"
    elif args.model == "smoothkv":
        return f"{t}_{m}_smoothkv{grp_tag}"
    else:  # pertoken
        flat = "_flat" if args.group_size == 128 else ""
        return f"{t}_{m}_pertoken{bits_tag}{flat}{res_tag}"


def is_llama(model_path):
    return "llama" in model_path.lower()


def load_model(args):
    mp = args.model_path

    if args.model == "fp16":
        from transformers import AutoModelForCausalLM
        print(f"Loading FP16 {mp} (no quantization, flash_attention_2)...")
        return AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).cuda()

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(mp)
    config.k_bits         = args.k_bits
    config.v_bits         = args.v_bits
    config.group_size     = args.group_size
    config.residual_length = args.residual
    config.use_flash      = True

    if args.model == "fp8":
        config.use_flash = False
        if is_llama(mp):
            from models.llama_kivi_fp8 import LlamaForCausalLM_FP8
            print(f"Loading FP8 Llama {mp} (group={args.group_size})...")
            return LlamaForCausalLM_FP8.from_pretrained(
                mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
            ).cuda()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            config.sliding_window = config.max_position_embeddings
        from models.mistral_kivi_fp8 import MistralForCausalLM_FP8
        print(f"Loading FP8 {mp} (group={args.group_size})...")
        return MistralForCausalLM_FP8.from_pretrained(
            mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()

    if args.model == "smoothkv":
        assert args.calib_path is not None, "--calib_path required for smoothkv"
        config.use_flash = False
        if is_llama(mp):
            from models.llama_smoothkv import LlamaForCausalLM_SmoothKV
            print(f"Loading SmoothKV Llama {mp} (calib={args.calib_path}, "
                  f"group={args.group_size})...")
            return LlamaForCausalLM_SmoothKV.from_pretrained_with_calib(
                mp, args.calib_path, config=config,
                low_cpu_mem_usage=True, torch_dtype=torch.float16
            ).cuda()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            config.sliding_window = config.max_position_embeddings
        from models.mistral_smoothkv import MistralForCausalLM_SmoothKV
        print(f"Loading SmoothKV {mp} (calib={args.calib_path}, "
              f"group={args.group_size})...")
        return MistralForCausalLM_SmoothKV.from_pretrained_with_calib(
            mp, args.calib_path, config=config,
            low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()

    if args.model == "pertoken":
        config.use_flash = False
        if is_llama(mp):
            from models.llama_kivi_pertoken import LlamaForCausalLM_KIVI_PerToken
            print(f"Loading pertoken Llama {mp} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
                  f"group={args.group_size}, residual={args.residual})...")
            return LlamaForCausalLM_KIVI_PerToken.from_pretrained(
                mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
            ).cuda()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            config.sliding_window = config.max_position_embeddings
        from models.mistral_kivi_pertoken import MistralForCausalLM_KIVI_PerToken
        print(f"Loading pertoken {mp} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
              f"group={args.group_size}, residual={args.residual})...")
        return MistralForCausalLM_KIVI_PerToken.from_pretrained(
            mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()

    if is_llama(mp):
        if args.model == "kivi":
            from models.llama_kivi import LlamaForCausalLM_KIVI
            print(f"Loading KIVI {mp} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
                  f"group={args.group_size}, residual={args.residual})...")
            return LlamaForCausalLM_KIVI.from_pretrained(
                mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
            ).cuda()
    else:
        # Mistral KIVI
        from models.mistral_kivi import MistralForCausalLM_KIVI
        print(f"Loading KIVI {mp} (k_bits={args.k_bits}, v_bits={args.v_bits}, "
              f"group={args.group_size}, residual={args.residual})...")
        return MistralForCausalLM_KIVI.from_pretrained(
            mp, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    name     = output_name(args)
    out_path = f"logs/{name}_results.json"

    print(f"\n{'='*60}")
    if args.model == "fp16":
        print(f"  model={args.model}  task={args.task}  path={args.model_path}")
    else:
        print(f"  model={args.model}  task={args.task}  path={args.model_path}  "
              f"group_size={args.group_size}  residual={args.residual}")
    print(f"  output → {out_path}")
    print(f"{'='*60}\n")

    model = load_model(args)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    # Register custom MATH500 task. The API differs between lm-eval versions:
    #   * paper env (commit c9bbec6e): include_path is a module-level function
    #   * modern env (0.4.2): include_path is a TaskManager method, and the
    #     TaskManager must be passed to simple_evaluate for it to see the task.
    math500_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "math500")
    tm = None
    try:
        from lm_eval.tasks import include_path as _include_path
        _include_path(math500_dir)
    except (ImportError, AttributeError):
        try:
            from lm_eval.tasks import TaskManager
            tm = TaskManager(include_path=math500_dir)
        except Exception as e:
            print(f"[warn] include_path math500 failed: {e}")

    kwargs = dict(**TASK_CFG[args.task])
    if tm is not None:
        kwargs["task_manager"] = tm
    if args.max_gen_toks is not None:
        kwargs["gen_kwargs"] = f"max_gen_toks={args.max_gen_toks}"
    results = simple_evaluate(
        model=lm,
        batch_size=args.batch_size,
        log_samples=False,
        **kwargs,
    )

    print(utils.make_table(results))
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
