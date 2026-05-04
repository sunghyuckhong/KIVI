"""SmoothKV calibration: collect post-RoPE Q/K and pre-attention V activation
stats, then compute the smoothing scales used at inference time.

Per (layer, head, channel):
  s_K = max|K_rot|^alpha / max|Q_rot_grouped|^(1-alpha)   (SmoothQuant)
  s_V = max|V|^beta

This file is a thin CLI on top of the ``calib/`` package:
  calib.StatCollector   — running max|x| (+ optional reservoir)
  calib.install_v_hook  — V capture via v_proj forward-hook
  calib.monkey_patch_rope — Q/K post-RoPE capture via apply_rotary_pos_emb wrap
  calib.compute_scales  — apply the SmoothQuant formulas

Usage:
  python run_smoothkv_calibrate.py \\
      --model Qwen/Qwen3-8B \\
      --num_samples 512 --seq_length 2048 \\
      --alpha 1.0 --beta 1.0 \\
      --output calib/smoothkv_qwen3-8b_a1.0_b1.0.pt
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from calib import (
    StatCollector,
    compute_scales,
    install_v_hook,
    monkey_patch_rope,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model path or hub id (e.g. Qwen/Qwen3-8B)")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--seq_length", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="SmoothQuant alpha for K-side scale migration")
    p.add_argument("--beta", type=float, default=0.5,
                   help="V-side scaling power")
    p.add_argument("--dataset", type=str,
                   default="neuralmagic/LLM_compression_calibration")
    p.add_argument("--dataset_config", type=str, default=None,
                   help="HF config name (e.g. 'wikitext-2-raw-v1' for 'wikitext'). Optional.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text",
                   help="Single column with the text. Ignored if --text_columns is set.")
    p.add_argument("--text_columns", nargs="+", type=str, default=None,
                   help="Concatenate multiple columns (e.g. 'problem solution') "
                        "joined by --text_join into one calibration sample.")
    p.add_argument("--text_join", type=str, default="\n\n",
                   help="Separator used when joining --text_columns.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--samples_per_channel", type=int, default=0,
                   help="If >0, uniformly subsample this many abs values per "
                        "(layer, head, channel) for offline percentile calibration. "
                        "Adds 32*nh*D*R*2B CPU memory; 0 disables (max-only, backward-compat).")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="If set, render each sample via tokenizer.apply_chat_template "
                        "using the dataset's 'messages' column instead of raw 'text'. "
                        "Required for chat-on eval distribution matching.")
    return p.parse_args()


def load_model(model_path, device):
    """Load HF model + tokenizer; falls back to AutoModel for multimodal
    wrappers (e.g. Exaone4_5_ForConditionalGeneration) that hold the LM
    under .language_model and have no AutoModelForCausalLM registry entry.
    Returns (model, tokenizer, resolved_device)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype="auto", low_cpu_mem_usage=True)
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM failed ({type(e).__name__}); falling back to AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, **load_kwargs)

    model = model if device == "auto" else model.to(device)
    model.eval()
    if device == "auto":
        device = "cuda:0"  # collector buffers go here
    return model, tokenizer, device


def get_arch_dims(model):
    """Read (num_layers, num_q_heads, num_kv_heads, head_dim, has_qk_norm)
    from the model. Multimodal configs (e.g. Exaone4_5_Config) hold LM dims
    under cfg.text_config; fall back to that when the top-level lacks them."""
    cfg = model.config
    if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_q_heads)

    # Q/K-RMSNorm models (Qwen3, Qwen3-MoE, Olmo2, …) require head-uniform s_K
    # downstream because q_norm.weight / k_norm.weight are (head_dim,) shared
    # across heads — pre-norm row scaling on qkv_proj does not survive RMSNorm.
    has_qk_norm = any(
        hasattr(m, "q_norm") and hasattr(m, "k_norm")
        and isinstance(getattr(m, "q_norm", None), torch.nn.Module)
        and isinstance(getattr(m, "k_norm", None), torch.nn.Module)
        for m in model.modules()
    )
    return num_layers, num_q_heads, num_kv_heads, head_dim, has_qk_norm


def load_calib_dataset(args):
    """Load a calibration dataset from HF hub or a local .json[l] file."""
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        return load_dataset("json", data_files=args.dataset, split="train")
    if args.dataset_config:
        return load_dataset(args.dataset, args.dataset_config, split=args.split)
    return load_dataset(args.dataset, split=args.split)


def encode_sample(row, tokenizer, args, device):
    """Encode one calibration sample. Returns a dict suitable for model(**enc),
    or None if the sample is empty / too short."""
    if args.apply_chat_template:
        messages = row.get("messages")
        if not messages:
            return None
        tmpl = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            truncation=True, max_length=args.seq_length,
            add_generation_prompt=False,
        )
        # apply_chat_template returns either a Tensor or a BatchEncoding.
        input_ids = tmpl.input_ids if hasattr(tmpl, "input_ids") else tmpl
        input_ids = input_ids.to(device)
        if input_ids.shape[1] < 16:
            return None
        return {"input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)}

    if args.text_columns:
        parts = [str(row[c]) for c in args.text_columns if row.get(c)]
        text = args.text_join.join(parts)
    else:
        text = row[args.text_column]
    if not text or not text.strip():
        return None
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=args.seq_length, padding=False).to(device)
    if enc.input_ids.shape[1] < 16:
        return None
    return enc


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, args.device)

    num_layers, num_q_heads, num_kv_heads, head_dim, has_qk_norm = get_arch_dims(model)
    print(f"Model: {num_layers} layers, {num_q_heads} Q heads, "
          f"{num_kv_heads} KV heads, head_dim={head_dim}")
    print(f"Q/K-RMSNorm: {has_qk_norm}  "
          f"(zero-runtime fusion {'requires' if has_qk_norm else 'does NOT need'} head-uniform s_K)")

    collector = StatCollector(
        num_layers, num_kv_heads, num_q_heads, head_dim, device,
        samples_per_channel=args.samples_per_channel,
    )
    monkey_patch_rope(model, collector)
    hooks = install_v_hook(model, collector)

    print(f"Loading dataset: {args.dataset}")
    ds = load_calib_dataset(args)

    print(f"Calibrating on {args.num_samples} samples at {args.seq_length} tokens"
          f"{' (chat-template applied)' if args.apply_chat_template else ''}...")
    with torch.no_grad():
        for i in tqdm(range(min(args.num_samples, len(ds)))):
            enc = encode_sample(ds[i], tokenizer, args, device)
            if enc is None:
                continue
            model(**enc)

    for h in hooks:
        h.remove()

    s_K, s_V, max_q_grouped = compute_scales(collector, args.alpha, args.beta)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "s_K": s_K.cpu(),
        "s_V": s_V.cpu(),
        "max_q":         collector.max_q.cpu(),
        "max_q_grouped": max_q_grouped.cpu(),
        "max_k":         collector.max_k.cpu(),
        "max_v":         collector.max_v.cpu(),
        "alpha": args.alpha,
        "beta": args.beta,
        "model_path": args.model,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "has_qk_norm": has_qk_norm,
    }
    if collector.samples_k is not None:
        payload["samples_k"] = collector.samples_k
        payload["samples_v"] = collector.samples_v
        payload["count_k"]   = collector.count_k
        payload["count_v"]   = collector.count_v
    torch.save(payload, args.output)
    print(f"Saved calibration to {args.output}")
    print(f"  s_K: min={s_K.min():.4f}, max={s_K.max():.4f}, mean={s_K.mean():.4f}")
    print(f"  s_V: min={s_V.min():.4f}, max={s_V.max():.4f}, mean={s_V.mean():.4f}")


if __name__ == "__main__":
    main()
