"""
SmoothKV calibration: collect per-channel stats post-RoPE and compute smoothing scales.

For each (layer, head, channel) we compute:
  s_K[c] = max|K_rot|[c]^alpha / max|Q_rot|[c]^(1-alpha)  (SmoothQuant formula)
  s_V[c] = max|V|[c]^beta / geometric_mean_over_c(max|V|^beta)  (normalized)

Usage:
  python run_smoothkv_calibrate.py \
      --model_path mistralai/Mistral-7B-Instruct-v0.2 \
      --num_samples 128 \
      --seq_length 2048 \
      --alpha 0.5 \
      --beta 0.5 \
      --output calib/smoothkv_mistral-7b-instruct-v0.2_a0.5.pt
"""
import argparse
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--seq_length", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="SmoothQuant alpha for K-side scale migration")
    p.add_argument("--beta", type=float, default=0.5,
                   help="V-side scaling power")
    p.add_argument("--dataset", type=str,
                   default="neuralmagic/LLM_compression_calibration")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_batches_before_summary", type=int, default=32,
                   help="Reduce peak GPU mem by flushing per-channel stats periodically")
    return p.parse_args()


class StatCollector:
    """Accumulates running max|x| per channel for each (layer, head) over many batches."""

    def __init__(self, num_layers, num_kv_heads, num_q_heads, head_dim, device):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.device = device

        # max|Q| per (layer, num_q_heads, channel) in fp32
        self.max_q = torch.zeros(num_layers, num_q_heads, head_dim, device=device)
        self.max_k = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)
        self.max_v = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)

    def update_q(self, layer_idx, q):
        """q: (B, num_q_heads, T, D) — post-RoPE."""
        m = q.abs().amax(dim=(0, 2)).to(torch.float32)  # (nh, D)
        torch.maximum(self.max_q[layer_idx], m, out=self.max_q[layer_idx])

    def update_k(self, layer_idx, k):
        """k: (B, num_kv_heads, T, D) — post-RoPE."""
        m = k.abs().amax(dim=(0, 2)).to(torch.float32)
        torch.maximum(self.max_k[layer_idx], m, out=self.max_k[layer_idx])

    def update_v(self, layer_idx, v):
        """v: (B, num_kv_heads, T, D)."""
        m = v.abs().amax(dim=(0, 2)).to(torch.float32)
        torch.maximum(self.max_v[layer_idx], m, out=self.max_v[layer_idx])


def install_hooks(model, collector):
    """Hook into each attention module to capture post-RoPE Q, K and pre-RoPE V."""
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # The attention module's forward gets hidden_states, position_ids, etc.
            # We need post-RoPE Q, K. Since we can't easily intercept that from the
            # output, we'll monkey-patch apply_rotary_pos_emb instead (see below).
            pass
        return hook_fn

    # Alternative: monkey-patch apply_rotary_pos_emb to capture Q, K
    # and a v_proj hook to capture V.
    # This is cleaner than trying to post-process module output.

    # Capture V via v_proj hook (pre-RoPE since V has no RoPE in Mistral/Llama)
    for i, layer in enumerate(model.model.layers):
        def make_v_hook(layer_idx):
            def hook_v(module, inp, out):
                # out: (B, T, num_kv_heads * head_dim)
                B, T, _ = out.shape
                nh = collector.num_kv_heads
                D = collector.head_dim
                v = out.view(B, T, nh, D).transpose(1, 2).contiguous()
                collector.update_v(layer_idx, v)
            return hook_v
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

    return hooks


def monkey_patch_rope(model, collector):
    """Wrap apply_rotary_pos_emb to capture post-RoPE Q and K."""
    # Figure out which module we're in by looking at q.shape[1]
    # (num_q_heads) — this lets us route the stats to the right layer.
    # But we don't know the layer index from apply_rotary_pos_emb alone.
    # Instead, wrap each attention module's forward to capture post-RoPE Q, K.

    from transformers.models.mistral import modeling_mistral as mm
    from transformers.models.llama   import modeling_llama   as ll

    orig_apply_rope_mistral = mm.apply_rotary_pos_emb
    orig_apply_rope_llama   = ll.apply_rotary_pos_emb

    # State: track which layer is currently being processed.
    current_layer = [0]

    def wrap_forward(layer_idx, orig_fwd):
        def new_fwd(*args, **kwargs):
            current_layer[0] = layer_idx
            return orig_fwd(*args, **kwargs)
        return new_fwd

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = wrap_forward(i, layer.self_attn.forward)

    def patched_rope_mistral(q, k, cos, sin, position_ids=None, *a, **kw):
        qr, kr = orig_apply_rope_mistral(q, k, cos, sin, position_ids, *a, **kw)
        li = current_layer[0]
        collector.update_q(li, qr)
        collector.update_k(li, kr)
        return qr, kr

    def patched_rope_llama(q, k, cos, sin, position_ids=None, *a, **kw):
        qr, kr = orig_apply_rope_llama(q, k, cos, sin, position_ids, *a, **kw)
        li = current_layer[0]
        collector.update_q(li, qr)
        collector.update_k(li, kr)
        return qr, kr

    mm.apply_rotary_pos_emb = patched_rope_mistral
    ll.apply_rotary_pos_emb = patched_rope_llama


def main():
    args = parse_args()
    device = args.device

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads

    print(f"Model: {num_layers} layers, {num_q_heads} Q heads, "
          f"{num_kv_heads} KV heads, head_dim={head_dim}")

    collector = StatCollector(num_layers, num_kv_heads, num_q_heads, head_dim, device)
    monkey_patch_rope(model, collector)
    hooks = install_hooks(model, collector)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split)

    print(f"Calibrating on {args.num_samples} samples at {args.seq_length} tokens...")
    with torch.no_grad():
        for i in tqdm(range(min(args.num_samples, len(ds)))):
            text = ds[i][args.text_column]
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=args.seq_length,
                            padding=False).to(device)
            if enc.input_ids.shape[1] < 16:
                continue  # skip tiny samples
            model(**enc)

    for h in hooks:
        h.remove()

    # Compute scales
    # K-side: s_K[c] = max|K|[c]^alpha / max|Q|[c]^(1-alpha)
    # Since Q may have more heads than K (GQA), aggregate Q across each KV-group.
    n_rep = num_q_heads // num_kv_heads
    max_q_grouped = collector.max_q.view(num_layers, num_kv_heads, n_rep, head_dim).amax(dim=2)
    # max_q_grouped: (L, num_kv_heads, D)

    alpha = args.alpha
    beta = args.beta

    eps = 1e-5
    s_K = (collector.max_k.clamp(min=eps) ** alpha) / \
          (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    # Normalize s_K per (layer, head) so the geometric mean across channels = 1
    # (prevents inflating range; keeps overall magnitude stable)
    log_s_K = s_K.log()
    s_K = (log_s_K - log_s_K.mean(dim=-1, keepdim=True)).exp()

    # V-side: s_V[c] = max|V|[c]^beta, normalized per (layer, head)
    s_V = collector.max_v.clamp(min=eps) ** beta
    log_s_V = s_V.log()
    s_V = (log_s_V - log_s_V.mean(dim=-1, keepdim=True)).exp()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "s_K": s_K.cpu(),  # (L, num_kv_heads, D)
        "s_V": s_V.cpu(),
        # Raw max statistics — used by make_alpha_variants.py to generate
        # new s_K/s_V without rerunning the forward pass.
        "max_q":         collector.max_q.cpu(),          # (L, num_q_heads, D)
        "max_q_grouped": max_q_grouped.cpu(),            # (L, num_kv_heads, D)
        "max_k":         collector.max_k.cpu(),          # (L, num_kv_heads, D)
        "max_v":         collector.max_v.cpu(),          # (L, num_kv_heads, D)
        "alpha": alpha,
        "beta": beta,
        "model_path": args.model_path,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }, args.output)
    print(f"Saved calibration to {args.output}")
    print(f"  s_K range per layer: min={s_K.min():.4f}, max={s_K.max():.4f}, "
          f"mean={s_K.mean():.4f}")
    print(f"  s_V range per layer: min={s_V.min():.4f}, max={s_V.max():.4f}, "
          f"mean={s_V.mean():.4f}")


if __name__ == "__main__":
    main()
