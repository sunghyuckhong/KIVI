"""
SmoothKV calibration: collect per-channel stats post-RoPE and compute smoothing scales.

For each (layer, head, channel) we compute:
  s_K[c] = max|K_rot|[c]^alpha / max|Q_rot|[c]^(1-alpha)  (SmoothQuant formula)
  s_V[c] = max|V|[c]^beta / geometric_mean_over_c(max|V|^beta)  (normalized)

Usage:
  python run_smoothkv_calibrate.py \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
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
    p.add_argument("--max_batches_before_summary", type=int, default=32,
                   help="Reduce peak GPU mem by flushing per-channel stats periodically")
    p.add_argument("--samples_per_channel", type=int, default=0,
                   help="If >0, uniformly subsample this many abs values per "
                        "(layer, head, channel) for offline percentile calibration. "
                        "Adds 32*nh*D*R*2B CPU memory; 0 disables (max-only, backward-compat).")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="If set, render each sample via tokenizer.apply_chat_template "
                        "using the dataset's 'messages' column instead of raw 'text'. "
                        "Required for chat-on eval distribution matching.")
    return p.parse_args()


class StatCollector:
    """Running max|x| per (layer, head, channel), + optional uniform-random
    subsample of abs values for offline percentile calibration (Eq 11)."""

    def __init__(self, num_layers, num_kv_heads, num_q_heads, head_dim, device,
                 samples_per_channel: int = 0):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.device = device
        self.R = samples_per_channel

        self.max_q = torch.zeros(num_layers, num_q_heads, head_dim, device=device)
        self.max_k = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)
        self.max_v = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)

        if self.R > 0:
            # CPU bf16 buffers — match Qwen3's native dtype (no lossy fp16 downcast).
            self.samples_k = torch.zeros(num_layers, num_kv_heads, head_dim, self.R,
                                         dtype=torch.bfloat16)
            self.samples_v = torch.zeros(num_layers, num_kv_heads, head_dim, self.R,
                                         dtype=torch.bfloat16)
            self.count_k = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.long)
            self.count_v = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.long)
        else:
            self.samples_k = self.samples_v = None
            self.count_k  = self.count_v  = None

    def update_q(self, layer_idx, q):
        # Move incoming tensor to the collector's device — necessary when the
        # model is sharded across GPUs via device_map='auto' (layers may live
        # on cuda:1 while the collector buffers are on cuda:0).
        m = q.abs().amax(dim=(0, 2)).to(torch.float32).to(self.max_q.device)
        torch.maximum(self.max_q[layer_idx], m, out=self.max_q[layer_idx])

    def _update(self, layer_idx, x, max_buf, sample_buf, count_buf):
        """Update running max and (optionally) reservoir sample for x: (B,nh,T,D)."""
        a = x.abs()
        m = a.amax(dim=(0, 2)).to(torch.float32).to(max_buf.device)  # (nh, D)
        torch.maximum(max_buf[layer_idx], m, out=max_buf[layer_idx])

        if sample_buf is None:
            return
        R = self.R
        B, nh, T, D = a.shape
        N = B * T

        # Reshape to (nh, D, N) then move to CPU once per batch. Match sample_buf dtype
        # (bf16) so we don't downcast bf16 activations through fp16.
        flat = a.permute(1, 3, 0, 2).reshape(nh, D, N).to(sample_buf.dtype).cpu()
        # Per channel, fill empty slots first then do reservoir replacement.
        buf = sample_buf[layer_idx]   # (nh, D, R)
        for h in range(nh):
            for c in range(D):
                sb = int(count_buf[layer_idx, h, c].item())  # seen before this batch
                vals = flat[h, c]      # (N,)
                if sb < R:
                    n_fill = min(R - sb, N)
                    buf[h, c, sb:sb + n_fill] = vals[:n_fill]
                    rest = vals[n_fill:]
                    base = sb + n_fill
                else:
                    rest = vals
                    base = sb
                if rest.numel() > 0:
                    # reservoir replacement: for the i-th remaining val, probability
                    # of replacing some slot in the buffer is R / (base + i + 1).
                    t_range = torch.arange(base + 1, base + rest.numel() + 1,
                                           dtype=torch.float32)
                    accept = torch.rand(rest.numel()) < R / t_range
                    accepted_idx = accept.nonzero(as_tuple=True)[0]
                    if accepted_idx.numel() > 0:
                        slots = torch.randint(0, R, (accepted_idx.numel(),))
                        buf[h, c, slots] = rest[accepted_idx]
        count_buf[layer_idx] += N

    def update_k(self, layer_idx, k):
        self._update(layer_idx, k, self.max_k, self.samples_k, self.count_k)

    def update_v(self, layer_idx, v):
        self._update(layer_idx, v, self.max_v, self.samples_v, self.count_v)


def _get_layers(model):
    """Return the transformer layer ModuleList, walking past common wrappers.
    Handles plain ForCausalLM (model.model.layers), AutoModel base
    (model.layers), and multimodal wrappers like Exaone4_5_Model that hold
    the LM under .language_model (model.language_model.layers)."""
    for attr_chain in (("model", "layers"), ("language_model", "layers"),
                       ("model", "language_model", "layers"), ("layers",)):
        m = model
        ok = True
        for a in attr_chain:
            if not hasattr(m, a):
                ok = False
                break
            m = getattr(m, a)
        if ok:
            return m
    raise AttributeError(
        f"Could not find transformer layers on {type(model).__name__}; "
        f"tried .model.layers, .language_model.layers, .model.language_model.layers, .layers"
    )


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
    for i, layer in enumerate(_get_layers(model)):
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

    # State: track which layer is currently being processed.
    current_layer = [0]

    def wrap_forward(layer_idx, orig_fwd):
        def new_fwd(*args, **kwargs):
            current_layer[0] = layer_idx
            return orig_fwd(*args, **kwargs)
        return new_fwd

    for i, layer in enumerate(_get_layers(model)):
        layer.self_attn.forward = wrap_forward(i, layer.self_attn.forward)

    def _make_patch(orig):
        # transformers 4.x: apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
        # transformers 5.x: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)   # position_ids removed
        # Forward *args/**kwargs verbatim so we don't introduce spurious positional args.
        def patched(*args, **kwargs):
            qr, kr = orig(*args, **kwargs)
            li = current_layer[0]
            collector.update_q(li, qr)
            collector.update_k(li, kr)
            return qr, kr
        return patched

    # Patch every architecture we know about. Transformers loads modeling
    # modules lazily, so we guard each import.
    _targets = []
    try:
        from transformers.models.mistral import modeling_mistral as mm
        _targets.append(("mistral", mm))
    except Exception:
        pass
    try:
        from transformers.models.llama import modeling_llama as ll
        _targets.append(("llama", ll))
    except Exception:
        pass
    try:
        from transformers.models.qwen3 import modeling_qwen3 as qw
        _targets.append(("qwen3", qw))
    except Exception:
        pass
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as qwm
        _targets.append(("qwen3_moe", qwm))
    except Exception:
        pass
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qw2
        _targets.append(("qwen2", qw2))
    except Exception:
        pass
    try:
        from transformers.models.exaone4 import modeling_exaone4 as ex4
        _targets.append(("exaone4", ex4))
    except Exception:
        pass

    for _name, _mod in _targets:
        _mod.apply_rotary_pos_emb = _make_patch(_mod.apply_rotary_pos_emb)


def main():
    args = parse_args()
    device = args.device

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Multi-GPU auto-shard kicks in when --device is set to "auto" — used for
    # 33B+ models like EXAONE-4.5-33B that don't comfortably fit on a single 80GB.
    load_kwargs = dict(torch_dtype="auto", low_cpu_mem_usage=True)
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    # AutoModelForCausalLM works for plain text-only models. For multimodal
    # wrappers like Exaone4_5_ForConditionalGeneration (which holds the LM
    # under .language_model), AutoModelForCausalLM has no registry entry —
    # fall back to AutoModel and trust the LM weights still load. Calibration
    # hooks attach to modeling_exaone4.apply_rotary_pos_emb so they fire
    # regardless of which wrapper holds the language model.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM failed ({type(e).__name__}); trying AutoModel for multimodal wrapper")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, **load_kwargs)
    model = model if device == "auto" else model.to(device)
    model = model.eval()
    if device == "auto":
        device = "cuda:0"  # collector uses this for stat tensors

    cfg = model.config
    # Multimodal wrappers (e.g. Exaone4_5_Config) hold the LM hyperparams under
    # cfg.text_config; fall back to that if the top-level config lacks them.
    if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_q_heads)

    print(f"Model: {num_layers} layers, {num_q_heads} Q heads, "
          f"{num_kv_heads} KV heads, head_dim={head_dim}")

    # Detect Q/K-RMSNorm presence — relevant for downstream zero-runtime fusion:
    # models with q_norm/k_norm (Qwen3, Qwen3-MoE, Olmo2, …) require head-uniform
    # s_K because the gamma vectors q_norm.weight / k_norm.weight are shape
    # (head_dim,) shared across heads, and pre-norm row scaling on qkv_proj does
    # not survive RMSNorm. Models without q_norm/k_norm get full
    # per-(kv_head, channel) granularity via direct W_K row scaling.
    has_qk_norm = any(
        hasattr(m, "q_norm") and hasattr(m, "k_norm")
        and isinstance(getattr(m, "q_norm", None), torch.nn.Module)
        and isinstance(getattr(m, "k_norm", None), torch.nn.Module)
        for m in model.modules()
    )
    print(f"Q/K-RMSNorm: {has_qk_norm}  "
          f"(zero-runtime fusion {'requires' if has_qk_norm else 'does NOT need'} head-uniform s_K)")

    collector = StatCollector(
        num_layers, num_kv_heads, num_q_heads, head_dim, device,
        samples_per_channel=args.samples_per_channel,
    )
    monkey_patch_rope(model, collector)
    hooks = install_hooks(model, collector)

    print(f"Loading dataset: {args.dataset}")
    # Local file support: a path ending in .json[l] is loaded via the 'json' loader.
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    elif args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    print(f"Calibrating on {args.num_samples} samples at {args.seq_length} tokens"
          f"{' (chat-template applied)' if args.apply_chat_template else ''}...")
    with torch.no_grad():
        for i in tqdm(range(min(args.num_samples, len(ds)))):
            row = ds[i]
            if args.apply_chat_template:
                messages = row.get("messages")
                if not messages:
                    continue
                tmpl = tokenizer.apply_chat_template(
                    messages, tokenize=True, return_tensors="pt",
                    truncation=True, max_length=args.seq_length,
                    add_generation_prompt=False,
                )
                # Handle both Tensor and BatchEncoding return shapes
                if hasattr(tmpl, "input_ids"):
                    input_ids = tmpl.input_ids.to(device)
                else:
                    input_ids = tmpl.to(device)
                enc = {"input_ids": input_ids,
                       "attention_mask": torch.ones_like(input_ids)}
                if input_ids.shape[1] < 16:
                    continue
            else:
                if args.text_columns:
                    parts = [str(row[c]) for c in args.text_columns if row.get(c)]
                    text = args.text_join.join(parts)
                else:
                    text = row[args.text_column]
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
    # SmoothKV is invariant under any positive per-(layer, head) rescaling of
    # s_K (the same factor cancels across K/=s_K and Q*=s_K), so the overall
    # magnitude is a free parameter. We keep raw values — make_alpha_variants.py
    # is the single source of truth for downstream s_K and never normalized.
    s_K = (collector.max_k.clamp(min=eps) ** alpha) / \
          (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    s_V = collector.max_v.clamp(min=eps) ** beta

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "s_K": s_K.cpu(),
        "s_V": s_V.cpu(),
        "max_q":         collector.max_q.cpu(),
        "max_q_grouped": max_q_grouped.cpu(),
        "max_k":         collector.max_k.cpu(),
        "max_v":         collector.max_v.cpu(),
        "alpha": alpha,
        "beta": beta,
        "model_path": args.model,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "has_qk_norm": has_qk_norm,
    }
    if collector.samples_k is not None:
        payload["samples_k"] = collector.samples_k       # (L, nh, D, R) fp16 CPU
        payload["samples_v"] = collector.samples_v
        payload["count_k"]   = collector.count_k
        payload["count_v"]   = collector.count_v
    torch.save(payload, args.output)
    print(f"Saved calibration to {args.output}")
    print(f"  s_K range per layer: min={s_K.min():.4f}, max={s_K.max():.4f}, "
          f"mean={s_K.mean():.4f}")
    print(f"  s_V range per layer: min={s_V.min():.4f}, max={s_V.max():.4f}, "
          f"mean={s_V.mean():.4f}")


if __name__ == "__main__":
    main()
