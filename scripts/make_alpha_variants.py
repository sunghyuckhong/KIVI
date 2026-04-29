"""Offline: from a calibration .pt that includes raw max stats, generate
new s_K / s_V for a grid of alpha values. Writes one .pt per alpha.

Usage:
  python scripts/make_alpha_variants.py \
      --base logs/calib/smoothkv_mistral-7b-v0.1_a0.5.pt \
      --alphas 0.25 0.75 1.0
"""
import argparse, os, torch
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Calibration .pt with raw max_q/max_k/max_v")
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="Generate calib variants at these α values (β fixed to base).")
    p.add_argument("--betas", nargs="+", type=float, default=None,
                   help="Generate calib variants at these β values (α fixed to base).")
    p.add_argument("--pair_max_k", dest="pair_max_k", action="store_true",
                   default=False,
                   help="Pair-max s_K over ADJACENT channels (2i, 2i+1) — satisfies "
                        "the Eq. 2 pair-equal constraint for INTERLEAVED / GPTNeoX RoPE "
                        "(KIVI's original Llama-2 / older models). Appends '_pair' to "
                        "the calib filename. NOT correct for modern rotate_half models.")
    p.add_argument("--half_pair_max_k", dest="half_pair_max_k", action="store_true",
                   default=False,
                   help="Half-pair-max s_K over channel pairs (i, i+d/2) — satisfies "
                        "the pair-equal constraint for `rotate_half` RoPE used by Llama-3, "
                        "Mistral, DSR1-distill, Qwen3, Qwen3-MoE, and most modern HF models. "
                        "Appends '_halfpair' to the calib filename. Use this for any "
                        "fusion-style SmoothKV (RoPE-merge, gamma-merge, pre-RoPE weight "
                        "merge) on rotate_half models.")
    huk_grp = p.add_mutually_exclusive_group()
    huk_grp.add_argument("--head_uniform_k", dest="head_uniform_k", action="store_true",
                   default=None,
                   help="Reduce s_K to be head-uniform (shared across all kv heads) by taking "
                        "max over the kv-heads dimension. Required for q_norm/k_norm models "
                        "(Qwen3 / Qwen3-MoE) where the only zero-runtime fusion path is to "
                        "absorb s_K into q_norm.gamma / k_norm.gamma — and those gammas have "
                        "shape (head_dim,) shared across heads, so s_K must also be shared "
                        "across heads. Output shape stays (L, num_kv_heads, head_dim) but all "
                        "heads carry the same row. Appends '_huk' to the filename. "
                        "If unset, AUTO: enabled when the base calib reports `has_qk_norm=True` "
                        "(set at calibration time) or when AutoConfig.model_type matches a known "
                        "qk_norm architecture (qwen3, qwen3_moe, olmo2, …).")
    huk_grp.add_argument("--no_head_uniform_k", dest="head_uniform_k", action="store_false",
                         help="Force-disable the head-uniform reduction even on a qk_norm model "
                              "(use only when explicitly building a per-(head, channel) calib for "
                              "research / non-zero-runtime paths).")
    p.add_argument("--out_dir", default=None, help="Default: same dir as --base")
    args = p.parse_args()
    if args.pair_max_k and args.half_pair_max_k:
        p.error("--pair_max_k and --half_pair_max_k are mutually exclusive (different RoPE conventions)")
    return args


# Architectures whose attention has Q/K-RMSNorm between qkv_proj and RoPE.
# For these models the only zero-runtime SmoothKV fusion path is to fold s_K
# into the (head_dim,) gamma vectors of q_norm/k_norm, which forces head-uniform
# s_K. Used as a fallback when the base calib doesn't carry `has_qk_norm`.
_QK_NORM_MODEL_TYPES = {"qwen3", "qwen3_moe", "olmo2"}


def detect_has_qk_norm(base_payload):
    """Return True if the base calib was produced from a model with Q/K-norm.

    Resolution order:
      1. Explicit `has_qk_norm` field saved at calibration time (preferred).
      2. AutoConfig(model_path).model_type lookup against `_QK_NORM_MODEL_TYPES`.
      3. None (caller falls back to no head-uniform reduction).
    """
    if "has_qk_norm" in base_payload:
        return bool(base_payload["has_qk_norm"])
    mp = base_payload.get("model_path")
    if not mp:
        return None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
        mt = getattr(cfg, "model_type", None) or ""
        return mt.lower() in _QK_NORM_MODEL_TYPES
    except Exception as e:
        print(f"  (auto-detect qk_norm failed for {mp!r}: {e}; assuming False)")
        return None


def compute(max_k, max_q_grouped, max_v, alpha, beta,
            pair_max_k=False, half_pair_max_k=False, head_uniform_k=False, eps=1e-5):
    s_K = (max_k.clamp(min=eps) ** alpha) / (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    if pair_max_k:
        # Adjacent-pair-max: s_K[2i] = s_K[2i+1] (interleaved/GPTNeoX RoPE)
        L, nh, D = s_K.shape
        assert D % 2 == 0
        s_K_pairs = s_K.view(L, nh, D // 2, 2).max(dim=-1, keepdim=True).values
        s_K = s_K_pairs.expand(L, nh, D // 2, 2).reshape(L, nh, D).clone()
    elif half_pair_max_k:
        # Half-pair-max: s_K[i] = s_K[i+d/2] (rotate_half RoPE, Llama-3 / Mistral / Qwen3 / DSR1)
        L, nh, D = s_K.shape
        assert D % 2 == 0
        half_max = torch.maximum(s_K[..., :D // 2], s_K[..., D // 2:])
        s_K = torch.cat([half_max, half_max], dim=-1).contiguous()
    if head_uniform_k:
        # Head-uniform: s_K shared across all kv heads — required when fusing into
        # q_norm.gamma / k_norm.gamma (shape (head_dim,) shared across heads).
        # We max over the kv-heads dim and broadcast back so the output shape stays
        # (L, num_kv_heads, head_dim).
        L, nh, D = s_K.shape
        s_K_per_layer = s_K.max(dim=1, keepdim=True).values  # (L, 1, D)
        s_K = s_K_per_layer.expand(L, nh, D).contiguous()
    s_V = max_v.clamp(min=eps) ** beta
    return s_K, s_V


def main():
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    for k in ("max_k", "max_q_grouped", "max_v"):
        if k not in base:
            raise RuntimeError(f"{args.base} missing raw '{k}'. Re-run the calibration with "
                               f"the patched run_smoothkv_calibrate.py first.")

    # Auto-resolve --head_uniform_k when the user didn't pass an explicit choice.
    # qk_norm models (Qwen3, Qwen3-MoE, Olmo2, …) MUST use head-uniform s_K because
    # the only zero-runtime fusion path absorbs s_K into q_norm/k_norm gammas of
    # shape (head_dim,) shared across heads. Non-qk_norm models default to
    # per-(kv_head, channel) granularity (head_uniform_k=False).
    if args.head_uniform_k is None:
        detected = detect_has_qk_norm(base)
        if detected is True:
            args.head_uniform_k = True
            print(f"AUTO --head_uniform_k=True  (base reports/looks-like a qk_norm model: "
                  f"{base.get('model_path', '?')!r}). Use --no_head_uniform_k to override.")
        else:
            args.head_uniform_k = False
            mp = base.get("model_path", "?")
            print(f"AUTO --head_uniform_k=False (base is non-qk_norm: {mp!r}). "
                  f"Use --head_uniform_k to force on.")

    out_dir = args.out_dir or os.path.dirname(args.base)
    base_name = Path(args.base).stem  # e.g. smoothkv_mistral-7b-v0.1_a0.5
    import re
    # strip variant tags that may have been appended on a previous pass
    # (e.g. "_a1b1_halfpair_slim") so the new output filename stays clean.
    stem = base_name
    for pat in (
        r"_slim$",
        r"_(huk_)?(half)?pair$",
        r"_a[\d.]+(?:_?b[\d.]+)?$",   # strip "_a1", "_a1b1", "_a0.5_b0.75", etc.
    ):
        stem = re.sub(pat, "", stem)

    def fmt(x):
        return (f"{x}".rstrip("0").rstrip(".") or "0")

    if args.pair_max_k:
        pair_suffix = "_pair"
    elif args.half_pair_max_k:
        pair_suffix = "_halfpair"
    else:
        pair_suffix = ""
    if args.head_uniform_k:
        pair_suffix = "_huk" + pair_suffix  # e.g. "_huk_halfpair"

    def _validate(s_K):
        if args.pair_max_k:
            diff = (s_K[..., 0::2] - s_K[..., 1::2]).abs().max().item()
            assert diff < 1e-6, f"adjacent-pair-max s_K violates pair-equal: diff={diff}"
        if args.half_pair_max_k:
            D = s_K.shape[-1]
            diff = (s_K[..., :D//2] - s_K[..., D//2:]).abs().max().item()
            assert diff < 1e-6, f"half-pair-max s_K violates pair-equal (i, i+d/2): diff={diff}"
        if args.head_uniform_k:
            head_diff = (s_K - s_K[:, :1]).abs().max().item()
            assert head_diff < 1e-6, f"head-uniform s_K violates head-shared: diff={head_diff}"

    common_kwargs = dict(
        pair_max_k=args.pair_max_k,
        half_pair_max_k=args.half_pair_max_k,
        head_uniform_k=args.head_uniform_k,
    )

    if args.alphas:
        for a in args.alphas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=base["beta"], **common_kwargs)
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}{pair_suffix}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a,
                       "pair_max_k": args.pair_max_k,
                       "half_pair_max_k": args.half_pair_max_k,
                       "head_uniform_k": args.head_uniform_k}
            _validate(s_K)
            torch.save(payload, out)
            print(f"α={a:.2f} β={base['beta']:.2f} pair={args.pair_max_k} half_pair={args.half_pair_max_k} "
                  f"huk={args.head_uniform_k}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")

    if args.betas:
        a = base["alpha"]
        for b in args.betas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=b, **common_kwargs)
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}_b{fmt(b)}{pair_suffix}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a, "beta": b,
                       "pair_max_k": args.pair_max_k,
                       "half_pair_max_k": args.half_pair_max_k,
                       "head_uniform_k": args.head_uniform_k}
            _validate(s_K)
            torch.save(payload, out)
            print(f"α={a:.2f} β={b:.2f} pair={args.pair_max_k} half_pair={args.half_pair_max_k} "
                  f"huk={args.head_uniform_k}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")


if __name__ == "__main__":
    main()
