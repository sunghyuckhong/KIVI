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
                   help="Pair-max s_K over adjacent channels (2i, 2i+1) to satisfy "
                        "the Eq. 2 pair-equal constraint (mergeable form). Appends "
                        "'_pair' to the calib filename.")
    p.add_argument("--out_dir", default=None, help="Default: same dir as --base")
    return p.parse_args()


def compute(max_k, max_q_grouped, max_v, alpha, beta, pair_max_k=False, eps=1e-5):
    s_K = (max_k.clamp(min=eps) ** alpha) / (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    if pair_max_k:
        L, nh, D = s_K.shape
        assert D % 2 == 0
        s_K_pairs = s_K.view(L, nh, D // 2, 2).max(dim=-1, keepdim=True).values
        s_K = s_K_pairs.expand(L, nh, D // 2, 2).reshape(L, nh, D).clone()
    s_V = max_v.clamp(min=eps) ** beta
    return s_K, s_V


def main():
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    for k in ("max_k", "max_q_grouped", "max_v"):
        if k not in base:
            raise RuntimeError(f"{args.base} missing raw '{k}'. Re-run the calibration with "
                               f"the patched run_smoothkv_calibrate.py first.")

    out_dir = args.out_dir or os.path.dirname(args.base)
    base_name = Path(args.base).stem  # e.g. smoothkv_mistral-7b-v0.1_a0.5
    import re
    # strip existing alpha/beta suffixes from the stem
    stem = re.sub(r"_a[\d.]+(_b[\d.]+)?$", "", base_name)

    def fmt(x):
        return (f"{x}".rstrip("0").rstrip(".") or "0")

    pair_suffix = "_pair" if args.pair_max_k else ""

    if args.alphas:
        for a in args.alphas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=base["beta"],
                               pair_max_k=args.pair_max_k)
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}{pair_suffix}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a,
                       "pair_max_k": args.pair_max_k}
            if args.pair_max_k:
                diff = (s_K[..., 0::2] - s_K[..., 1::2]).abs().max().item()
                assert diff < 1e-6, f"pair-max s_K violates pair-equal: diff={diff}"
            torch.save(payload, out)
            print(f"α={a:.2f} β={base['beta']:.2f} pair={args.pair_max_k}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")

    if args.betas:
        a = base["alpha"]
        for b in args.betas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=b,
                               pair_max_k=args.pair_max_k)
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}_b{fmt(b)}{pair_suffix}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a, "beta": b,
                       "pair_max_k": args.pair_max_k}
            torch.save(payload, out)
            print(f"α={a:.2f} β={b:.2f} pair={args.pair_max_k}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")


if __name__ == "__main__":
    main()
