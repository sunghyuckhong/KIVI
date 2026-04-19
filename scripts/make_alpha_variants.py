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
    p.add_argument("--out_dir", default=None, help="Default: same dir as --base")
    return p.parse_args()


def compute(max_k, max_q_grouped, max_v, alpha, beta, eps=1e-5):
    s_K = (max_k.clamp(min=eps) ** alpha) / (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    log_s_K = s_K.log()
    s_K = (log_s_K - log_s_K.mean(dim=-1, keepdim=True)).exp()

    s_V = max_v.clamp(min=eps) ** beta
    log_s_V = s_V.log()
    s_V = (log_s_V - log_s_V.mean(dim=-1, keepdim=True)).exp()
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

    if args.alphas:
        for a in args.alphas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=base["beta"])
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a}
            torch.save(payload, out)
            print(f"α={a:.2f} β={base['beta']:.2f}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")

    if args.betas:
        a = base["alpha"]
        for b in args.betas:
            s_K, s_V = compute(base["max_k"], base["max_q_grouped"], base["max_v"],
                               alpha=a, beta=b)
            out = os.path.join(out_dir, f"{stem}_a{fmt(a)}_b{fmt(b)}.pt")
            payload = {**base, "s_K": s_K, "s_V": s_V, "alpha": a, "beta": b}
            torch.save(payload, out)
            print(f"α={a:.2f} β={b:.2f}  →  {out}   "
                  f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
                  f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")


if __name__ == "__main__":
    main()
