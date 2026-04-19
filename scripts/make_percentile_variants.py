"""Offline: from a calibration .pt that includes reservoir samples (samples_k,
samples_v), generate new s_K / s_V as percentile-based scales per Eq 11 of
the SmoothKV paper. Writes one .pt per (p_K, p_V) pair.

Output filename: replace/append _p<pK>_<pV> to the base stem.

Usage:
  python scripts/make_percentile_variants.py \
      --base logs/calib/smoothkv_llama-2-7b-hf_perc.pt \
      --pk 95 99 99.9 --pv 95 99 99.9 --symmetric
"""
import argparse, os, re
import torch
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Calibration .pt with samples_k/samples_v (run calibrate "
                        "with --samples_per_channel first)")
    p.add_argument("--pk", nargs="+", type=float, required=True,
                   help="Percentile values for K (e.g. 95 99 99.9)")
    p.add_argument("--pv", nargs="+", type=float, default=None,
                   help="Percentile values for V. Default: same as --pk.")
    p.add_argument("--symmetric", action="store_true",
                   help="If set, only produce (p, p) pairs from --pk; ignores --pv.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def compute_percentile_scales(samples_k, samples_v, count_k, count_v, pk, pv, eps=1e-5):
    """samples_k, samples_v: (L, nh, D, R) fp16 CPU tensors.
    count_k, count_v: (L, nh, D) long — how many real values each reservoir holds
    (we only quantile over the valid prefix when count < R)."""
    L, nh, D, R = samples_k.shape

    def _quant(s, counts, p):
        # Per channel (layer, head, c), compute percentile over the valid values.
        out = torch.zeros(L, nh, D, dtype=torch.float32)
        for l in range(L):
            for h in range(nh):
                for c in range(D):
                    k = int(counts[l, h, c].item())
                    n = min(k, R)
                    if n == 0:
                        out[l, h, c] = eps
                        continue
                    vals = s[l, h, c, :n].to(torch.float32)
                    out[l, h, c] = torch.quantile(vals, p / 100.0)
        return out

    s_K = _quant(samples_k, count_k, pk).clamp(min=eps)
    s_V = _quant(samples_v, count_v, pv).clamp(min=eps)
    # Geo-mean normalize per (layer, head) — same as max-based calibration
    for s in (s_K, s_V):
        log_s = s.log()
        s.copy_((log_s - log_s.mean(dim=-1, keepdim=True)).exp())
    return s_K, s_V


def fmt_p(x):
    s = f"{x}".rstrip("0").rstrip(".") or "0"
    return s.replace(".", "p")


def main():
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    for k in ("samples_k", "samples_v", "count_k", "count_v"):
        if k not in base:
            raise RuntimeError(
                f"{args.base} missing '{k}'. Re-run calibration with "
                f"--samples_per_channel >= 1000 first."
            )

    out_dir = args.out_dir or os.path.dirname(args.base)
    base_stem = Path(args.base).stem
    stem = re.sub(r"_a[\d.]+(_b[\d.]+)?$", "", base_stem)
    stem = re.sub(r"_perc$", "", stem)

    if args.symmetric:
        pairs = [(p, p) for p in args.pk]
    else:
        pvs = args.pv if args.pv is not None else args.pk
        pairs = [(pk, pv) for pk in args.pk for pv in pvs]

    for pk, pv in pairs:
        s_K, s_V = compute_percentile_scales(
            base["samples_k"], base["samples_v"],
            base["count_k"],   base["count_v"],
            pk, pv,
        )
        tag = f"_pK{fmt_p(pk)}_pV{fmt_p(pv)}"
        out_name = f"{stem}{tag}.pt"
        out = os.path.join(out_dir, out_name)
        payload = {k: v for k, v in base.items()
                   if k not in ("samples_k", "samples_v", "count_k", "count_v",
                                "s_K", "s_V", "alpha", "beta")}
        payload["s_K"] = s_K
        payload["s_V"] = s_V
        payload["percentile_pk"] = pk
        payload["percentile_pv"] = pv
        torch.save(payload, out)
        print(f"pK={pk:>5}% pV={pv:>5}%  →  {out}   "
              f"s_K[{s_K.min():.3f}, {s_K.max():.3f}]   "
              f"s_V[{s_V.min():.3f}, {s_V.max():.3f}]")


if __name__ == "__main__":
    main()
