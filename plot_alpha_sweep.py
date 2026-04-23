"""
Plot relRMSE vs α for the classical SmoothKV formula:
    s_K = max|K|^α / max|Q|^(1-α)
    s_V = max|V|^β   (β=0.5 fixed)

Reads alpha variants from logs/calib/ and plots mean K/V relRMSE vs α.
Also overlays the two percentile-based pair variants and puremax as references.
"""
import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from screen_quant_error import load_samples, per_layer_rmse, method_smoothkv

CALIB_DIR = "logs/calib"
STEM = "meta-llama-3-8b-instruct"


def mean_rmse(K, V, calib_path, bits=4, group_size=128):
    K_q, V_q = method_smoothkv(K, V, calib_path, bits, group_size)
    return per_layer_rmse(K, K_q).mean().item(), per_layer_rmse(V, V_q).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=f"{CALIB_DIR}/smoothkv_{STEM}_perc.pt")
    ap.add_argument("--subsample", type=int, default=2048)
    ap.add_argument("--output", default="logs/alpha_sweep.png")
    args = ap.parse_args()

    print("Loading samples...")
    K, V = load_samples(args.calib)
    T = K.shape[2]
    if args.subsample < T:
        idx = torch.randperm(T, device=K.device)[:args.subsample]
        K, V = K[:, :, idx, :], V[:, :, idx, :]

    alphas = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    alpha_paths = [f"{CALIB_DIR}/smoothkv_{STEM}_perc_a{a}_pair.pt" for a in alphas]
    # fix filename for a=1.0 (written as "a1" not "a1.0")
    alpha_paths[-1] = f"{CALIB_DIR}/smoothkv_{STEM}_perc_a1_pair.pt"

    print("\nα sweep (classical: s_K = max|K|^α / max|Q|^(1-α)):")
    rKs, rVs = [], []
    for a, p in zip(alphas, alpha_paths):
        if not os.path.exists(p):
            print(f"  α={a}  MISSING: {p}")
            rKs.append(None); rVs.append(None); continue
        rK, rV = mean_rmse(K, V, p)
        print(f"  α={a}  K={rK:.4f}  V={rV:.4f}")
        rKs.append(rK); rVs.append(rV)

    # Reference points
    refs = {
        "puremax (α=1 no /Q)":   f"{CALIB_DIR}/smoothkv_{STEM}_puremax_a1b1_pair.pt",
        "pairK99p9_pV99p9":      f"{CALIB_DIR}/smoothkv_{STEM}_pairK99p9_pV99p9.pt",
        "pairK99_pV99":          f"{CALIB_DIR}/smoothkv_{STEM}_pairK99_pV99.pt",
    }
    ref_rmses = {}
    for name, p in refs.items():
        if os.path.exists(p):
            rK, rV = mean_rmse(K, V, p)
            ref_rmses[name] = (rK, rV)
            print(f"  ref {name:30s}  K={rK:.4f}  V={rV:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, ref_idx, title in zip(
            axes, [rKs, rVs], [0, 1],
            ["K relRMSE vs α", "V relRMSE vs α"]):
        ax.plot(alphas, vals, "o-", linewidth=2, markersize=8,
                color="#d62728", label="α sweep (classical formula)")
        # Reference horizontals
        colors = ["#8c564b", "#2ca02c", "#1f77b4"]
        for (name, rmses), c in zip(ref_rmses.items(), colors):
            ax.axhline(rmses[ref_idx], color=c, linestyle="--", linewidth=1.3,
                       label=f"{name} ({rmses[ref_idx]:.3f})", alpha=0.85)
        ax.set_xlabel("α")
        ax.set_ylabel("mean relRMSE across 32 layers")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        ax.set_xticks(alphas)

    fig.suptitle("SmoothKV α sweep — classical formula s_K = max|K|^α / max|Q|^(1-α)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
