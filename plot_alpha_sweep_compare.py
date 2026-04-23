"""
Compare α sweep on ns=128 vs ns=512 calibration bases.

Overlays relRMSE(K) and relRMSE(V) for both calibration sizes.
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


def alpha_paths(prefix, alphas):
    paths = []
    for a in alphas:
        # make_alpha_variants writes "a1" not "a1.0" for α=1.0 exactly
        tag = "a1" if a == 1.0 else f"a{a}"
        paths.append(f"{CALIB_DIR}/smoothkv_{STEM}_{prefix}_{tag}_pair.pt")
    return paths


def sweep(K, V, prefix, alphas):
    rKs, rVs = [], []
    for a, p in zip(alphas, alpha_paths(prefix, alphas)):
        if not os.path.exists(p):
            print(f"  [{prefix}] α={a}  MISSING: {p}")
            rKs.append(None); rVs.append(None); continue
        rK, rV = mean_rmse(K, V, p)
        print(f"  [{prefix}] α={a:.2f}  K={rK:.4f}  V={rV:.4f}")
        rKs.append(rK); rVs.append(rV)
    return rKs, rVs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=f"{CALIB_DIR}/smoothkv_{STEM}_perc_ns512.pt",
                    help="Source of K/V samples for scoring (ns=512 has fresher/larger reservoir).")
    ap.add_argument("--subsample", type=int, default=2048)
    ap.add_argument("--output", default="logs/alpha_sweep_compare.png")
    args = ap.parse_args()

    print("Loading samples from ns=512 reservoir (larger, fresher)...")
    K, V = load_samples(args.calib)
    T = K.shape[2]
    if args.subsample < T:
        idx = torch.randperm(T, device=K.device)[:args.subsample]
        K, V = K[:, :, idx, :], V[:, :, idx, :]

    alphas = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    print("\nns=128 sweep:")
    rK_128, rV_128 = sweep(K, V, "perc", alphas)
    print("\nns=512 sweep:")
    rK_512, rV_512 = sweep(K, V, "perc_ns512", alphas)

    # References from ns=128 base
    refs = {
        "puremax":            f"{CALIB_DIR}/smoothkv_{STEM}_puremax_a1b1_pair.pt",
        "pairK99p9_pV99p9":   f"{CALIB_DIR}/smoothkv_{STEM}_pairK99p9_pV99p9.pt",
        "pairK99_pV99":       f"{CALIB_DIR}/smoothkv_{STEM}_pairK99_pV99.pt",
    }
    ref_rmses = {}
    for name, p in refs.items():
        if os.path.exists(p):
            rK, rV = mean_rmse(K, V, p)
            ref_rmses[name] = (rK, rV)
            print(f"  ref {name:25s}  K={rK:.4f}  V={rV:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rmses_128, rmses_512, ref_idx, title in zip(
            axes, [rK_128, rV_128], [rK_512, rV_512], [0, 1],
            ["K relRMSE vs α", "V relRMSE vs α"]):
        ax.plot(alphas, rmses_128, "o-", linewidth=2.2, markersize=8,
                color="#d62728", label="ns=128 (original calib)")
        ax.plot(alphas, rmses_512, "s-", linewidth=2.2, markersize=8,
                color="#1f77b4", label="ns=512 (larger calib)")
        colors = ["#8c564b", "#2ca02c", "#9467bd"]
        for (name, rmses), c in zip(ref_rmses.items(), colors):
            ax.axhline(rmses[ref_idx], color=c, linestyle="--", linewidth=1.2,
                       label=f"{name} ({rmses[ref_idx]:.3f})", alpha=0.8)
        ax.set_xlabel("α")
        ax.set_ylabel("mean relRMSE across 32 layers")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="best")
        ax.set_xticks(alphas)

    fig.suptitle("SmoothKV α sweep — ns=128 vs ns=512 calibration\n"
                 "s_K = max|K|^α / max|Q|^(1-α), β=0.5 fixed",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
