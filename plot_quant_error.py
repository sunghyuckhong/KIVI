"""
Plot per-layer relRMSE for the key quant methods, using the same samples and
quant hooks as screen_quant_error.py. Saves a PNG to logs/.
"""
import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from screen_quant_error import (
    load_samples,
    load_smoothkv_scales,
    per_layer_rmse,
    method_fp8,
    method_pertoken,
    method_smoothkv,
)


CALIB_DIR = "logs/calib"

# (label, color, fn)
METHODS = [
    ("fp8 g=128",                   "#1f77b4",
        lambda K, V: method_fp8(K, V, 128)),
    ("pertoken int4 g=128",         "#ff7f0e",
        lambda K, V: method_pertoken(K, V, 4, 128)),
    ("pertoken int4 g=32",          "#d62728",
        lambda K, V: method_pertoken(K, V, 4, 32)),
    ("smoothkv α=0.75 pair (K+V, classical)",  "#17becf",
        lambda K, V: method_smoothkv(K, V,
            f"{CALIB_DIR}/smoothkv_meta-llama-3-8b-instruct_perc_a0.75_pair.pt", 4, 128)),
    ("smoothkv pairK99p9_pV99p9 (K+V)",         "#2ca02c",
        lambda K, V: method_smoothkv(K, V,
            f"{CALIB_DIR}/smoothkv_meta-llama-3-8b-instruct_pairK99p9_pV99p9.pt", 4, 128)),
    ("smoothkv puremax_a1b1 (K+V)",             "#8c564b",
        lambda K, V: method_smoothkv(K, V,
            f"{CALIB_DIR}/smoothkv_meta-llama-3-8b-instruct_puremax_a1b1_pair.pt", 4, 128)),
    ("smoothkv kOnly_p99p9 (K only, V=1)",      "#9467bd",
        lambda K, V: method_smoothkv(K, V,
            f"{CALIB_DIR}/smoothkv_meta-llama-3-8b-instruct_kOnly_p99p9_pair.pt", 4, 128)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=f"{CALIB_DIR}/smoothkv_meta-llama-3-8b-instruct_perc.pt")
    ap.add_argument("--subsample", type=int, default=2048)
    ap.add_argument("--output", default="logs/relrmse_per_layer.png")
    args = ap.parse_args()

    print("Loading samples...")
    K, V = load_samples(args.calib)
    T = K.shape[2]
    if args.subsample < T:
        idx = torch.randperm(T, device=K.device)[:args.subsample]
        K, V = K[:, :, idx, :], V[:, :, idx, :]
    L = K.shape[0]
    print(f"K {tuple(K.shape)}  V {tuple(V.shape)}")

    results = []
    for label, color, fn in METHODS:
        print(f"  {label} ...")
        K_q, V_q = fn(K, V)
        rK = per_layer_rmse(K, K_q).numpy()
        rV = per_layer_rmse(V, V_q).numpy()
        results.append((label, color, rK, rV))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    layers = list(range(L))

    for label, color, rK, rV in results:
        # kOnly overlaps pairK99p9 on K (same s_K). Dash it so both are visible.
        ls = "--" if "kOnly" in label else "-"
        axes[0].plot(layers, rK, marker="o", markersize=3.5, linewidth=1.5,
                     linestyle=ls,
                     label=f"{label}  (μ={rK.mean():.3f})", color=color)
        axes[1].plot(layers, rV, marker="o", markersize=3.5, linewidth=1.5,
                     linestyle=ls,
                     label=f"{label}  (μ={rV.mean():.3f})", color=color)

    for ax, title in zip(axes, ["K relRMSE per layer", "V relRMSE per layer"]):
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylabel("relRMSE = ||x - x̂||_F / ||x||_F")
    axes[1].set_xlabel("layer index")

    fig.suptitle("Per-layer relative quantization error on real K/V samples "
                 "(Llama-3-8B-Instruct)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
