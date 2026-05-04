"""
Quick-estimation screener: compute per-layer relRMSE for each KV-quant method
using real K/V samples stored in the calibration file. No forward passes, no
generation — ranks variants in ~30s.

Usage:
    python screen_quant_error.py                 # runs a default method panel
    python screen_quant_error.py --calib logs/calib/smoothkv_<model>_perc.pt
"""
import argparse
import glob
import os
import torch

from vllm.model_executor.layers.quantization.kv_fake_quant import (
    fake_quantize_fp8,
    fake_quantize_k_pertoken,
    fake_quantize_v_pertoken,
    fake_quantize_k_perchannel,
)


def load_samples(calib_path):
    """Return K, V tensors of shape (L, nh, T, D) using the reservoir samples."""
    calib = torch.load(calib_path, weights_only=True, map_location="cpu")
    # samples_k/v: (L, nh, D, N) → transpose to (L, nh, N, D)
    K = calib["samples_k"].transpose(-1, -2).contiguous().to(torch.float16).cuda()
    V = calib["samples_v"].transpose(-1, -2).contiguous().to(torch.float16).cuda()
    return K, V  # (L, nh, T, D)


def load_smoothkv_scales(calib_path):
    d = torch.load(calib_path, weights_only=True, map_location="cpu")
    return d["s_K"].to(torch.float16).cuda(), d["s_V"].to(torch.float16).cuda()  # (L, nh, D)


def per_layer_rmse(x, x_q):
    """Per-layer relative RMSE: ||x - x_q||_F / ||x||_F per layer."""
    L = x.shape[0]
    err = (x - x_q).float().view(L, -1)
    norm = x.float().view(L, -1).norm(dim=1)
    return (err.norm(dim=1) / norm).cpu()  # (L,)


def quant_passthrough_per_layer(K, V, quant_fn_k, quant_fn_v):
    """Apply quant→dequant per layer and return reconstructed (K_q, V_q)."""
    L, nh, T, D = K.shape
    K_q = torch.empty_like(K)
    V_q = torch.empty_like(V)
    for l in range(L):
        # vLLM utils expect (T, nh*D) layout
        kl = K[l].transpose(0, 1).reshape(T, nh * D)  # (T, nh*D)
        vl = V[l].transpose(0, 1).reshape(T, nh * D)
        kl_q = quant_fn_k(kl)
        vl_q = quant_fn_v(vl)
        K_q[l] = kl_q.view(T, nh, D).transpose(0, 1)
        V_q[l] = vl_q.view(T, nh, D).transpose(0, 1)
    return K_q, V_q


def method_fp16(K, V, **kw):
    return K.clone(), V.clone()


def method_fp8(K, V, group_size=128, **kw):
    nh, D = K.shape[1], K.shape[3]
    fn = lambda x: fake_quantize_fp8(x, nh, D, group_size)
    return quant_passthrough_per_layer(K, V, fn, fn)


def method_pertoken(K, V, bits=4, group_size=128, **kw):
    nh, D = K.shape[1], K.shape[3]
    kf = lambda x: fake_quantize_k_pertoken(x, nh, D, group_size, bits)
    vf = lambda x: fake_quantize_v_pertoken(x, nh, D, group_size, bits)
    return quant_passthrough_per_layer(K, V, kf, vf)


def method_smoothkv(K, V, calib_path, bits=4, group_size=128, **kw):
    """Apply K/s_K, V/s_V → quant → *s_K, *s_V layer by layer."""
    nh, D = K.shape[1], K.shape[3]
    sK_all, sV_all = load_smoothkv_scales(calib_path)  # (L, nh, D)
    K_q = torch.empty_like(K)
    V_q = torch.empty_like(V)
    L, _, T, _ = K.shape
    for l in range(L):
        sk_flat = sK_all[l].reshape(-1)  # (nh*D,)
        sv_flat = sV_all[l].reshape(-1)
        kl = (K[l].transpose(0, 1).reshape(T, nh * D)) / sk_flat
        vl = (V[l].transpose(0, 1).reshape(T, nh * D)) / sv_flat
        kl_q = fake_quantize_k_pertoken(kl, nh, D, group_size, bits)
        vl_q = fake_quantize_v_pertoken(vl, nh, D, group_size, bits)
        kl_q = kl_q * sk_flat
        vl_q = vl_q * sv_flat
        K_q[l] = kl_q.view(T, nh, D).transpose(0, 1)
        V_q[l] = vl_q.view(T, nh, D).transpose(0, 1)
    return K_q, V_q


def build_panel(calib_dir, base_calib):
    """Return list of (display_name, fn) pairs."""
    panel = [
        ("fp16",                            lambda K, V: method_fp16(K, V)),
        ("fp8 g=128",                       lambda K, V: method_fp8(K, V, 128)),
        ("fp8 g=32",                        lambda K, V: method_fp8(K, V, 32)),
        ("pertoken int4 g=128",             lambda K, V: method_pertoken(K, V, 4, 128)),
        ("pertoken int4 g=32",              lambda K, V: method_pertoken(K, V, 4, 32)),
        ("pertoken int2 g=128",             lambda K, V: method_pertoken(K, V, 2, 128)),
        ("pertoken int2 g=32",              lambda K, V: method_pertoken(K, V, 2, 32)),
    ]
    # Auto-add every smoothkv variant found in calib_dir
    for f in sorted(glob.glob(os.path.join(calib_dir, "smoothkv_*_pair*.pt"))):
        tag = os.path.basename(f).replace("smoothkv_", "").replace(".pt", "")
        if "perc" in tag and "_pair" not in tag:
            continue
        panel.append((f"smoothkv {tag}", lambda K, V, f=f: method_smoothkv(K, V, f, 4, 128)))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="logs/calib/smoothkv_meta-llama-3-8b-instruct_perc.pt",
                    help="Base calib file containing samples_k / samples_v reservoirs.")
    ap.add_argument("--calib_dir", default="logs/calib",
                    help="Directory of smoothkv_*_pair*.pt variants to include.")
    ap.add_argument("--subsample", type=int, default=2048,
                    help="Use N samples per (layer, head, channel) for speed (cap 10000).")
    args = ap.parse_args()

    print(f"Loading samples from {args.calib} ...")
    K, V = load_samples(args.calib)
    # Subsample along the T axis
    T = K.shape[2]
    if args.subsample < T:
        idx = torch.randperm(T, device=K.device)[:args.subsample]
        K = K[:, :, idx, :]
        V = V[:, :, idx, :]
    print(f"K shape={tuple(K.shape)}  V shape={tuple(V.shape)}")

    panel = build_panel(args.calib_dir, args.calib)

    print()
    print(f"{'method':55s}  {'K relRMSE':>10s}  {'V relRMSE':>10s}  {'combined':>10s}")
    print("-" * 92)
    rows = []
    for name, fn in panel:
        K_q, V_q = fn(K, V)
        rK = per_layer_rmse(K, K_q).mean().item()
        rV = per_layer_rmse(V, V_q).mean().item()
        combined = (rK + rV) / 2
        rows.append((name, rK, rV, combined))
    # Sort ascending by combined (except fp16 first)
    rows.sort(key=lambda r: (r[0] != "fp16", r[3]))
    for name, rK, rV, c in rows:
        print(f"{name:55s}  {rK:>10.4f}  {rV:>10.4f}  {c:>10.4f}")


if __name__ == "__main__":
    main()
