#!/usr/bin/env python3
"""
Derive per-layer NVFP4 global scales for K and V caches from existing
SmoothKV calibration .pt files.

NVFP4 formula (from compressed_tensors/quantization/utils/helpers.py:329):
    global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX / amax(tensor)
                 = 448.0 * 6.0 / amax(tensor)
                 = 2688.0 / amax(tensor)

The "tensor" here = the layer's K cache (or V cache) at the point of
NVFP4 quantization. Two cases:

  raw     — plain NVFP4 (no SmoothKV). amax = amax_per_layer(K_raw).
            Per-channel max already aggregated over calibration tokens
            during the forward pass:  amax = amax(max_k[h,d]).

  smooth  — SmoothKV + NVFP4. The runtime kernel divides K by s_K before
            quant:  K_q_in = K / s_K. Per-channel max of the input:
              max_k_smooth[h,d] = max_k[h,d] / s_K[h,d]
            Per-layer amax = amax over (h,d).

Standard NVFP4 (`compressed_tensors.quantization.utils.helpers.generate_global_scale`)
returns shape `[1]` per tensor. We produce one global_scale per layer
(per K, per V).

Usage:
  python scripts/derive_nvfp4_global_scales.py \\
    --input  logs/calib/smoothkv_qwen3-30b-a3b_perc_ns512_chat.pt \\
    --output logs/calib/nvfp4_global_scales_qwen3-30b-a3b_ns512_chat.pt
"""

import argparse
import sys
import torch

# NVFP4 standard formula (from compressed_tensors):
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
NVFP4_NUMERATOR = FP8_E4M3_MAX * FP4_E2M1_MAX  # = 2688.0

# SmoothKV calibration uses 1e-5 as the floor for s_K/s_V. If a calib has
# s_K uniformly at the floor AND max_k all-zero, the calib is broken
# (forward pass never observed K) — flag and skip the smooth derivation.
SK_FLOOR = 1e-5


def derive(calib: dict) -> dict:
    """Derive per-layer NVFP4 global scales from a SmoothKV calib dict."""
    for required in ('max_k', 'max_v', 's_K', 's_V'):
        if required not in calib:
            raise ValueError(
                f"Input calib missing {required!r}. "
                f"Got top-level keys: {list(calib.keys())}"
            )

    max_k = calib['max_k']  # (num_layers, kv_heads, head_dim)
    max_v = calib['max_v']
    s_K   = calib['s_K']
    s_V   = calib['s_V']

    if max_k.ndim != 3:
        raise ValueError(
            f"Expected max_k of ndim=3 (num_layers, kv_heads, head_dim), "
            f"got shape {tuple(max_k.shape)}"
        )

    num_layers = max_k.shape[0]

    amax_K_raw = max_k.abs().amax(dim=(1, 2))  # (num_layers,)
    amax_V_raw = max_v.abs().amax(dim=(1, 2))

    # Detect a broken calib: max_k all-zero AND s_K uniformly at the floor.
    k_broken = (max_k.abs().amax().item() == 0.0) and \
               (s_K.min().item() >= SK_FLOOR * 0.99 and s_K.max().item() <= SK_FLOOR * 1.01)
    v_broken = (max_v.abs().amax().item() == 0.0) and \
               (s_V.min().item() >= SK_FLOOR * 0.99 and s_V.max().item() <= SK_FLOOR * 1.01)

    if k_broken:
        print(
            "[WARN] K-side calib looks broken: max_k all-zero with s_K "
            f"uniformly at the floor ({SK_FLOOR}). Smooth gs_K_smooth will be "
            "garbage; raw gs_K is meaningless too.",
            file=sys.stderr,
        )
    if v_broken:
        print(
            "[WARN] V-side calib looks broken: max_v all-zero with s_V "
            f"uniformly at the floor ({SK_FLOOR}).",
            file=sys.stderr,
        )

    # Per-channel post-smoothing max: max_k / s_K (avoid div-by-zero).
    sk_safe = s_K.abs().clamp(min=1e-30)
    sv_safe = s_V.abs().clamp(min=1e-30)
    max_k_smooth = max_k.abs() / sk_safe
    max_v_smooth = max_v.abs() / sv_safe
    amax_K_smooth = max_k_smooth.amax(dim=(1, 2))
    amax_V_smooth = max_v_smooth.amax(dim=(1, 2))

    gs_K_raw    = NVFP4_NUMERATOR / amax_K_raw.clamp(min=1e-30)
    gs_V_raw    = NVFP4_NUMERATOR / amax_V_raw.clamp(min=1e-30)
    gs_K_smooth = NVFP4_NUMERATOR / amax_K_smooth.clamp(min=1e-30)
    gs_V_smooth = NVFP4_NUMERATOR / amax_V_smooth.clamp(min=1e-30)

    return {
        # SmoothKV+NVFP4 (input to NVFP4 = K/s_K)
        'gs_K_smooth':    gs_K_smooth.float(),
        'gs_V_smooth':    gs_V_smooth.float(),
        'amax_K_smooth':  amax_K_smooth.float(),
        'amax_V_smooth':  amax_V_smooth.float(),
        # NVFP4-only (input = K_raw)
        'gs_K_raw':       gs_K_raw.float(),
        'gs_V_raw':       gs_V_raw.float(),
        'amax_K_raw':     amax_K_raw.float(),
        'amax_V_raw':     amax_V_raw.float(),
        # Bookkeeping
        'num_layers':     num_layers,
        'fp4_e2m1_max':   FP4_E2M1_MAX,
        'fp8_e4m3_max':   FP8_E4M3_MAX,
        'k_calib_broken': k_broken,
        'v_calib_broken': v_broken,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='SmoothKV calib .pt (base, e.g. *_perc_ns512_chat.pt)')
    parser.add_argument('--output', required=True,
                        help='NVFP4 global scales output .pt')
    args = parser.parse_args()

    calib = torch.load(args.input, map_location='cpu', weights_only=True)

    out = derive(calib)
    out['derived_from'] = args.input
    out['model_path']   = calib.get('model_path', 'unknown')
    out['num_samples']  = calib.get('num_samples', 'unknown')

    torch.save(out, args.output)

    def stat(name, t):
        return (f"  {name:14s} range [{t.min().item():.4f}, "
                f"{t.max().item():.4f}]  mean {t.mean().item():.4f}")

    print(f"Saved {args.output}")
    print(f"  num_layers: {out['num_layers']}")
    print(stat('amax_K_raw',   out['amax_K_raw']))
    print(stat('amax_V_raw',   out['amax_V_raw']))
    print(stat('amax_K_smooth', out['amax_K_smooth']))
    print(stat('amax_V_smooth', out['amax_V_smooth']))
    print(stat('gs_K_raw',     out['gs_K_raw']))
    print(stat('gs_V_raw',     out['gs_V_raw']))
    print(stat('gs_K_smooth',  out['gs_K_smooth']))
    print(stat('gs_V_smooth',  out['gs_V_smooth']))
    if out['k_calib_broken'] or out['v_calib_broken']:
        print(f"  WARN: k_calib_broken={out['k_calib_broken']}  "
              f"v_calib_broken={out['v_calib_broken']}", file=sys.stderr)


if __name__ == '__main__':
    main()
