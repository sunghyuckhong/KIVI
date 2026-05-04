#!/usr/bin/env python3
"""Verify that vLLM's torch.compile captured our patched forward.

After install_method() patches Qwen3Attention.forward, vLLM traces through
the patched code and inductor emits a `computation_graph.py` with source-line
annotations pointing back to the original Python files. We can grep that
file to confirm our quant kernels are in the captured graph (i.e., the patch
wasn't silently bypassed by a stale compile cache, plugin override, or
graph break).

Per-variant signatures we look for:
    bf16/fp16     → NO quant calls (baseline; must not see fake_quantize_*)
    fp8           → fake_quantize_dequantize_fp8
    pertoken      → quant_and_pack_vcache (the int4 kernel)
    smoothkv      → quant_and_pack_vcache (post-smoothing K/V quant)
    smoothkv_fused → quant_and_pack_vcache (after k_norm γ fusion)

Returns 0 if all expected signatures present and no forbidden ones; 1 otherwise.

Usage:
    python scripts/verify_compiled_graph.py --variant pertoken
    python scripts/verify_compiled_graph.py --variant smoothkv_fused --cache_root /tmp/vllm_cache_1746040
"""
import argparse
import glob
import os
import sys


EXPECTED = {
    "bf16":           [],
    "fp16":           [],
    "fp8":            ["fake_quantize_dequantize_fp8"],
    "pertoken":       ["quant_and_pack_vcache", "unpack_and_dequant_vcache"],
    "smoothkv":       ["quant_and_pack_vcache"],
    "smoothkv_fused": ["quant_and_pack_vcache"],
}

FORBIDDEN = {
    "bf16": ["quant_and_pack", "fake_quantize"],
    "fp16": ["quant_and_pack", "fake_quantize"],
}


def find_graph_files(cache_root: str):
    patterns = [
        f"{cache_root}/torch_compile_cache/*/rank_0_0/*/computation_graph.py",
        f"{cache_root}/torch_compile_cache/*/computation_graph.py",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return files


def verify(variant: str, cache_root: str, verbose: bool = True) -> bool:
    expected = EXPECTED.get(variant, None)
    if expected is None:
        if verbose:
            print(f"[verify-graph] unknown variant {variant!r}", file=sys.stderr)
        return False
    forbidden = FORBIDDEN.get(variant, [])

    graph_files = find_graph_files(cache_root)
    if not graph_files:
        if verbose:
            print(f"[verify-graph] ❌ no computation_graph.py found in {cache_root}",
                  file=sys.stderr)
        return False

    all_ok = True
    for gf in graph_files:
        try:
            content = open(gf).read()
        except OSError:
            continue
        missing = [s for s in expected if s not in content]
        present_forbidden = [s for s in forbidden if s in content]
        ok = not missing and not present_forbidden
        if verbose:
            tag = "[PASS]" if ok else "[FAIL]"
            print(f"[verify-graph] {tag} variant={variant} graph={gf}")
            if expected:
                print(f"  expected:  {expected}")
            if missing:
                print(f"  MISSING:   {missing}")
            if present_forbidden:
                print(f"  FORBIDDEN PRESENT (silent quant on baseline!): {present_forbidden}")
        if not ok:
            all_ok = False
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   help="bf16 | fp16 | fp8 | pertoken | smoothkv | smoothkv_fused")
    p.add_argument("--cache_root", default=None,
                   help="Path to per-PID cache. Defaults to $VLLM_CACHE_ROOT or /tmp/vllm_cache_<pid>.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cache_root = args.cache_root or os.environ.get("VLLM_CACHE_ROOT") or f"/tmp/vllm_cache_{os.getpid()}"
    ok = verify(args.variant, cache_root, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
