#!/usr/bin/env python3
"""Verify that vLLM's torch.compile captured our patched forward.

After install_method() patches Qwen3Attention.forward, vLLM traces through
the patched code and inductor emits a `computation_graph.py` with source-line
annotations pointing back to the original Python files. We grep that file
for ACTUAL op invocations (`torch.ops.vllm_kv_quant.<op>(`) — not just the
op name as a string, because inductor's comment annotations like
`# File: /KIVI/quant/new_pack.py:38 in quant_and_pack_vcache, code: ...`
also contain the op name and would false-positive a substring match.

Per-variant signatures we look for:
    bf16/fp16     → NO actual quant op invocations (baseline)
    fp8           → vllm_kv_quant.fake_quantize_dequantize_fp8(
    pertoken      → vllm_kv_quant.{quant_and_pack,unpack_and_dequant}_vcache(
    smoothkv      → same as pertoken (the runtime kernel calls
                    fake_quantize_pertoken which dispatches to those ops)
    smoothkv_fused → same as pertoken (post-fusion the layer is plain pertoken)

Returns 0 if all expected signatures present and no forbidden ones; 1 otherwise.

Usage:
    python scripts/verify_compiled_graph.py --variant pertoken
    python scripts/verify_compiled_graph.py --variant smoothkv_fused --cache_root /tmp/vllm_cache_1746040
"""
import argparse
import glob
import os
import re
import sys


# Each value is a regex that must match at least one ACTUAL op invocation
# (i.e., the op name immediately followed by `(`). This excludes inductor's
# `# in <op>,` comment annotations.
EXPECTED = {
    "bf16":           [],
    "fp16":           [],
    "fp8":            [r"vllm_kv_quant\.fake_quantize_dequantize_fp8\("],
    "pertoken":       [r"vllm_kv_quant\.quant_and_pack_vcache\(",
                       r"vllm_kv_quant\.unpack_and_dequant_vcache\("],
    "smoothkv":       [r"vllm_kv_quant\.quant_and_pack_vcache\(",
                       r"vllm_kv_quant\.unpack_and_dequant_vcache\("],
    "smoothkv_fused": [r"vllm_kv_quant\.quant_and_pack_vcache\(",
                       r"vllm_kv_quant\.unpack_and_dequant_vcache\("],
    "nvfp4":          [r"vllm_kv_quant\.fake_quantize_dequantize_nvfp4\("],
    "smkv_nvfp4":     [r"vllm_kv_quant\.fake_quantize_dequantize_nvfp4\("],
}

# For baselines we must NOT see any actual quant op invocations. Catches
# the case where a stale compile cache or plugin override silently quantizes
# a "bf16" run.
FORBIDDEN = {
    "bf16": [r"vllm_kv_quant\.\w+\("],
    "fp16": [r"vllm_kv_quant\.\w+\("],
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


def _count(pattern: str, content: str) -> int:
    return len(re.findall(pattern, content))


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
        # actual call counts per expected op
        op_counts = {pat: _count(pat, content) for pat in expected}
        forbidden_counts = {pat: _count(pat, content) for pat in forbidden}
        missing = [pat for pat, n in op_counts.items() if n == 0]
        present_forbidden = [pat for pat, n in forbidden_counts.items() if n > 0]
        ok = not missing and not present_forbidden
        if verbose:
            tag = "[PASS]" if ok else "[FAIL]"
            # Strip the regex escapes (`\.`, `\(`) for the human-readable summary.
            def _short(pat):
                base = pat.split(".")[-1]
                return base.replace("\\(", "").replace("\\.", ".")
            ops_str = " ".join(f"{_short(p)}={n}" for p, n in op_counts.items())
            print(f"[verify-graph] {tag} variant={variant} graph={gf}"
                  + (f" ops=({ops_str})" if expected else ""))
            if missing:
                print(f"  MISSING (no ACTUAL op calls in graph): {missing}")
            if present_forbidden:
                print(f"  FORBIDDEN PRESENT (silent quant on baseline!): "
                      f"{[p for p in present_forbidden]} "
                      f"(counts: { {p: forbidden_counts[p] for p in present_forbidden} })")
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
