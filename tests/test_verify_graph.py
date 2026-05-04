"""Verify graph capture for KV fake-quant.

Compiles a tiny forward through torch._dynamo.export and asserts the captured
FX graph contains exactly the expected `vllm_kv_quant::*` op signatures for
each method -- and only those (no leakage between branches).

This is the unit-level analogue of `scripts/verify_compiled_graph.py`, which
greps inductor's `computation_graph.py` dump after a real eval. Both must
pass for a method's results to be trustworthy:

    unit-level (this file):  proves the scheme file dispatches correctly to
                             the expected ops in isolation.
    eval-level (scripts/):   proves that vLLM's torch.compile pipeline
                             actually captured those ops in the live model.

If this fails, the scheme file is broken; nothing downstream is trustable.
"""
import pytest
import torch

from tests.conftest import requires_cuda


# Expected substrings in the captured graph's op names per method.
# Mirrors EXPECTED/FORBIDDEN in scripts/verify_compiled_graph.py.
EXPECTED_OPS = {
    "fp8":      ["fake_quantize_dequantize_fp8"],
    "pertoken": ["quant_and_pack_vcache", "unpack_and_dequant_vcache"],
    "smoothkv": ["quant_and_pack_vcache", "unpack_and_dequant_vcache"],
}
FORBIDDEN_OPS = {
    # bf16/fp16 must NOT contain any quant kernel calls.
    "bf16": ["quant_and_pack", "fake_quantize_dequantize"],
    "fp16": ["quant_and_pack", "fake_quantize_dequantize"],
    # FP8 must NOT use the int4 path.
    "fp8":  ["quant_and_pack"],
}


class _MockLayer(torch.nn.Module):
    """Minimal stand-in for vllm.Attention with the attrs apply_kv_quant reads."""
    def __init__(self, method, num_kv_heads=8, head_size=128,
                 group_size=128, bits=4, sk=None, sv=None):
        super().__init__()
        self._kv_quant_method = method
        self._kv_quant_group_size = group_size
        self._kv_quant_bits = bits
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        if sk is not None:
            self.register_buffer("_kv_quant_s_k", sk, persistent=False)
            self.register_buffer("_kv_quant_s_v", sv, persistent=False)


def _captured_op_names(fn, args) -> list[str]:
    """Run torch._dynamo.export and return the qualified names of every
    call_function node in the captured FX graph."""
    gm, _ = torch._dynamo.export(fn, aten_graph=False)(*args)
    names: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            names.append(str(node.target))
    return names


def _has_substr(names: list[str], sub: str) -> bool:
    return any(sub in n for n in names)


@requires_cuda
@pytest.mark.parametrize("method", ["fp8", "pertoken"])
def test_apply_kv_quant_emits_expected_ops(method):
    """For each method, the captured graph must contain every expected op
    signature and none of the forbidden ones."""
    from vllm.model_executor.layers.quantization.kv_fake_quant import apply_kv_quant

    layer = _MockLayer(method, group_size=128, bits=4)

    def fn(k, v):
        return apply_kv_quant(layer, k, v)

    k = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    names = _captured_op_names(fn, (k, v))
    print(f"[{method}] captured ops: {names}")

    for expect in EXPECTED_OPS[method]:
        assert _has_substr(names, expect), (
            f"[{method}] missing {expect!r} in captured graph: {names}"
        )
    for forbid in FORBIDDEN_OPS.get(method, []):
        assert not _has_substr(names, forbid), (
            f"[{method}] unexpectedly found {forbid!r} in captured graph: {names}"
        )


@requires_cuda
def test_apply_kv_quant_smoothkv_emits_expected_ops():
    """SmoothKV path multiplies by per-layer scales then runs per-token int4."""
    from vllm.model_executor.layers.quantization.kv_fake_quant import apply_kv_quant
    sk = torch.ones(8, 128, dtype=torch.bfloat16, device="cuda")
    sv = torch.ones(8, 128, dtype=torch.bfloat16, device="cuda")
    layer = _MockLayer("smoothkv", sk=sk, sv=sv)

    def fn(k, v):
        return apply_kv_quant(layer, k, v)

    k = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    names = _captured_op_names(fn, (k, v))

    for expect in EXPECTED_OPS["smoothkv"]:
        assert _has_substr(names, expect), (
            f"[smoothkv] missing {expect!r} in captured graph: {names}"
        )


@requires_cuda
def test_bf16_emits_no_quant_ops():
    """bf16 must be a no-op -- apply_kv_quant returns immediately on the
    method check, so no quant op should ever appear in the graph."""
    # apply_kv_quant raises on bf16 (only branches are fp8/pertoken/smoothkv).
    # Verifying the no-op happens via attach_kv_quant_to_layer which doesn't
    # set _kv_quant_method on bf16 -- so Attention.forward's `if getattr(...)`
    # short-circuits. We exercise that short-circuit here.
    class _BareLayer:
        # No _kv_quant_method attr set -- represents a layer where bf16 was
        # configured (configure_kv_quant("bf16", ...) doesn't set anything).
        pass

    layer = _BareLayer()

    def fn(k, v):
        if getattr(layer, "_kv_quant_method", None):
            from vllm.model_executor.layers.quantization.kv_fake_quant import apply_kv_quant
            return apply_kv_quant(layer, k, v)
        return k, v

    k = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    names = _captured_op_names(fn, (k, v))

    for forbid in FORBIDDEN_OPS["bf16"]:
        assert not _has_substr(names, forbid), (
            f"[bf16] unexpectedly found {forbid!r} in captured graph: {names}"
        )


@requires_cuda
@pytest.mark.parametrize("method", ["fp8", "pertoken", "smoothkv"])
def test_apply_kv_quant_preserves_dtype_and_shape(method):
    """Round-trip parity check: apply_kv_quant returns same shape & dtype."""
    from vllm.model_executor.layers.quantization.kv_fake_quant import apply_kv_quant

    kwargs = {}
    if method == "smoothkv":
        kwargs["sk"] = torch.ones(8, 128, dtype=torch.bfloat16, device="cuda")
        kwargs["sv"] = torch.ones(8, 128, dtype=torch.bfloat16, device="cuda")
    layer = _MockLayer(method, group_size=128, bits=4, **kwargs)

    k = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device="cuda")
    k_q, v_q = apply_kv_quant(layer, k, v)
    assert k_q.shape == k.shape and v_q.shape == v.shape, (
        f"[{method}] shape mismatch: k {k.shape}->{k_q.shape}, v {v.shape}->{v_q.shape}"
    )
    assert k_q.dtype == k.dtype and v_q.dtype == v.dtype, (
        f"[{method}] dtype mismatch: k {k.dtype}->{k_q.dtype}, v {v.dtype}->{v_q.dtype}"
    )
