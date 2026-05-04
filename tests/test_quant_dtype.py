"""Fake-quant dtype propagation + MSE bounds.

Would have caught the bf16 → fp16 silent demotion bug in fake_quant_utils.py.
Also locks in MSE thresholds so future kernel rewrites can't regress accuracy
without flagging it.
"""
import pytest
import torch

from tests.conftest import requires_cuda


@pytest.fixture
def k_bf16():
    torch.manual_seed(0)
    return torch.randn(8, 4 * 128, dtype=torch.bfloat16, device="cuda")


@pytest.fixture
def v_bf16():
    torch.manual_seed(1)
    return torch.randn(8, 4 * 128, dtype=torch.bfloat16, device="cuda")


# ── dtype propagation ──

@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fake_quantize_v_pertoken_preserves_dtype(dtype):
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_v_pertoken
    x = torch.randn(8, 4 * 128, dtype=dtype, device="cuda")
    out = fake_quantize_v_pertoken(x, num_kv_heads=4, head_dim=128, group_size=128, bits=4)
    assert out.dtype == dtype, f"input {dtype} → output {out.dtype}"
    assert out.shape == x.shape


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fake_quantize_k_pertoken_preserves_dtype(dtype):
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_k_pertoken
    x = torch.randn(8, 4 * 128, dtype=dtype, device="cuda")
    out = fake_quantize_k_pertoken(x, num_kv_heads=4, head_dim=128, group_size=128, bits=4)
    assert out.dtype == dtype


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fake_quantize_fp8_preserves_dtype(dtype):
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_fp8
    x = torch.randn(8, 4 * 128, dtype=dtype, device="cuda")
    out = fake_quantize_fp8(x, num_kv_heads=4, head_dim=128, group_size=128)
    assert out.dtype == dtype


# ── MSE bounds (catch accidental degradation in quant kernels) ──

@requires_cuda
def test_int4_pertoken_mse_under_threshold(v_bf16):
    """Round-trip MSE for INT4 g=128 should stay below historical baseline."""
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_v_pertoken
    out = fake_quantize_v_pertoken(v_bf16, num_kv_heads=4, head_dim=128, group_size=128, bits=4)
    mse = (v_bf16.float() - out.float()).pow(2).mean().item()
    # Empirical baseline: INT4 g=128 round-trip on N(0,1) is ~0.005-0.015.
    assert mse < 0.05, f"INT4 g=128 MSE={mse:.4f} exceeded 0.05 threshold"


@requires_cuda
def test_fp8_mse_well_below_int4(v_bf16):
    """FP8 should be much more accurate than INT4 (sanity check)."""
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_v_pertoken, fake_quantize_fp8
    out_fp8 = fake_quantize_fp8(v_bf16, 4, 128, group_size=128)
    out_int4 = fake_quantize_v_pertoken(v_bf16, 4, 128, 128, bits=4)
    mse_fp8 = (v_bf16.float() - out_fp8.float()).pow(2).mean().item()
    mse_int4 = (v_bf16.float() - out_int4.float()).pow(2).mean().item()
    assert mse_fp8 < mse_int4, "FP8 MSE should be lower than INT4"
    assert mse_fp8 < 0.005, f"FP8 MSE={mse_fp8:.5f} too high — possible bf16/fp16 cast regression"


# ── shape preservation ──

@requires_cuda
@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_fake_quantize_preserves_shape(layout):
    from vllm.model_executor.layers.quantization.kv_fake_quant import fake_quantize_v_pertoken
    if layout == "2d":
        x = torch.randn(16, 4 * 128, dtype=torch.bfloat16, device="cuda")
    else:
        x = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
    out = fake_quantize_v_pertoken(x, 4, 128, 128, bits=4)
    assert out.shape == x.shape
