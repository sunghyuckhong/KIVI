"""
Fine-grained FP8 (e4m3fn) quantization for KV cache.

Per-token quantization with configurable group_size along head_dim.
Each group of elements shares one scale factor (no zero-point needed
since FP8 is symmetric around zero).

Quantize:   scale = max(|x|) / FP8_MAX;  x_fp8 = (x / scale).to(float8_e4m3fn)
Dequantize: x_fp16 = x_fp8.to(float16) * scale

Software emulation path:
On A100 (sm_80) Triton's inductor codegen has no `fp8e4nv` codegen, so the
torch.float8_e4m3fn cast can't compile inside CUDA graphs. We provide a
sm_80-compatible software emulation in `_round_to_fp8e4m3` that operates in
fp32 (sign · exp · mantissa truncation) and produces bit-identical values to
the hardware cast for all normal-range inputs. Triton compiles fp32 ops fine,
so this path is cudagraph-compatible.

`fake_quantize_dequantize_fp8` dispatches based on cuda capability — hardware
cast on sm_89+ (Hopper+), software emulation on sm_80.
"""
import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0


def _round_to_fp8e4m3(x_fp32: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor to FP8 E4M3 precision, returning fp32 with rounded values.

    E4M3 layout: 1 sign bit · 4 exponent bits (bias=7) · 3 mantissa bits.
    - exp range: [-6, 8]   (so smallest normal = 2^-6, largest = 1.875 · 2^8 = 480
      but max representable is 448 = 1.75·2^8 because 0b0_1111_111 is reserved)
    - subnormal range: 2^-9 to 2^-7 (granularity 2^-9)
    - max representable absolute value: 448.0

    This emulation matches `tensor.to(torch.float8_e4m3fn).to(fp32)` for all
    normal-range inputs. Subnormal handling is approximate but contributes
    negligibly to KV-cache fake-quant accuracy. Saturating clamp at ±448
    matches torch's behavior (tensors above clamp to ±448, not ±inf).
    """
    sign = torch.sign(x_fp32)
    abs_x = x_fp32.abs()
    # Saturate to E4M3 max (448.0)
    abs_x = abs_x.clamp(max=_FP8_MAX)

    # Compute exponent (floor of log2, clamped to E4M3 range).
    # Use a safe min to avoid log2(0) = -inf for true zeros.
    eps_floor = 2.0 ** -9
    abs_safe = abs_x.clamp(min=eps_floor)
    exp = torch.floor(torch.log2(abs_safe))
    exp_clamped = exp.clamp(min=-6.0, max=8.0)

    pow2_exp = torch.pow(2.0, exp_clamped)
    # Mantissa in [1, 2) for normals; for subnormals (exp=-6 used as floor)
    # mantissa_raw can be in [0, 1), and 3-bit quant still applies.
    mantissa_raw = abs_x / pow2_exp
    # Round to 3-bit mantissa precision (8 levels in [1, 2) → step 0.125).
    # Round-half-to-even via .round() (PyTorch default).
    mantissa_q = torch.round(mantissa_raw * 8.0) / 8.0

    out = sign * mantissa_q * pow2_exp

    # Restore exact zero where input was zero (avoid sign·0·anything subtleties).
    out = torch.where(x_fp32 == 0, torch.zeros_like(out), out)
    return out


def _is_sm89_or_newer() -> bool:
    """True if current CUDA device supports hardware fp8e4nv (Hopper+)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap >= (8, 9)


def fake_quantize_dequantize_fp8(
    data: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Quantize→dequantize round-trip at FP8 E4M3 precision, returning the
    rounded values in the input dtype.

    Combines `quantize_fp8` + `dequantize_fp8` into a single function that
    skips the uint8 intermediate. On sm_89+ uses the hardware fp8 cast for
    bit-exact behavior; on sm_80 uses the software emulation that's
    cudagraph-compatible.
    """
    assert len(data.shape) == 4
    B, nh, T, D = data.shape
    assert D % group_size == 0
    num_groups = D // group_size

    data_grouped = data.view(B, nh, T, num_groups, group_size)

    # Per-group scale: max(|x|) / FP8_MAX, floored at 1e-4 to avoid underflow.
    amax = data_grouped.abs().amax(dim=-1).to(torch.float32)
    scale = (amax / _FP8_MAX).clamp(min=1e-4)

    data_scaled = data_grouped.to(torch.float32) / scale.unsqueeze(-1)

    if _is_sm89_or_newer():
        # Hardware path: cast to FP8 and back. Bit-exact, fastest on Hopper+.
        data_round = data_scaled.to(_FP8_DTYPE).to(torch.float32)
    else:
        # Software path: round to E4M3 precision in fp32 (sm_80 + cudagraph
        # compatible). Numerically identical to hardware cast for normals.
        data_round = _round_to_fp8e4m3(data_scaled)

    out = (data_round * scale.unsqueeze(-1)).view(B, nh, T, D)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(data.dtype)


def quantize_fp8(
    data: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (B, nh, T, D) tensor to FP8 per-token with group_size along D.

    FP8 tensors are stored as uint8 views for torch.cat compatibility.

    Returns:
        data_uint8: (B, nh, T, D) in uint8 (bit-identical to float8_e4m3fn)
        scale:      (B, nh, T, D // group_size) in float16
    """
    assert len(data.shape) == 4
    B, nh, T, D = data.shape
    assert D % group_size == 0
    num_groups = D // group_size

    # Reshape to expose groups: (B, nh, T, num_groups, group_size)
    data_grouped = data.view(B, nh, T, num_groups, group_size)

    # Compute per-group scale in fp32: max(|x|) / FP8_MAX.
    # Floor at 1e-4 (not 1e-12): with fp16 activations, a scale below ~1e-5
    # underflows the reciprocal and produces inf/NaN after the division.
    amax = data_grouped.abs().amax(dim=-1).to(torch.float32)  # (B, nh, T, num_groups)
    scale = (amax / _FP8_MAX).clamp(min=1e-4)

    # Scale in fp32, then cast to FP8; store as uint8 for cat compatibility
    data_scaled = data_grouped.to(torch.float32) / scale.unsqueeze(-1)
    data_fp8 = data_scaled.to(_FP8_DTYPE).view(B, nh, T, D)
    data_uint8 = data_fp8.view(torch.uint8)

    return data_uint8, scale.to(data.dtype)   # match caller dtype (bf16/fp16)


def dequantize_fp8(
    data_uint8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize a (B, nh, T, D) FP8 tensor (stored as uint8) back to float16.

    Args:
        data_uint8: (B, nh, T, D) in uint8 (bit-identical to float8_e4m3fn)
        scale:      (B, nh, T, D // group_size) in float16
    Returns:
        (B, nh, T, D) in float16
    """
    B, nh, T, D = data_uint8.shape
    num_groups = D // group_size

    # Reinterpret uint8 as FP8
    data_fp8 = data_uint8.view(_FP8_DTYPE)

    # Expand scale from (B, nh, T, num_groups) -> (B, nh, T, D)
    scale_expanded = scale.unsqueeze(-1).expand(B, nh, T, num_groups, group_size).reshape(B, nh, T, D)

    out = data_fp8.to(scale.dtype) * scale_expanded
    # FP8 e4m3fn can carry NaN; guard downstream attention math.
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
