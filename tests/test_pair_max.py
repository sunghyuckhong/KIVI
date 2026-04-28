"""Pair-max constraint for SmoothKV scales.

The repo's variant-generation code uses the **adjacent-pair** layout:
    s_K.view(L, nh, D // 2, 2).max(dim=-1, keepdim=True) ...
which yields a tensor where s_K[..., 2d] == s_K[..., 2d+1] for every d.

These tests pin that convention down so a future refactor can't silently
change it (e.g. accidentally switch to split-half [d, d+D/2]).
"""
import torch


def make_pair_max(s_raw: torch.Tensor) -> torch.Tensor:
    """Reference implementation: max within each (2k, 2k+1) pair."""
    L, nh, D = s_raw.shape
    assert D % 2 == 0
    paired = (
        s_raw.view(L, nh, D // 2, 2)
             .max(dim=-1, keepdim=True)
             .values
             .expand(L, nh, D // 2, 2)
             .reshape(L, nh, D)
             .clone()
    )
    return paired


def test_pair_max_adjacent_pairs_equal():
    """After pair-max: s[..., 2d] == s[..., 2d+1] for every pair index d."""
    torch.manual_seed(0)
    s = torch.rand(8, 4, 128) + 0.1
    s_pm = make_pair_max(s)
    even = s_pm[..., 0::2]   # indices 0, 2, 4, ...
    odd  = s_pm[..., 1::2]   # indices 1, 3, 5, ...
    assert torch.equal(even, odd)


def test_pair_max_picks_max_within_pair():
    """Each pair value equals max(orig[2d], orig[2d+1])."""
    torch.manual_seed(1)
    s = torch.rand(2, 2, 16) + 0.5
    s_pm = make_pair_max(s)
    expected = torch.maximum(s[..., 0::2], s[..., 1::2])
    assert torch.allclose(s_pm[..., 0::2], expected)
    assert (s_pm >= s).all()


def test_pair_max_idempotent():
    torch.manual_seed(2)
    s = torch.rand(4, 8, 64)
    once = make_pair_max(s)
    twice = make_pair_max(once)
    assert torch.equal(once, twice)


def test_calib_variant_files_satisfy_pair_max():
    """Spot-check committed pair-max calib files actually satisfy the invariant.

    Skips silently if no calib files are present.
    """
    import glob, os
    import pytest
    files = sorted(glob.glob("logs/calib/smoothkv_*_pair.pt"))[:3]
    if not files:
        pytest.skip("no calib files in logs/calib/")
    for f in files:
        d = torch.load(f, weights_only=True, map_location="cpu")
        if "s_K" not in d:
            continue
        s_K = d["s_K"]
        L, nh, D = s_K.shape
        if D % 2 != 0:
            continue
        even = s_K[..., 0::2]
        odd  = s_K[..., 1::2]
        assert torch.equal(even, odd), \
            f"{os.path.basename(f)} violates pair-max invariant"
