"""SmoothKV scale computation from running max|x| stats.

Implements the SmoothQuant formula for K-side migration and a power-only
V-side formula:
    s_K[l, h, c] = max|K|[l, h, c]^alpha / max|Q_grouped|[l, h, c]^(1-alpha)
    s_V[l, h, c] = max|V|[l, h, c]^beta

For GQA, ``max_q`` has ``num_q_heads`` while K/V have ``num_kv_heads``; we
aggregate Q over each KV group via amax so s_K is per-(KV head, channel).

SmoothKV is invariant under any positive per-(layer, head) rescaling of
s_K — the same factor cancels across K/=s_K and Q*=s_K — so we leave the
overall magnitude unnormalized; downstream tools (make_alpha_variants.py)
are the single source of truth for any per-layer rescaling.
"""
import torch


def compute_scales(stats, alpha: float, beta: float, eps: float = 1e-5):
    """Compute (s_K, s_V) from a populated StatCollector.

    Args:
        stats: a StatCollector after calibration. Reads num_layers,
            num_q_heads, num_kv_heads, head_dim, max_q, max_k, max_v.
        alpha: SmoothQuant K-side migration strength in [0, 1].
        beta:  V-side scaling power.
        eps:   floor for max|x| before pow() to avoid 0**(-x).

    Returns:
        (s_K, s_V, max_q_grouped) — all on the collector's device.
        s_K, s_V: (num_layers, num_kv_heads, head_dim).
        max_q_grouped: (num_layers, num_kv_heads, head_dim) — Q max
            aggregated across each KV group, returned for diagnostics.
    """
    L = stats.num_layers
    nq = stats.num_q_heads
    nkv = stats.num_kv_heads
    D = stats.head_dim
    n_rep = nq // nkv
    max_q_grouped = stats.max_q.view(L, nkv, n_rep, D).amax(dim=2)  # (L, nkv, D)

    s_K = (stats.max_k.clamp(min=eps) ** alpha) / \
          (max_q_grouped.clamp(min=eps) ** (1 - alpha))
    s_V = stats.max_v.clamp(min=eps) ** beta
    return s_K, s_V, max_q_grouped
