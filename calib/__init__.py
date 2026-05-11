# SmoothKV calibration: collect per-channel stats post-RoPE, compute scales.
#
# Public API:
#   StatCollector       — running max|x| (+ optional reservoir samples) per
#                         (layer, head, channel) for Q, K, V
#   install_v_hook      — V capture via forward-hook on v_proj
#   monkey_patch_rope   — Q/K (post-RoPE) capture via apply_rotary_pos_emb wrap
#   compute_scales      — SmoothQuant formulas:
#                         s_K = max|K|^α / max|Q|^(1-α)
#                         s_V = max|V|^β

from .hooks import install_v_hook, install_nope_qk_hook, monkey_patch_rope
from .scales import compute_scales
from .stat_collector import StatCollector

__all__ = [
    "StatCollector",
    "install_v_hook",
    "install_nope_qk_hook",
    "monkey_patch_rope",
    "compute_scales",
]
