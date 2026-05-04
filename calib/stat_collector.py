"""Running statistics collector for SmoothKV calibration.

Captures per-(layer, head, channel) running ``max|x|`` for Q, K, V activations
and (optionally) a uniform-random reservoir sample of abs values for offline
percentile-based scale computation.

Designed for streaming: one transformer forward pass updates the stats; a
calibration run is just N forward passes followed by ``compute_scales``.
"""
import torch


class StatCollector:
    """Running max|x| per (layer, head, channel), plus optional reservoir sample.

    Buffers:
        max_q, max_k, max_v   running max|x| on each input dim, on the chosen
                              device (typically cuda:0). For the GQA case,
                              max_q has num_q_heads while max_k/max_v have
                              num_kv_heads.
        samples_k, samples_v  reservoir samples of abs values on CPU (bf16),
                              shape (L, nh, D, R). Only allocated when
                              ``samples_per_channel > 0``.
        count_k, count_v      per-(layer, head, channel) sample counts seen so
                              far (for reservoir replacement probability).
    """

    def __init__(self, num_layers, num_kv_heads, num_q_heads, head_dim, device,
                 samples_per_channel: int = 0):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.device = device
        self.R = samples_per_channel

        self.max_q = torch.zeros(num_layers, num_q_heads, head_dim, device=device)
        self.max_k = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)
        self.max_v = torch.zeros(num_layers, num_kv_heads, head_dim, device=device)

        if self.R > 0:
            # CPU bf16 buffers — match Qwen3's native dtype (no lossy fp16 downcast).
            self.samples_k = torch.zeros(num_layers, num_kv_heads, head_dim, self.R,
                                         dtype=torch.bfloat16)
            self.samples_v = torch.zeros(num_layers, num_kv_heads, head_dim, self.R,
                                         dtype=torch.bfloat16)
            self.count_k = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.long)
            self.count_v = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.long)
        else:
            self.samples_k = self.samples_v = None
            self.count_k = self.count_v = None

    def update_q(self, layer_idx, q):
        # Move incoming tensor to the collector's device — necessary when the
        # model is sharded across GPUs via device_map='auto' (layers may live
        # on cuda:1 while the collector buffers are on cuda:0).
        m = q.abs().amax(dim=(0, 2)).to(torch.float32).to(self.max_q.device)
        torch.maximum(self.max_q[layer_idx], m, out=self.max_q[layer_idx])

    def update_k(self, layer_idx, k):
        self._update(layer_idx, k, self.max_k, self.samples_k, self.count_k)

    def update_v(self, layer_idx, v):
        self._update(layer_idx, v, self.max_v, self.samples_v, self.count_v)

    def _update(self, layer_idx, x, max_buf, sample_buf, count_buf):
        """Update running max and (optionally) reservoir sample for x: (B, nh, T, D)."""
        a = x.abs()
        m = a.amax(dim=(0, 2)).to(torch.float32).to(max_buf.device)  # (nh, D)
        torch.maximum(max_buf[layer_idx], m, out=max_buf[layer_idx])

        if sample_buf is None:
            return
        R = self.R
        B, nh, T, D = a.shape
        N = B * T

        # Reshape to (nh, D, N) then move to CPU once per batch. Match sample_buf
        # dtype (bf16) so we don't downcast bf16 activations through fp16.
        flat = a.permute(1, 3, 0, 2).reshape(nh, D, N).to(sample_buf.dtype).cpu()
        # Per channel, fill empty slots first then do reservoir replacement.
        buf = sample_buf[layer_idx]   # (nh, D, R)
        for h in range(nh):
            for c in range(D):
                sb = int(count_buf[layer_idx, h, c].item())  # seen before this batch
                vals = flat[h, c]      # (N,)
                if sb < R:
                    n_fill = min(R - sb, N)
                    buf[h, c, sb:sb + n_fill] = vals[:n_fill]
                    rest = vals[n_fill:]
                    base = sb + n_fill
                else:
                    rest = vals
                    base = sb
                if rest.numel() > 0:
                    # reservoir replacement: for the i-th remaining val, probability
                    # of replacing some slot in the buffer is R / (base + i + 1).
                    t_range = torch.arange(base + 1, base + rest.numel() + 1,
                                           dtype=torch.float32)
                    accept = torch.rand(rest.numel()) < R / t_range
                    accepted_idx = accept.nonzero(as_tuple=True)[0]
                    if accepted_idx.numel() > 0:
                        slots = torch.randint(0, R, (accepted_idx.numel(),))
                        buf[h, c, slots] = rest[accepted_idx]
        count_buf[layer_idx] += N
