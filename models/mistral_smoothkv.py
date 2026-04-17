"""
Mistral SmoothKV: calibrated channel smoothing + per-token INT4 KV cache.

Tier 1 default:
  - K-side: diagonal scale s_K applied post-RoPE, per-token INT4 quantization of smoothed K
  - V-side: diagonal scale s_V, per-token INT4 quantization of smoothed V
  - No mu_V shift, no rotation (R_V = I)
  - No residual buffer — every token quantized immediately (streaming-friendly)

Uses Mistral base for both Mistral and Llama (with sliding_window config patch for Llama).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from quant.smoothkv_quant import (
    quantize_int4_pertoken, dequantize_int4_pertoken,
    apply_smooth_key, apply_smooth_query,
    apply_smooth_value, unsmooth_value,
)

from models.mistral_kivi import (
    MistralAttention_KIVI,
    MistralDecoderLayer_KIVI,
    MistralModel_KIVI,
    MistralForCausalLM_KIVI,
    repeat_kv,
    apply_rotary_pos_emb,
    MistralPreTrainedModel,
)


class MistralAttention_SmoothKV(MistralAttention_KIVI):
    """
    SmoothKV attention: calibrated channel smoothing + per-token INT4 KV cache.

    Cache layout:
      (key_int4, key_scale, key_zero,  # 0,1,2
       val_int4, val_scale, val_zero,  # 3,4,5
       kv_seq_len)                      # 6

    The smoothing scales s_K, s_V are loaded from config and stored as buffers.
    """

    def __init__(self, config, s_K: torch.Tensor = None, s_V: torch.Tensor = None):
        super().__init__(config)
        self.group_size_smkv = getattr(config, "group_size", 128)

        if s_K is None:
            s_K = torch.ones(self.num_key_value_heads, self.head_dim)
        if s_V is None:
            s_V = torch.ones(self.num_key_value_heads, self.head_dim)

        self.register_buffer("s_K", s_K.to(torch.float16), persistent=False)
        self.register_buffer("s_V", s_V.to(torch.float16), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # === SmoothKV TRANSFORM ===
        # K-side: smooth K_rot by dividing by s_K, and multiply Q by s_K.
        # This preserves Q K^T exactly in fp16, and we then quantize the smoothed K.
        query_states = apply_smooth_query(query_states, self.s_K)
        key_states   = apply_smooth_key  (key_states,   self.s_K)
        # V-side: store smoothed V; multiply back at attention output time
        value_states_sm = apply_smooth_value(value_states, self.s_V)

        g = self.group_size_smkv

        # ========= Decode path =========
        if past_key_value is not None:
            key_int4_p   = past_key_value[0]
            key_scale_p  = past_key_value[1]
            key_zero_p   = past_key_value[2]
            val_int4_p   = past_key_value[3]
            val_scale_p  = past_key_value[4]
            val_zero_p   = past_key_value[5]

            # Quantize the new token's K, V (smoothed)
            new_k_q, new_k_s, new_k_z = quantize_int4_pertoken(key_states.contiguous(), g)
            new_v_q, new_v_s, new_v_z = quantize_int4_pertoken(value_states_sm.contiguous(), g)

            # Concatenate to cache
            key_int4  = torch.cat([key_int4_p,  new_k_q], dim=2)
            key_scale = torch.cat([key_scale_p, new_k_s], dim=2)
            key_zero  = torch.cat([key_zero_p,  new_k_z], dim=2)
            val_int4  = torch.cat([val_int4_p,  new_v_q], dim=2)
            val_scale = torch.cat([val_scale_p, new_v_s], dim=2)
            val_zero  = torch.cat([val_zero_p,  new_v_z], dim=2)

            # Dequantize for attention
            key_deq = dequantize_int4_pertoken(key_int4, key_scale, key_zero, g)
            val_deq = dequantize_int4_pertoken(val_int4, val_scale, val_zero, g)

            # Repeat KV heads for GQA
            key_deq = repeat_kv(key_deq, self.num_key_value_groups)
            val_deq = repeat_kv(val_deq, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_deq.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                  dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_deq)

            # Undo V-side smoothing (s_V) on attention output
            attn_output = unsmooth_value(attn_output, self.s_V, self.num_key_value_groups)

        # ========= Prefill path =========
        else:
            # Full-precision attention during prefill (Q already * s_K, K already / s_K,
            # so attention scores are identical to fp16 baseline up to numerical error)
            key_rep = repeat_kv(key_states, self.num_key_value_groups)
            val_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                  dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_rep)  # uses unsmoothed V

            # Quantize smoothed K, V for storage
            key_int4, key_scale, key_zero = quantize_int4_pertoken(key_states.contiguous(), g)
            val_int4, val_scale, val_zero = quantize_int4_pertoken(value_states_sm.contiguous(), g)

        past_key_value = (
            key_int4, key_scale, key_zero,
            val_int4, val_scale, val_zero,
            kv_seq_len,
        ) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class MistralDecoderLayer_SmoothKV(MistralDecoderLayer_KIVI):
    def __init__(self, config, s_K=None, s_V=None):
        super().__init__(config)
        self.self_attn = MistralAttention_SmoothKV(config=config, s_K=s_K, s_V=s_V)


class MistralModel_SmoothKV(MistralModel_KIVI):
    def __init__(self, config, calib=None):
        super().__init__(config)
        # Replace decoder layers with SmoothKV-equipped ones.
        # calib is a dict: {"s_K": (L, num_kv_heads, D), "s_V": ...}
        layers = []
        for i in range(config.num_hidden_layers):
            sK = calib["s_K"][i] if calib is not None else None
            sV = calib["s_V"][i] if calib is not None else None
            layers.append(MistralDecoderLayer_SmoothKV(config, s_K=sK, s_V=sV))
        self.layers = nn.ModuleList(layers)


class MistralForCausalLM_SmoothKV(MistralForCausalLM_KIVI):
    """Loader uses a class attribute `_calib` to pass scales to the model init."""
    _calib = None

    def __init__(self, config):
        super(MistralPreTrainedModel, self).__init__(config)
        self.model = MistralModel_SmoothKV(config, calib=self._calib)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained_with_calib(cls, model_path, calib_path, config, **kwargs):
        """Load pretrained weights and inject calibration scales."""
        calib = torch.load(calib_path, map_location="cpu", weights_only=False)
        cls._calib = calib
        try:
            model = cls.from_pretrained(model_path, config=config, **kwargs)
        finally:
            cls._calib = None
        return model
