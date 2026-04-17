"""
FP8 (e4m3fn) per-token KV cache quantization for Mistral — baseline.

Both key and value caches are quantized per-token with group_size along head_dim
using float8_e4m3fn. No residual buffer — every token is quantized immediately.

This serves as a baseline to compare against KIVI's asymmetric INT2 scheme.
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from quant.fp8_quant import quantize_fp8, dequantize_fp8

from models.mistral_kivi import (
    MistralAttention_KIVI,
    MistralDecoderLayer_KIVI,
    MistralModel_KIVI,
    MistralForCausalLM_KIVI,
    repeat_kv,
    apply_rotary_pos_emb,
    _prepare_4d_causal_attention_mask,
    MistralPreTrainedModel,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)


class MistralAttention_FP8(MistralAttention_KIVI):
    """
    Pure per-token FP8 quantization for both K and V.
    No residual buffer — all tokens quantized immediately.

    KV cache tuple: (key_fp8, key_scale, val_fp8, val_scale, kv_seq_len)
    """

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

        # ── Decode path ──────────────────────────────────────────────────────
        if past_key_value is not None:
            key_fp8   = past_key_value[0]
            key_scale = past_key_value[1]
            val_fp8   = past_key_value[2]
            val_scale = past_key_value[3]

            # Quantize new token
            new_key_fp8, new_key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            new_val_fp8, new_val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

            # Append to cache
            key_fp8   = torch.cat([key_fp8, new_key_fp8], dim=2)
            key_scale = torch.cat([key_scale, new_key_scale], dim=2)
            val_fp8   = torch.cat([val_fp8, new_val_fp8], dim=2)
            val_scale = torch.cat([val_scale, new_val_scale], dim=2)

            # Dequantize for attention
            key_deq = dequantize_fp8(key_fp8, key_scale, self.group_size)
            val_deq = dequantize_fp8(val_fp8, val_scale, self.group_size)

            key_deq = repeat_kv(key_deq, self.num_key_value_groups)
            val_deq = repeat_kv(val_deq, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_deq.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_deq)

        # ── Prefill path ─────────────────────────────────────────────────────
        else:
            # Full-precision attention during prefill
            key_rep   = repeat_kv(key_states, self.num_key_value_groups)
            value_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_rep)

            # Quantize all KV to FP8 for storage
            key_fp8, key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            val_fp8, val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

        # ── Build past_key_value tuple ───────────────────────────────────────
        past_key_value = (
            key_fp8, key_scale,
            val_fp8, val_scale,
            kv_seq_len,
        ) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class MistralDecoderLayer_FP8(MistralDecoderLayer_KIVI):
    def __init__(self, config):
        super().__init__(config)
        self.self_attn = MistralAttention_FP8(config=config)


class MistralModel_FP8(MistralModel_KIVI):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer_FP8(config) for _ in range(config.num_hidden_layers)]
        )


class MistralForCausalLM_FP8(MistralForCausalLM_KIVI):
    def __init__(self, config):
        super(MistralPreTrainedModel, self).__init__(config)
        self.model = MistralModel_FP8(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
