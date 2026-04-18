"""
Paper-env port of models/llama_kivi_fp8.py.

Uses transformers 4.36.2 API (rotary_emb takes seq_len=int, not position_ids).
Inherits from models_paper.llama_kivi so the lm-eval HFLM wrapper and
LlamaForCausalLM_KIVI.forward paths stay consistent with the paper code path.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from quant.fp8_quant import quantize_fp8, dequantize_fp8

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaMLP,
    apply_rotary_pos_emb,
)

from models_paper.llama_kivi import (
    LlamaAttention_KIVI,
    LlamaDecoderLayer_KIVI,
    LlamaModel_KIVI,
    LlamaForCausalLM_KIVI,
)


class LlamaAttention_FP8(LlamaAttention_KIVI):
    """Per-token FP8 KV cache. No residual buffer — all tokens quantized immediately."""

    def __init__(self, config):
        super().__init__(config)
        # Override KIVI's 2/4-bit setting: we use fp8, group_size is the
        # head_dim partition (usually 128 = whole head).
        self.group_size = getattr(config, "group_size", 128)

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

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_fp8, key_scale = past_key_value[0], past_key_value[1]
            val_fp8, val_scale = past_key_value[2], past_key_value[3]

            new_key_fp8, new_key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            new_val_fp8, new_val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

            key_fp8   = torch.cat([key_fp8, new_key_fp8], dim=2)
            key_scale = torch.cat([key_scale, new_key_scale], dim=2)
            val_fp8   = torch.cat([val_fp8, new_val_fp8], dim=2)
            val_scale = torch.cat([val_scale, new_val_scale], dim=2)

            key_deq = dequantize_fp8(key_fp8, key_scale, self.group_size)
            val_deq = dequantize_fp8(val_fp8, val_scale, self.group_size)

            attn_weights = torch.matmul(query_states, key_deq.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_deq)

        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            key_fp8, key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            val_fp8, val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

        past_key_value = (key_fp8, key_scale, val_fp8, val_scale, kv_seq_len) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


class LlamaDecoderLayer_FP8(LlamaDecoderLayer_KIVI):
    def __init__(self, config):
        # Skip the parent KIVI layer's __init__ (which instantiates the KIVI
        # attention + asserts use_flash). Build from scratch here.
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention_FP8(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaModel_FP8(LlamaModel_KIVI):
    def __init__(self, config):
        super().__init__(config)
        # Overwrite the KIVI layers with FP8 layers (same signature).
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_FP8(config) for _ in range(config.num_hidden_layers)]
        )


class LlamaForCausalLM_FP8(LlamaForCausalLM_KIVI):
    def __init__(self, config):
        # Skip LlamaForCausalLM_KIVI.__init__ (builds the KIVI model stack);
        # go straight to the LlamaPreTrainedModel base and install our stack.
        from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel_FP8(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
