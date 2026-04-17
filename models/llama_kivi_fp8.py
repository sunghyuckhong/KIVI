"""
FP8 (e4m3fn) per-token KV cache quantization for Llama — baseline.

Mirror of models/mistral_kivi_fp8.py for the Llama architecture. Both key and
value caches are quantized per-token with group_size along head_dim using
float8_e4m3fn. No residual buffer — every token is quantized immediately.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from quant.fp8_quant import quantize_fp8, dequantize_fp8

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from models.llama_kivi import (
    LlamaDecoderLayer_KIVI,
    LlamaModel_KIVI,
    LlamaForCausalLM_KIVI,
)


class LlamaAttention_FP8(nn.Module):
    """Llama attention with per-token FP8 KV cache. No residual buffer."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.group_size = config.group_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_fp8   = past_key_value[0]
            key_scale = past_key_value[1]
            val_fp8   = past_key_value[2]
            val_scale = past_key_value[3]

            new_key_fp8, new_key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            new_val_fp8, new_val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

            key_fp8   = torch.cat([key_fp8, new_key_fp8], dim=2)
            key_scale = torch.cat([key_scale, new_key_scale], dim=2)
            val_fp8   = torch.cat([val_fp8, new_val_fp8], dim=2)
            val_scale = torch.cat([val_scale, new_val_scale], dim=2)

            key_deq = dequantize_fp8(key_fp8, key_scale, self.group_size)
            val_deq = dequantize_fp8(val_fp8, val_scale, self.group_size)

            key_deq = repeat_kv(key_deq, self.num_key_value_groups)
            val_deq = repeat_kv(val_deq, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_deq.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_deq)

        else:
            key_rep   = repeat_kv(key_states, self.num_key_value_groups)
            value_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_rep)

            key_fp8, key_scale = quantize_fp8(key_states.contiguous(), self.group_size)
            val_fp8, val_scale = quantize_fp8(value_states.contiguous(), self.group_size)

        past_key_value = (key_fp8, key_scale, val_fp8, val_scale, kv_seq_len) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


class LlamaDecoderLayer_FP8(LlamaDecoderLayer_KIVI):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention_FP8(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaModel_FP8(LlamaModel_KIVI):
    def __init__(self, config):
        # Avoid the parent's LlamaDecoderLayer_KIVI construction (its attention
        # asserts use_flash=True). Temporarily toggle use_flash so the assert
        # passes during parent init, then overwrite layers with FP8 variants.
        prev_use_flash = getattr(config, "use_flash", False)
        config.use_flash = True
        super().__init__(config)
        config.use_flash = prev_use_flash
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_FP8(config) for _ in range(config.num_hidden_layers)]
        )


class LlamaForCausalLM_FP8(LlamaForCausalLM_KIVI):
    def __init__(self, config):
        super(LlamaPreTrainedModel, self).__init__(config)
        self.model = LlamaModel_FP8(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
