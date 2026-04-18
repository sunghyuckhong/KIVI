"""
Per-token KV cache quantization for Llama — mirror of mistral_kivi_pertoken.py.

Both keys and values quantized per-token (groups along head_dim). Supports
residual_length=0 (no FP16 buffer) for the naive 4bit comparison.
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim, dequantize_cache_pertoken
from quant.matmul import cuda_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models_paper.llama_kivi import (
    LlamaAttention_KIVI,
    LlamaDecoderLayer_KIVI,
    LlamaModel_KIVI,
    LlamaForCausalLM_KIVI,
)
from models_paper.mistral_kivi import repeat_kv_quant  # (no-op when num_kv_groups=1)


class LlamaAttention_KIVI_PerToken(LlamaAttention_KIVI):
    """Key + value cache quantized per-token (groups along head_dim).

    Inherits projections, rotary, and config bookkeeping from the paper-env
    LlamaAttention_KIVI. Only the forward() body is overridden.
    """

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
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_quant_packed = past_key_value[0]
            key_states_full  = past_key_value[1]
            key_scale        = past_key_value[2]
            key_mn           = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full  = past_key_value[5]
            value_scale        = past_key_value[6]
            value_mn           = past_key_value[7]

            if key_quant_packed is not None:
                key_quant_rep = repeat_kv_quant(key_quant_packed, self.num_key_value_groups)
                key_scale_rep = repeat_kv_quant(key_scale, self.num_key_value_groups)
                key_mn_rep    = repeat_kv_quant(key_mn, self.num_key_value_groups)
                key_deq = dequantize_cache_pertoken(
                    key_quant_rep, key_scale_rep, key_mn_rep, self.group_size, self.k_bits
                )
                att_qkquant = torch.matmul(query_states, key_deq.transpose(2, 3))
            else:
                att_qkquant = None

            if self.residual_length == 0:
                key_states_cur_rep = repeat_kv(key_states, self.num_key_value_groups)
                att_qkcur = torch.matmul(query_states, key_states_cur_rep.transpose(2, 3))
                if att_qkquant is not None:
                    attn_weights = torch.cat([att_qkquant, att_qkcur], dim=-1) / math.sqrt(self.head_dim)
                else:
                    attn_weights = att_qkcur / math.sqrt(self.head_dim)

                key_quant_new, key_scale_new, key_mn_new = triton_quantize_and_pack_along_last_dim(
                    key_states.contiguous(), self.group_size, self.k_bits
                )
                if key_quant_packed is not None:
                    key_quant_packed = torch.cat([key_quant_packed, key_quant_new], dim=2)
                    key_scale        = torch.cat([key_scale,        key_scale_new], dim=2)
                    key_mn           = torch.cat([key_mn,           key_mn_new],    dim=2)
                else:
                    key_quant_packed = key_quant_new
                    key_scale        = key_scale_new
                    key_mn           = key_mn_new
                key_states_full = None
            else:
                if key_states_full is not None:
                    key_states_full = torch.cat([key_states_full, key_states], dim=2)
                else:
                    key_states_full = key_states

                key_states_full_rep = repeat_kv(key_states_full, self.num_key_value_groups)
                att_qkfull = torch.matmul(query_states, key_states_full_rep.transpose(2, 3))

                if att_qkquant is not None:
                    attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
                else:
                    attn_weights = att_qkfull / math.sqrt(self.head_dim)

                if key_states_full.shape[-2] == self.residual_length:
                    key_quant_new, key_scale_new, key_mn_new = triton_quantize_and_pack_along_last_dim(
                        key_states_full.contiguous(), self.group_size, self.k_bits
                    )
                    key_states_full = None
                    if key_quant_packed is not None:
                        key_quant_packed = torch.cat([key_quant_packed, key_quant_new], dim=2)
                        key_scale        = torch.cat([key_scale,        key_scale_new], dim=2)
                        key_mn           = torch.cat([key_mn,           key_mn_new],    dim=2)
                    else:
                        key_quant_packed = key_quant_new
                        key_scale        = key_scale_new
                        key_mn           = key_mn_new

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            if value_states_full is not None:
                value_states_full = torch.cat([value_states_full, value_states], dim=2)
            else:
                value_states_full = value_states
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_rep = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_rep)
            else:
                value_quant_rep = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
                value_scale_rep = repeat_kv_quant(value_scale, self.num_key_value_groups)
                value_mn_rep    = repeat_kv_quant(value_mn, self.num_key_value_groups)
                attn_output = cuda_bmm_fA_qB_outer(
                    self.group_size,
                    attn_weights[:, :, :, :-value_full_length],
                    value_quant_rep, value_scale_rep, value_mn_rep,
                    self.v_bits
                )
                value_states_full_rep = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(
                    attn_weights[:, :, :, -value_full_length:], value_states_full_rep
                )

            if value_states_full.shape[-2] > self.residual_length:
                assert value_states_full.shape[-2] == self.residual_length + 1
                value_quant_new, v_scale_new, v_mn_new = triton_quantize_and_pack_along_last_dim(
                    value_states_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits
                )
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_quant_new], dim=2)
                    value_scale        = torch.cat([value_scale,        v_scale_new],     dim=2)
                    value_mn           = torch.cat([value_mn,           v_mn_new],        dim=2)
                else:
                    value_states_quant = value_quant_new
                    value_scale        = v_scale_new
                    value_mn           = v_mn_new

        else:
            if self.residual_length == 0:
                key_to_quant    = key_states
                key_states_full = None
            elif key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_to_quant   = None
                    key_states_full = key_states
                else:
                    key_to_quant   = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_to_quant   = key_states
                key_states_full = None

            if key_to_quant is not None:
                key_quant_packed, key_scale, key_mn = triton_quantize_and_pack_along_last_dim(
                    key_to_quant.contiguous(), self.group_size, self.k_bits
                )
            else:
                key_quant_packed = key_scale = key_mn = None

            if self.residual_length == 0:
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_states.contiguous(), self.group_size, self.v_bits
                )
                value_states_full = None
            elif value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full  = value_states
                value_scale = value_mn = None
            else:
                value_to_quant     = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full  = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_to_quant, self.group_size, self.v_bits
                )

            key_rep   = repeat_kv(key_states,   self.num_key_value_groups)
            value_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output  = torch.matmul(attn_weights, value_rep)

        past_key_value = (
            key_quant_packed, key_states_full,
            key_scale, key_mn,
            value_states_quant, value_states_full,
            value_scale, value_mn,
            kv_seq_len,
        ) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


class LlamaDecoderLayer_KIVI_PerToken(LlamaDecoderLayer_KIVI):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention_KIVI_PerToken(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaModel_KIVI_PerToken(LlamaModel_KIVI):
    def __init__(self, config):
        # Paper-env LlamaAttention_KIVI doesn't assert use_flash, so no toggle needed.
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_KIVI_PerToken(config) for _ in range(config.num_hidden_layers)]
        )


class LlamaForCausalLM_KIVI_PerToken(LlamaForCausalLM_KIVI):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel_KIVI_PerToken(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
