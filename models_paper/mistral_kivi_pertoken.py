"""
Per-token KV cache quantization for Mistral, built on top of mistral_kivi.py.

KIVI default:
  keys   -> quantized per-channel  (groups of tokens share a scale per head-dim channel)
  values -> quantized per-token    (each token gets its own scale per head-dim group)

This module changes the key cache to also use per-token quantization:
  keys   -> quantized per-token    (each token gets its own scale per head-dim group)
  values -> unchanged (already per-token)

Implementation: keys are no longer transposed before quantization.
                quantize along head_dim (D) instead of sequence (T).
                During decode, keys are dequantized to fp16 before Q*K matmul
                using standard torch.matmul (no custom CUDA kernel required).
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim, dequantize_cache_pertoken
from quant.matmul import cuda_bmm_fA_qB_outer

from models_paper.mistral_kivi import (
    MistralAttention_KIVI,
    MistralDecoderLayer_KIVI,
    MistralModel_KIVI,
    MistralForCausalLM_KIVI,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralMLP,
    repeat_kv,
    repeat_kv_quant,
    apply_rotary_pos_emb,
)


class MistralAttention_KIVI_PerToken(MistralAttention_KIVI):
    """
    Key cache quantized per-token (groups along head_dim, like the value cache).
    Stored as (B, nh, T, D//feat_per_int) — no transpose.
    Dequantized to fp16 before the Q*K^T matmul.
    Value cache unchanged (already per-token in the base class).
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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )
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
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # ── Decode path (past_key_value present) ──────────────────────────────
        if past_key_value is not None:
            # Unpack stored per-token key cache
            # Indices mirror the base class tuple layout; only the key tensors
            # differ in shape: (B, nh, T, D//feat_per_int) instead of transposed.
            key_quant_packed = past_key_value[0]   # (B, nh, T_past, D//feat_per_int)
            key_states_full  = past_key_value[1]
            key_scale        = past_key_value[2]   # (B, nh, T_past, D//group_size)
            key_mn           = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full  = past_key_value[5]
            value_scale        = past_key_value[6]
            value_mn           = past_key_value[7]

            if key_quant_packed is not None:
                # Expand KV heads for GQA
                key_quant_rep = repeat_kv_quant(key_quant_packed, self.num_key_value_groups)
                key_scale_rep = repeat_kv_quant(key_scale,        self.num_key_value_groups)
                key_mn_rep    = repeat_kv_quant(key_mn,           self.num_key_value_groups)
                # Dequantize to fp16 and compute attention logits
                key_deq = dequantize_cache_pertoken(
                    key_quant_rep, key_scale_rep, key_mn_rep, self.group_size, self.k_bits
                )  # (B, num_heads, T_past_quant, D)
                att_qkquant = torch.matmul(query_states, key_deq.transpose(2, 3))
            else:
                att_qkquant = None

            if self.residual_length == 0:
                # No residual buffer — quantize the current token immediately.
                # Compute attention using FP16 for the current token, then store quantized.
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
                # Full-precision residual keys
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

                # Quantize residual keys when the residual buffer is full
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

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                    f"but is {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                        f"but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ── Value cache (unchanged — already per-token in base class) ──
            # value_states_full is None when residual_length=0 after prefill
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
                value_scale_rep = repeat_kv_quant(value_scale,        self.num_key_value_groups)
                value_mn_rep    = repeat_kv_quant(value_mn,           self.num_key_value_groups)
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

        # ── Prefill path (no past_key_value) ──────────────────────────────────
        else:
            # Split keys into to-be-quantized portion and fp16 residual
            if self.residual_length == 0:
                # No residual buffer — quantize everything immediately
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
                # Per-token: quantize along head_dim (no transpose)
                key_quant_packed, key_scale, key_mn = triton_quantize_and_pack_along_last_dim(
                    key_to_quant.contiguous(), self.group_size, self.k_bits
                )
            else:
                key_quant_packed = key_scale = key_mn = None

            # Value cache split
            if self.residual_length == 0:
                # No residual buffer — quantize all values immediately
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_states.contiguous(), self.group_size, self.v_bits
                )
                value_states_full = None
            elif value_states.shape[-2] <= self.residual_length:
                value_states_quant = value_states_full_orig = None
                value_states_full  = value_states
                value_scale = value_mn = None
            else:
                value_to_quant     = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full  = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_to_quant, self.group_size, self.v_bits
                )

            # Standard full-precision prefill attention
            key_rep   = repeat_kv(key_states,   self.num_key_value_groups)
            value_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                    f"but is {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                        f"but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output  = torch.matmul(attn_weights, value_rep)

        # ── Build past_key_value tuple (same layout as base class) ────────────
        past_key_value = (
            key_quant_packed, key_states_full,
            key_scale, key_mn,
            value_states_quant, value_states_full,
            value_scale, value_mn,
            kv_seq_len,
        ) if use_cache else None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ── Wire up the per-token attention into a complete model stack ───────────────

class MistralDecoderLayer_KIVI_PerToken(MistralDecoderLayer_KIVI):
    def __init__(self, config):
        super().__init__(config)
        # Replace the attention module with the per-token variant
        self.self_attn = MistralAttention_KIVI_PerToken(config=config)


class MistralModel_KIVI_PerToken(MistralModel_KIVI):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer_KIVI_PerToken(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_init()


class MistralForCausalLM_KIVI_PerToken(MistralForCausalLM_KIVI):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel_KIVI_PerToken(config)
        self.post_init()
