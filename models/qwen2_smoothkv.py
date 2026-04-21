"""
Qwen2 SmoothKV: calibrated channel smoothing + per-token INT4 KV cache.

Mirror of models/mistral_smoothkv.py for Qwen2.
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

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2RotaryEmbedding,
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models.qwen2_kivi import (
    Qwen2DecoderLayer_KIVI,
    Qwen2Model_KIVI,
    Qwen2ForCausalLM_KIVI,
)


class Qwen2Attention_SmoothKV(nn.Module):
    """Qwen2 attention with SmoothKV: channel-smoothed per-token INT4 KV cache."""

    def __init__(self, config: Qwen2Config, s_K: torch.Tensor = None, s_V: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.group_size_smkv = getattr(config, "group_size", 128)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

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

        query_states   = apply_smooth_query(query_states, self.s_K)
        key_states     = apply_smooth_key  (key_states,   self.s_K)
        value_states_sm = apply_smooth_value(value_states, self.s_V)

        g = self.group_size_smkv

        if past_key_value is not None:
            key_int4_p   = past_key_value[0]
            key_scale_p  = past_key_value[1]
            key_zero_p   = past_key_value[2]
            val_int4_p   = past_key_value[3]
            val_scale_p  = past_key_value[4]
            val_zero_p   = past_key_value[5]

            new_k_q, new_k_s, new_k_z = quantize_int4_pertoken(key_states.contiguous(), g)
            new_v_q, new_v_s, new_v_z = quantize_int4_pertoken(value_states_sm.contiguous(), g)

            key_int4  = torch.cat([key_int4_p,  new_k_q], dim=2)
            key_scale = torch.cat([key_scale_p, new_k_s], dim=2)
            key_zero  = torch.cat([key_zero_p,  new_k_z], dim=2)
            val_int4  = torch.cat([val_int4_p,  new_v_q], dim=2)
            val_scale = torch.cat([val_scale_p, new_v_s], dim=2)
            val_zero  = torch.cat([val_zero_p,  new_v_z], dim=2)

            key_deq = dequantize_int4_pertoken(key_int4, key_scale, key_zero, g)
            val_deq = dequantize_int4_pertoken(val_int4, val_scale, val_zero, g)

            key_deq = repeat_kv(key_deq, self.num_key_value_groups)
            val_deq = repeat_kv(val_deq, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_deq.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_deq)
            attn_output = unsmooth_value(attn_output, self.s_V, self.num_key_value_groups)

        else:
            key_rep = repeat_kv(key_states, self.num_key_value_groups)
            val_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, val_rep)

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


class Qwen2DecoderLayer_SmoothKV(Qwen2DecoderLayer_KIVI):
    def __init__(self, config, s_K=None, s_V=None):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention_SmoothKV(config=config, s_K=s_K, s_V=s_V)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2Model_SmoothKV(Qwen2Model_KIVI):
    def __init__(self, config, calib=None):
        prev_use_flash = getattr(config, "use_flash", False)
        config.use_flash = True
        super().__init__(config)
        config.use_flash = prev_use_flash
        layers = []
        for i in range(config.num_hidden_layers):
            sK = calib["s_K"][i] if calib is not None else None
            sV = calib["s_V"][i] if calib is not None else None
            layers.append(Qwen2DecoderLayer_SmoothKV(config, s_K=sK, s_V=sV))
        self.layers = nn.ModuleList(layers)


class Qwen2ForCausalLM_SmoothKV(Qwen2ForCausalLM_KIVI):
    _calib = None

    def __init__(self, config):
        super(Qwen2PreTrainedModel, self).__init__(config)
        self.model = Qwen2Model_SmoothKV(config, calib=self._calib)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained_with_calib(cls, model_path, calib_path, config, **kwargs):
        calib = torch.load(calib_path, map_location="cpu", weights_only=False)
        cls._calib = calib
        try:
            model = cls.from_pretrained(model_path, config=config, **kwargs)
        finally:
            cls._calib = None
        return model
