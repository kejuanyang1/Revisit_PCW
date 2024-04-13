import math
from abc import ABC
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaRMSNorm, \
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM

def generate_pcw_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]]) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    n_task_tokens = position_ids.shape[1] - sum_windows_size
    position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:  # i.e., first token is already generated
        position_ids = position_ids[:, -1].unsqueeze(-1)
    elif windows_key_values:  # i.e., we are in the first token generation
        position_ids = position_ids[:, sum_windows_size:]
    return position_ids

"""
The following code is mainly copy+paste from the original modelling_llama.py:
LlamaAttention uses a caching mechanism for the positional rotation vectors (using LlamaRotaryEmbedding). 
This mechanism forces us to override LLaMa attention layer, which in turn forces us to override the decoder, 
and model (so that the correct forward function would be called).
"""


class LlamaForCausalLMPCW(LlamaForCausalLM, ABC):
    _no_split_modules = ["LlamaDecoderLayerPCW"]

    def __init__(self, config: LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        # using our Llama model variant:
        self.model = LlamaModelPCW(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class LlamaModelPCW(LlamaModel, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # using the alternative decoder layer:
        self.layers = nn.ModuleList([LlamaDecoderLayerPCW(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class LlamaDecoderLayerPCW(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # overriding attention:
        self.self_attn = LlamaAttentionPCW(config=config)


class LlamaAttentionPCW(LlamaAttention):
    # we have to override the forward attention due to the rotary embeddings caching mechanism
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # *** changes to the original code to accommodate PCW:
        # making sure that the model generates rotary embeddings in the correct length:
        seq_len = kv_seq_len if position_ids is None else int(torch.max(position_ids) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        # *** End of changes due to PCW, the rest of the function is copy-paste from the original transformer package.

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
