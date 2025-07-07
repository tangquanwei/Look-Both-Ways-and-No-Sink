import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel, 
    LlamaDecoderLayer, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    _prepare_4d_causal_attention_mask_with_cache_position
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from .modeling_utils import *
from functools import partial

logger = logging.getLogger(__name__)

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # -----------------------------------------------------------------------------------------------------------------
        self.architecture = getattr(config, 'architecture', 'NONE')
        self.mask_type = getattr(config, 'mask_type', "MASK0")
        self.num_unsink_layers = getattr(config, 'num_unsink_layers', 0)
        self.num_bidir_layers = getattr(config, 'num_bidir_layers', 0)
        self.unsink_layers = set(getattr(config, 'unsink_layers', set()))
        self.bidir_layers = set(getattr(config, 'bidir_layers', set()))
        self.connect_layers = getattr(config, 'res_connect', None)
        self.use_res_connect = partial(use_res_connect, self.num_unsink_layers, self.connect_layers)
        self.num_hidden_layers = config.num_hidden_layers # the total number of converted backbone layers
        num_converted_layers = self.num_unsink_layers + self.num_bidir_layers
        _is_mask0 = self.mask_type == "MASK0"

        assert not ((self.unsink_layers or self.bidir_layers) and (self.num_unsink_layers or self.num_bidir_layers))
        assert num_converted_layers <= self.num_hidden_layers

        if self.architecture == "EXTEND":
            self.num_hidden_layers += num_converted_layers
            self.layers.extend([LlamaDecoderLayer(config, config.num_hidden_layers + layer_idx) for layer_idx in range(num_converted_layers)])
        elif self.architecture == "EXTRA":
            self.num_hidden_layers = num_converted_layers

        if not (self.unsink_layers or self.bidir_layers):
            self.unsink_layers = self.num_hidden_layers - self.num_unsink_layers
            self.bidir_layers = self.num_hidden_layers - self.num_unsink_layers - self.num_bidir_layers
            for i in range(self.bidir_layers, self.num_hidden_layers if _is_mask0 else self.unsink_layers):
                self.layers[i].self_attn.is_causal = False
        else:
            self.unsink_layers = {layer if layer >= 0 else layer + self.num_hidden_layers for layer in self.unsink_layers}
            self.bidir_layers = {layer if layer >= 0 else layer + self.num_hidden_layers for layer in self.bidir_layers}
            for i in self.bidir_layers | (self.unsink_layers if _is_mask0 else set()):
                self.layers[i].self_attn.is_causal = False
        # -----------------------------------------------------------------------------------------------------------------

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        _is_flash_attn = self.config._attn_implementation == "flash_attention_2"
        _is_intera = self.architecture in {'INTER', 'EXTRA'}
        _is_mask0 = self.mask_type == "MASK0"

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        bidir_attention_mask = get_noncausal_attention_mask(self, attention_mask, input_ids.shape)        
        unsink_attention_mask = get_noncausal_attention_mask_0(self, attention_mask, input_ids.shape) if _is_mask0 else \
                                get_backward_attention_mask(self, attention_mask, inputs_embeds, output_attentions)
        
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        h1, h2 = None, 0
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for i in range(self.num_hidden_layers):
            decoder_layer = self.layers[i]
            if isinstance(self.unsink_layers, int):
                is_unsink = i >= self.unsink_layers
                is_bidir = i >= self.bidir_layers and not is_unsink
            else:
                is_unsink = i in self.unsink_layers
                is_bidir = i in self.bidir_layers
            layer_mask = unsink_attention_mask if is_unsink else \
                    bidir_attention_mask if is_bidir else causal_mask
            
            reverse_flag = is_unsink and _is_flash_attn and not _is_mask0
           
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if i == self.bidir_layers:
                h1 = hidden_states
            if self.use_res_connect(i):
                hidden_states += h2
                h2 = hidden_states

            hidden_states = flip_tensor(hidden_states, reverse_flag)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            
            hidden_states = flip_tensor(layer_outputs[0], reverse_flag)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        forward_hidden_states = h1 if _is_intera else h2
        for i in range(self.bidir_layers, self.config.num_hidden_layers if _is_intera else 0):
            decoder_layer = self.layers[i]

            tem = decoder_layer.self_attn.is_causal
            decoder_layer.self_attn.is_causal = True

            forward_hidden_states = decoder_layer(
                forward_hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
            
            decoder_layer.self_attn.is_causal = tem

        hidden_states += forward_hidden_states

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    def freeze_model(self, config=None):
        if config.freeze_type == "all":
            for param in self.parameters():
                param.requires_grad = False

        elif config.freeze_type == "backbone":
            self.embed_tokens.weight.requires_grad = False

            for param in self.layers[:self.num_hidden_layers - config.num_unfreeze_layers].parameters():
                param.requires_grad = False

            if config.num_unfreeze_layers == 0:
                self.norm.weight.requires_grad = False

    def model_init(self):
        first_new_layer = self.config.num_hidden_layers
        for layer in range(first_new_layer, len(self.layers)):
            self.layers[layer].load_state_dict(self.layers[first_new_layer - 1].state_dict())
