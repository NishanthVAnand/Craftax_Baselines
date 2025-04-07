from typing import Tuple, Optional, Union
from functools import partial

import torch

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    KwargsForCausalLM,
)


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)

    def forward_hidden_output(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_layer: Optional[list] = None,
        decay: Optional[float] = 0.9,
        emb_type: Optional[str] = "mean",
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[Tuple, Tuple]:
        """
        forward pass of the model, returning the hidden states and attention weights
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        indices = attention_mask.sum(1) - 1
        if emb_type == "exp":
            _, seq_len, _ = hidden_states.shape
            distance_from_last = indices.unsqueeze(1) - torch.arange(
                seq_len, device=indices.device
            )
            weights = (decay**distance_from_last) * attention_mask

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if output_hidden_states and 0 in target_layer:
            if emb_type == "exp":
                temp_hidden_states = (hidden_states * weights.unsqueeze(-1)).sum(axis=1)
                all_hidden_states += (
                    temp_hidden_states / temp_hidden_states.sum(axis=1, keepdim=True),
                )
            elif emb_type == "mean":
                all_hidden_states += (
                    (hidden_states * attention_mask.unsqueeze(2)).sum(axis=1)
                    / attention_mask.sum(axis=1).unsqueeze(1),
                )

        for idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
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
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_hidden_states and (idx + 1) in target_layer:
                if emb_type == "exp":
                    temp_hidden_states = (hidden_states * weights.unsqueeze(-1)).sum(
                        axis=1
                    )
                    all_hidden_states += (
                        temp_hidden_states
                        / temp_hidden_states.sum(axis=1, keepdim=True),
                    )
                elif emb_type == "mean":
                    all_hidden_states += (
                        (hidden_states * attention_mask.unsqueeze(2)).sum(axis=1)
                        / attention_mask.sum(axis=1).unsqueeze(1),
                    )

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            if idx + 1 == max(target_layer):
                return all_hidden_states, all_self_attns

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            if emb_type == "exp":
                temp_hidden_states = (hidden_states * weights.unsqueeze(-1)).sum(axis=1)
                all_hidden_states += (
                    temp_hidden_states / temp_hidden_states.sum(axis=1, keepdim=True),
                )
            elif emb_type == "mean":
                all_hidden_states += (
                    (hidden_states * attention_mask.unsqueeze(2)).sum(axis=1)
                    / attention_mask.sum(axis=1).unsqueeze(1),
                )

        return (all_hidden_states, all_self_attns)


class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)

    def forward_hidden_states(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_layer: Optional[list] = None,
        decay: Optional[float] = 0.9,
        emb_type: Optional[str] = "mean",
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Tuple[Tuple, Tuple]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs: Tuple = self.model.forward_hidden_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            target_layer=target_layer,
            decay=decay,
            emb_type=emb_type,
            **kwargs,
        )
        hidden_states, all_self_attns = outputs

        return (hidden_states, all_self_attns)
