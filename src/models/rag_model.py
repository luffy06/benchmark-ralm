import logging
import torch
from torch import nn
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)

class ReFusionModel(nn.Module):
    def __init__(self, model, retriever, retriever_args):
        super().__init__()
        self.model = model
        self.retriever = retriever
        self.retriever_args = retriever_args
        self.retriever.replace_modules(self.model, self.retriever_args)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        neighbors=None,
        neighbor_texts=None,
        **kwargs,
    ):
        if neighbors != None:
            self.retriever.save_in_cache(neighbors)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        neighbors=None,
        neighbor_texts=None,
        **kwargs,
    ):
        if neighbors != None:
            self.retriever.save_in_cache(neighbors)

        return self.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )
