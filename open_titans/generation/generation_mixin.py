from __future__ import annotations

from typing import Optional, Callable
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from .titans_cache import TitansCache, AtlasCache


def top_k_filtering(logits: Tensor, top_k: int) -> Tensor:
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1:]
    return torch.where(logits < threshold, float("-inf"), logits)


def top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[mask] = float("-inf")
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        logits = top_k_filtering(logits, top_k)

    if 0.0 < top_p < 1.0:
        logits = top_p_filtering(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class TitansGenerationMixin(ABC):

    @abstractmethod
    def forward(self, input_ids, **kwargs):
        ...

    @abstractmethod
    def _get_num_layers(self) -> int:
        ...

    @abstractmethod
    def _uses_atlas_cache(self) -> bool:
        ...

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values=None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def _init_cache(self):
        n = self._get_num_layers()
        if self._uses_atlas_cache():
            return AtlasCache.from_layer_states([None] * n, seen_tokens=0)
        return TitansCache.from_layer_states([None] * n, seen_tokens=0)

    def _forward_with_cache(self, input_ids, past_key_values, attention_mask=None):
        cache_arg = list(past_key_values.layer_states) if past_key_values is not None else None

        output = self.forward(
            input_ids,
            attention_mask=attention_mask,
            cache=cache_arg,
            return_cache=True,
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cache = self._init_cache()

        prefill_output = self._forward_with_cache(
            input_ids, cache, attention_mask=attention_mask
        )

        logits = prefill_output.logits[:, -1, :]
        new_cache_states = prefill_output.past_key_values

        if self._uses_atlas_cache():
            cache = AtlasCache.from_layer_states(new_cache_states, seen_tokens=prompt_len)
        else:
            cache = TitansCache.from_layer_states(new_cache_states, seen_tokens=prompt_len)

        cache = cache.detach()

        next_token = sample_from_logits(logits, temperature, top_k, top_p)
        generated = [next_token]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens - 1):
            if finished.all():
                break

            model_inputs = self.prepare_inputs_for_generation(
                next_token,
                past_key_values=cache,
                attention_mask=torch.ones(batch_size, 1, device=device),
            )

            step_output = self._forward_with_cache(
                model_inputs["input_ids"],
                model_inputs["past_key_values"],
                attention_mask=model_inputs["attention_mask"],
            )

            logits = step_output.logits[:, -1, :]
            new_cache_states = step_output.past_key_values

            if self._uses_atlas_cache():
                cache = AtlasCache.from_layer_states(new_cache_states, seen_tokens=cache.get_seq_length() + 1)
            else:
                cache = TitansCache.from_layer_states(new_cache_states, seen_tokens=cache.get_seq_length() + 1)

            cache = cache.detach()

            next_token = sample_from_logits(logits, temperature, top_k, top_p)

            if eos_token_id is not None:
                just_finished = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)

            generated.append(next_token)

        generated = torch.cat(generated, dim=-1)
        return torch.cat([input_ids, generated], dim=-1)


class AtlasGenerationMixin(TitansGenerationMixin):

    def _uses_atlas_cache(self) -> bool:
        return True

    def _forward_with_cache(self, input_ids, past_key_values, attention_mask=None):
        cache_arg = [s for s in past_key_values.layer_states] if past_key_values is not None else None
        output = self.forward(
            input_ids,
            attention_mask=attention_mask,
            cache=cache_arg,
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        enable_ttt_grad: bool = False,
    ) -> Tensor:
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cache = self._init_cache()

        grad_ctx = torch.enable_grad if enable_ttt_grad else torch.no_grad

        with grad_ctx():
            prefill_output = self._forward_with_cache(
                input_ids, cache, attention_mask=attention_mask
            )

        logits = prefill_output.logits[:, -1, :]
        new_cache_states = prefill_output.past_key_values

        cache = AtlasCache.from_layer_states(new_cache_states, seen_tokens=prompt_len)
        cache = cache.detach()

        next_token = sample_from_logits(logits, temperature, top_k, top_p)
        generated = [next_token]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens - 1):
            if finished.all():
                break

            model_inputs = self.prepare_inputs_for_generation(
                next_token,
                past_key_values=cache,
                attention_mask=torch.ones(batch_size, 1, device=device),
            )

            with grad_ctx():
                step_output = self._forward_with_cache(
                    model_inputs["input_ids"],
                    model_inputs["past_key_values"],
                    attention_mask=model_inputs["attention_mask"],
                )

            logits = step_output.logits[:, -1, :]
            new_cache_states = step_output.past_key_values

            cache = AtlasCache.from_layer_states(
                new_cache_states,
                seen_tokens=cache.get_seq_length() + 1,
            )
            cache = cache.detach()

            next_token = sample_from_logits(logits, temperature, top_k, top_p)

            if eos_token_id is not None:
                just_finished = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)

            generated.append(next_token)

        generated = torch.cat(generated, dim=-1)
        return torch.cat([input_ids, generated], dim=-1)
