from __future__ import annotations

from collections import namedtuple
from functools import partial
from itertools import zip_longest
from typing import Callable

import math
import einx
import torch
from torch import cat, stack
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap
from torch.nn import Module, Parameter, ParameterList
from torch.utils._pytree import tree_map
from assoc_scan import AssocScan

from .memory_model import MemoryMLP, ResidualNorm
from .functional import (
    AveragePool,
    AttentionPool,
    MultiheadRMSNorm,
    Sequential,
    default,
    default_adaptive_step_transform,
    default_loss_fn,
    dict_get_value_shapes,
    divisible_by,
    exists,
    is_empty_tensor,
    newtonschulz5,
    pad_at_dim,
    rearrange_dict_values,
    repeat_dict_values,
    round_down_multiple,
    round_up_multiple,
    safe_cat,
    softclamp_grad_norm,
)


LinearNoBias = partial(nn.Linear, bias=False)


NeuralMemState = namedtuple(
    "NeuralMemState",
    ["seq_index", "weights", "cache_store_segment", "states", "updates"],
)


def mem_state_detach(state: NeuralMemState) -> NeuralMemState:
    state = tree_map(lambda t: t.detach() if torch.is_tensor(t) else t, tuple(state))
    return NeuralMemState(*state)


def _pair(v):
    return (v, v) if not isinstance(v, tuple) else v


class NeuralMemory(Module):
    def __init__(
        self,
        dim: int,
        chunk_size: int | tuple[int, int] = 1,
        batch_size: int | None = None,
        dim_head: int | None = None,
        heads: int = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr: float = 1.0,
        per_parameter_lr_modulation: bool = False,
        max_mem_layer_modulation: float = 1.0,
        per_head_learned_parameters: bool = True,
        attn_pool_chunks: bool = False,
        momentum: bool = True,
        momentum_order: int = 1,
        learned_momentum_combine: bool = False,
        learned_combine_include_zeroth: bool = False,
        num_kv_per_token: int = 1,
        qkv_receives_diff_views: bool = False,
        pre_rmsnorm: bool = True,
        post_rmsnorm: bool = False,
        qk_rmsnorm: bool = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan: bool = False,
        activation: Module | None = None,
        init_adaptive_step_bias: float | None = None,
        init_momentum_bias: float | None = None,
        init_decay_bias: float | None = None,
        accept_weight_residual: bool = False,
        spectral_norm_surprises: bool = False,
        gated_transition: bool = False,
        mem_model_norm_add_residual: bool = True,
        store_with_lookahead_value: bool = False,
        default_model_kwargs: dict = dict(depth=2, expansion_factor=4.0),
    ):
        super().__init__()

        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = _pair(chunk_size)

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)
        self.batch_size = batch_size

        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        self.qkv_receives_diff_views = qkv_receives_diff_views

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        dim_inner = dim_head * heads
        self.heads = heads

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.split_kv_heads = Rearrange("b n (h u d) -> b h (n u) d", h=heads, u=num_kv_per_token)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = (
            Sequential(
                LinearNoBias(dim, heads),
                Rearrange("b n h -> b h n 1"),
                nn.Sigmoid(),
            )
            if heads > 1
            else None
        )

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        assert not exists(next(model.buffers(), None)), "model cannot have buffers for now"

        test_shape = (3, 2, dim_head)
        with torch.no_grad():
            try:
                out = model(torch.randn(test_shape))
            except Exception:
                raise RuntimeError(f"memory model unable to accept tensor of shape {test_shape}")
            assert out.shape == test_shape, "output of memory model must match input shape"

        if mem_model_norm_add_residual:
            model = ResidualNorm(dim=dim_head, model=model)

        self.memory_model = model
        mem_model_params = dict(model.named_parameters())
        self.num_memory_parameter_tensors = len(mem_model_params)
        self.memory_model_parameter_names = list(mem_model_params.keys())
        memory_model_parameters = list(mem_model_params.values())

        if per_head_learned_parameters:
            memory_model_parameters = [repeat(p, "... -> h ...", h=heads) for p in memory_model_parameters]

        self.init_weight_shape = [p.shape for p in memory_model_parameters]
        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        self.chunk_size = chunk_size

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target)
            return (loss * loss_weights).sum(), loss

        grad_fn = grad(forward_and_loss, has_aux=True)
        self.per_sample_grad_fn = vmap(grad_fn, in_dims=(0, 0, 0, 0))

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)
        self.to_keys = Sequential(LinearNoBias(dim, dim_inner * num_kv_per_token), activation)
        self.to_values = Sequential(LinearNoBias(dim, dim_inner * num_kv_per_token), activation)

        self.store_with_lookahead_value = store_with_lookahead_value
        self.store_memory_loss_fn = store_memory_loss_fn
        self.num_kv_per_token = num_kv_per_token

        chunk_size = self.store_chunk_size
        assert not (attn_pool_chunks and chunk_size == 1), "`attn_pool_chunks` requires chunk_size > 1"

        self.reduce_to_chunk_rep = (
            AttentionPool(dim, chunk_size=chunk_size)
            if attn_pool_chunks
            else AveragePool(chunk_size=chunk_size)
        )

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange("b n (h u) -> (b h) (n u)", u=num_kv_per_token),
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr=default_step_transform_max_lr)
        self.adaptive_step_transform = adaptive_step_transform

        self.to_momentum = (
            Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange("b n (h o) -> o (b h) n 1", o=momentum_order),
            )
            if momentum
            else None
        )
        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1
            effective_order = momentum_order + 1 if learned_combine_include_zeroth else momentum_order
            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * effective_order),
                Rearrange("b n (h o) -> o (b h) n", h=heads),
                nn.Softmax(dim=0),
            )
            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        self.to_layer_modulation = (
            Sequential(
                nn.Linear(dim, heads * self.num_memory_parameter_tensors),
                Rearrange("b n (h w) -> w (b h) n", h=heads),
                nn.Sigmoid(),
            )
            if per_parameter_lr_modulation
            else None
        )
        self.max_mem_layer_modulation = max_mem_layer_modulation

        self.to_learned_weight_residual_mix = (
            Sequential(
                nn.Linear(dim, heads),
                Rearrange("b n h -> b h n"),
                nn.Sigmoid(),
            )
            if accept_weight_residual
            else None
        )

        self.max_grad_norm = max_grad_norm
        self.spectral_norm_surprises = spectral_norm_surprises

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange("b n h -> (b h) n 1"),
        )

        self.transition_gate = nn.Parameter(torch.tensor(-5.0)) if gated_transition else None

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        self.use_accelerated_scan = use_accelerated_scan
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(self, batch: int):
        if self.per_head_learned_parameters:
            return repeat_dict_values(self.memory_model_parameter_dict, "h ... -> (b h) ...", b=batch)
        return repeat_dict_values(self.memory_model_parameter_dict, "... -> bh ...", bh=batch * self.heads)

    def init_momentum(self, batch: int):
        zeros = self.memory_model_parameter_dict.clone().zero_()
        if self.per_head_learned_parameters:
            return repeat_dict_values(zeros, "h ... -> o (b h) ...", b=batch, o=self.momentum_order)
        return repeat_dict_values(zeros, "... -> o bh ...", bh=batch * self.heads, o=self.momentum_order)

    def store_memories(
        self,
        seq: Tensor,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple | None = None,
        seq_index: int = 0,
        prev_weights=None,
        mask: Tensor | None = None,
        return_surprises: bool = True,
    ):
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size
        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]
        next_seq_len_index = seq_index + round_down_seq_len

        if not exists(weights):
            weights = self.init_weights(batch)
        weights = TensorDict(weights)

        weights_for_surprise = repeat_dict_values(weights, "b ... -> b n ...", n=num_chunks)

        seq = self.store_norm(seq)
        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        adaptive_lr = self.adaptive_step_transform(self.to_adaptive_step(seq))
        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size=chunk_size)
        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)
        learned_combine = False

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()
            learned_combine = exists(self.to_learned_momentum_combine)
            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        keys = self.k_norm(self.split_kv_heads(self.to_keys(seq)))
        values = self.split_kv_heads(self.to_values(values_seq))

        keys, values = (
            rearrange(t, "b h (n c u) d -> (b h n) (c u) d", c=chunk_size, u=num_updates)
            for t in (keys, values)
        )
        adaptive_lr = rearrange(adaptive_lr, "b (n c u) -> (b n) (c u)", c=chunk_size, u=num_updates)

        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, "b (n c) -> (b h n) (c u)", h=heads, u=num_updates, c=chunk_size)
            adaptive_lr = torch.where(mask, adaptive_lr, 0.0)

        assert not (exists(self.to_learned_weight_residual_mix) ^ exists(prev_weights))

        if exists(prev_weights):
            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks
            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = rearrange(self.to_learned_weight_residual_mix(chunked_seq), "b h n -> (b h) n")
                prev_weights = prev_weights.apply(lambda t: einx.multiply("bh n, bh n ... -> bh n ...", mix, t))

            weights_for_surprise = weights_for_surprise + prev_weights

        weights_for_surprise = rearrange_dict_values(weights_for_surprise, "b n ... -> (b n) ...")

        if self.store_with_lookahead_value:
            adaptive_lr = adaptive_lr[..., :-1]
            keys = keys[..., :-1, :]
            values = values[..., 1:, :]

        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)
        grads = TensorDict(grads)

        adaptive_lr = rearrange(adaptive_lr, "(b h n) c -> b h (n c)", b=batch, h=heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, "(b h n) c -> b h (n c)", b=batch, h=heads)

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        grads = rearrange_dict_values(grads, "(b n) ... -> b n ...", b=batch * heads)

        if need_layer_lr_mod:
            grads = TensorDict({
                name: einx.multiply("b h, b h ... -> b h ...", lr_mod, t)
                for lr_mod, (name, t) in zip(layer_lr_mod, grads.items())
            })

        surprises = grads.mul(-1)

        if not exists(past_state):
            past_state = (weights, self.init_momentum(batch))

        past_last_update, past_last_momentum = past_state

        if num_chunks == 0:
            updates = rearrange_dict_values(weights, "bh ... -> bh 1 ...")
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)
            output = (updates, next_store_state)
            return (*output, (unweighted_mem_model_loss, adaptive_lr)) if return_surprises else output

        updates = TensorDict()
        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):
            update = surprise

            if has_momentum:
                momentum = surprise
                momentums = []
                last_momentum = past_last_momentum[param_name]

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev=one_last_momentum)
                    momentums.append(momentum)

                momentums = stack(momentums)
                next_last_momentum[param_name] = momentums[:, :, -1]

                if learned_combine and self.learned_combine_include_zeroth:
                    momentums = cat((rearrange(surprise, "... -> 1 ..."), momentums), dim=0)

                if not learned_combine:
                    update = momentums[-1]
                else:
                    update = einsum(combine_momentums, momentums, "o b n, o b n ... -> b n ...")

            if self.spectral_norm_surprises:
                update = newtonschulz5(update)

            update = self.assoc_scan(1.0 - decay_factor, update, prev=last_update, remove_prev=False)
            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        next_state = (next_last_update, next_last_momentum)
        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(self, seq: Tensor, weights: dict[str, Tensor]) -> Tensor:
        import math
        chunk_size = self.retrieve_chunk_size
        weights_have_expanded_shape = dict_get_value_shapes(weights) != self.init_weight_shape

        batch, seq_len = seq.shape[:2]
        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1
        is_single_token_decode = is_one_token and is_one_weight

        if is_single_token_decode:
            chunk_size = 1

        need_pad = chunk_size > 1 or not is_one_weight

        if need_pad:
            seq = pad_at_dim(seq, (1, 0), dim=1)

        seq_len_plus_one = seq.shape[-2]
        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)
        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim=1)

        weights = TensorDict(weights)
        seq = self.retrieve_norm(seq)
        queries = self.q_norm(self.split_heads(self.to_queries(seq)))

        if weights_have_expanded_shape:
            weights = rearrange_dict_values(weights, "b n ... -> (b n) ...")

        queries = rearrange(queries, "b h (n c) d -> (b h n) c d", c=chunk_size)

        batched_call = vmap(lambda params, inp: functional_call(self.memory_model, params, inp))
        values = batched_call(dict(weights), queries)

        values = rearrange(values, "(b h n) c d -> b h (n c) d", b=batch, h=self.heads)
        values = self.multihead_rmsnorm(values)

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        values = self.combine_heads(self.merge_heads(values))

        if need_pad:
            values = values[:, 1:]

        return values[:, :seq_len]

    def forward(
        self,
        seq: Tensor,
        store_seq: Tensor | None = None,
        state: NeuralMemState | None = None,
        detach_mem_state: bool = False,
        prev_weights=None,
        store_mask: Tensor | None = None,
        return_surprises: bool = False,
        ttt_batch_size: int | None = None,
    ):

        is_multi_input = self.qkv_receives_diff_views

        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, "... b d -> ... b 1 d")

        is_single_token = seq.shape[-2] == 1

        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state
        store_seq = default(store_seq, seq)

        if exists(cache_store_seq):
            store_seq = safe_cat((cache_store_seq, store_seq))

        store_seq_len = store_seq.shape[-2]
        chunk_size = self.chunk_size
        batch_size = default(ttt_batch_size, self.batch_size)
        need_update_weights = exists(batch_size)

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)
            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = divisible_by(seq_range, batch_size)
            indices = seq_range[batch_boundary] - seq_index
            indices = torch.nn.functional.pad(indices, (1, 0), value=0)
            if indices[-1] != store_seq_len:
                indices = torch.nn.functional.pad(indices, (0, 1), value=store_seq_len)
            split_sizes = (indices[1:] - indices[:-1]).tolist()
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        updates = None
        surprises = (None, None)
        gate = self.transition_gate.sigmoid() if exists(self.transition_gate) else None

        def accum_updates(past, future):
            if not exists(past):
                return future
            return TensorDict({
                name: torch.cat((pu[:, :-1], fu), dim=1)
                for (name, pu), (_, fu) in zip(past.items(), future.items())
            })

        store_seqs = store_seq.split(split_sizes, dim=-2)
        store_masks = store_mask.split(split_sizes, dim=-1) if exists(store_mask) else (None,) * len(split_sizes)

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index=seq_index,
                past_state=past_state,
                prev_weights=prev_weights,
                mask=maybe_store_mask,
                return_surprises=True,
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states
            updates = accum_updates(updates, next_updates)
            surprises = tuple(safe_cat(args, dim=-1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = TensorDict({
                    name: w.lerp(lu, gate)
                    for (name, w), (_, lu) in zip(weights.items(), last_update.items())
                })

            past_state = (last_update, last_momentum)
            weights = last_update
            next_neural_mem_state = next_neural_mem_state._replace(weights=weights, states=past_state)

        next_neural_mem_state = next_neural_mem_state._replace(updates=updates)

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = rearrange_dict_values(last_update, "b ... -> b 1 ...")

        retrieved = self.retrieve_memories(retrieve_seq, updates)

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
