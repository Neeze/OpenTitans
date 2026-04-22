from __future__ import annotations
from typing import Callable
from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple
import tqdm
import torch
from torch import nn, stack, cat, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange
from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions
from ...modules import NeuralMemory
from ..modeling_utils import (
    PreTrainedModel,
    exists,
    default,
    identity,
    divisible_by,
    round_up_multiple,
    round_down_multiple,
    pack_with_inverse,
    pad_at_dim,
    pad_and_segment_with_inverse,
    log,
    gumbel_noise,
    gumbel_sample,
    min_p_filter,
    GEGLU,
    FeedForward,
)
from .configuration_mac import TitansMACConfig

flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding=False):
    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem
        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask
        return is_persist_mem | (~is_persist_mem & causal_mask)
    block_mask = create_block_mask(create_mac_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len + persist_mem_len, _compile=True)
    return block_mask

LinearNoBias = partial(Linear, bias=False)
AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))



class SegmentedAttention(Module):
    def __init__(self, dim, segment_len, num_persist_mem_tokens=0, num_longterm_mem_tokens=0, dim_head=64, heads=8, sliding=False, accept_value_residual=False, attend_kwargs=None, use_flex_attn=False):
        super().__init__()
        if attend_kwargs is None:
            attend_kwargs = {}
        self.norm = nn.RMSNorm(dim)
        dim_inner = dim_head * heads
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal=True, **attend_kwargs)
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)
        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None
        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.total_segment_len = segment_len + num_longterm_mem_tokens
        self.sliding = sliding
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))
        self.use_flex_attn = use_flex_attn
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(self, token, cache, value_residual=None, output_gating=None):
        batch = token.shape[0]
        token = self.norm(token)
        q, k, v = self.to_qkv(token).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)
        ck, cv = cache
        k = cat((ck, k), dim=-2)
        v = cat((cv, v), dim=-2)
        next_cache = (k, v)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        q, k, v = tuple(rearrange(t, 'b h n d -> b h n d') for t in (q, k, v))
        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b=k.shape[0])
        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)
        out, _ = self.attend(q, k, v)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if exists(output_gating):
            out = out * output_gating
        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(self, seq, value_residual=None, flex_attn_fn=None, output_gating=None, cache=None):
        batch, seq_len = seq.shape[:2]
        seq = self.norm(seq)
        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)
        next_cache = (k, v)
        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b=batch)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)
        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)
        out = flex_attn_fn(q, k, v)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if exists(output_gating):
            out = out * output_gating
        return out, AttnIntermediates(orig_v, next_cache)

    def forward(self, seq, value_residual=None, flex_attn_fn=None, disable_flex_attn=False, output_gating=None, cache=None):
        is_inferencing = exists(cache)
        if is_inferencing:
            return self.forward_inference(seq, cache, value_residual, output_gating=output_gating)
        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating=output_gating, cache=cache)
        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens
        batch, seq_len = seq.shape[:2]
        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch=False)
        seq = self.norm(seq)
        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)
        next_cache = tuple(map(inverse_segment, (k, v)))
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n=total_segment_len) for t in (q, k, v))
        attend_kwargs = dict()
        if self.sliding:
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b=batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value=0., dim=1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim=-2)
            v = cat((v[:, :-1], v[:, 1:]), dim=-2)
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))
            idx = torch.arange(seq.shape[-2], device=seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n=total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim=0, value=-1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim=-1)
            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')
            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value=True)
            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b=batch)
            attend_kwargs.update(mask=sliding_mask)
        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b=k.shape[0])
        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)
        out, _ = self.attend(q, k, v, **attend_kwargs)
        out = self.merge_heads(out)
        out = self.to_out(out)
        out = rearrange(out, '(b w) n d -> b (w n) d', b=batch)
        out = inverse_segment(out)
        if exists(output_gating):
            out = out * output_gating
        return out, AttnIntermediates(orig_v, next_cache)

class TitansMACModel(PreTrainedModel):
    def __init__(self, config: TitansMACConfig, neural_memory_model: Module | None = None, **neural_memory_kwargs):
        super().__init__(config)
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=config.hidden_size, num_axial_dims=2)
        self.segment_len = config.segment_len
        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        has_longterm_mems = config.num_longterm_mem_tokens > 0
        if has_longterm_mems:
            self.longterm_mems = nn.Parameter(torch.randn(config.num_longterm_mem_tokens, config.hidden_size) * 0.02)
        else:
            self.register_parameter('longterm_mems', None)
        self.sliding_window_attn = config.sliding_window_attn
        self.attn_window_size = config.segment_len + config.num_longterm_mem_tokens
        init_hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(
            config.num_residual_streams, dim=config.hidden_size, add_stream_embed=True, disable=config.num_residual_streams == 1
        )
        self.layers = ModuleList([])
        self.neural_memory_segment_len = default(config.neural_memory_segment_len, config.num_longterm_mem_tokens + config.segment_len)
        layers = tuple(range(1, config.num_hidden_layers + 1))
        neural_memory_layers = config.neural_memory_layers if config.neural_memory_layers else layers
        self.neural_mem_weight_residual = config.neural_mem_weight_residual
        is_first_neural_mem = True
        for layer in layers:
            is_first = layer == 1
            attn = SegmentedAttention(
                dim=config.hidden_size,
                dim_head=config.dim_head,
                heads=config.num_attention_heads,
                segment_len=config.segment_len,
                use_flex_attn=config.use_flex_attn,
                accept_value_residual=not is_first,
                num_longterm_mem_tokens=config.num_longterm_mem_tokens,
                num_persist_mem_tokens=config.num_persist_mem_tokens,
                sliding=config.sliding_window_attn
            )
            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None
            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual=not config.neural_mem_gate_attn_output)
                if not is_first and config.neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1
                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(config.hidden_size),
                        nn.Linear(config.hidden_size, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views=3),
                        nn.Softmax(dim=-1)
                    )
                mem = NeuralMemory(
                    dim=config.hidden_size,
                    chunk_size=self.neural_memory_segment_len,
                    batch_size=config.neural_memory_batch_size,
                    model=deepcopy(neural_memory_model) if neural_memory_model else None,
                    qkv_receives_diff_views=True,
                    accept_weight_residual=config.neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )
                is_first_neural_mem = False
            ff = FeedForward(dim=config.hidden_size, mult=config.intermediate_size // config.hidden_size)
            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(),
                init_hyper_conn(),
                mem_qkv_layer_selector,
                mem,
                attn,
                ff,
            ]))
        self.norm = nn.RMSNorm(config.hidden_size)
        self.to_logits = LinearNoBias(config.hidden_size, config.vocab_size)
        self.gate_attn_output = config.neural_mem_gate_attn_output
        self.register_buffer('zero', torch.tensor(0.), persistent=False)
        self.use_flex_attn = config.use_flex_attn
        self.num_persist_mem_tokens = config.num_persist_mem_tokens

    def seq_index_is_longterm(self, seq_index):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(self, seq_len):
        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    @torch.no_grad()
    def sample(self, prompt: Tensor, seq_len: int, temperature=1.5, filter_fn: Callable=min_p_filter, filter_kwargs=None, show_progress=True, use_cache=False):
        if filter_kwargs is None:
            filter_kwargs = {'min_p': 0.1}
        was_training = self.training
        self.eval()
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)
        cache = None
        factorized_pos_emb = None
        if use_cache:
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len_with_mem, (self.neural_memory_segment_len,))
            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized=True)
        with tqdm.tqdm(total=sample_num_times, disable=not show_progress) as pbar:
            while out.shape[-1] < seq_len:
                logits, next_cache = self.forward(out, disable_flex_attn=True, cache=cache, return_cache=True, factorized_pos_emb=factorized_pos_emb)
                if use_cache:
                    cache = next_cache
                if not exists(logits):
                    continue
                logits = logits[:, -1]
                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature=temperature)
                out = torch.cat((out, sample), dim=-1)
                pbar.update(1)
        self.train(was_training)
        return out[..., prompt_seq_len:]

    def forward(self, input_ids, return_loss=False, return_loss_breakdown=False, disable_flex_attn=False, cache=None, return_cache=False, factorized_pos_emb=None, labels=None):
        x = input_ids
        if return_loss and labels is None:
            x, labels = x[:, :-1], x[:, 1:]
        batch, seq_len = x.shape[:2]
        neural_mem_segment_len = self.neural_memory_segment_len
        segment_len = self.segment_len
        num_longterm_mem_tokens = self.num_longterm_mem_tokens
        attn_window_size = self.attn_window_size
        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
        x = self.token_emb(x)
        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad=False)
        if exists(self.longterm_mems):
            mems = repeat(self.longterm_mems, 'n d -> b n d', b=x.shape[0])
            x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')
        x = inverse_segment(x)
        x = x[:, :seq_len_with_mem]
        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized=factorized_pos_emb)
        x = x + pos_emb
        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn
        flex_attn_fn = None
        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)
        is_inferencing = exists(cache)
        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)
        inference_seq_index, kv_caches, neural_mem_caches = cache
        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))
        next_kv_caches = []
        next_neural_mem_caches = []
        value_residual = None
        mem_weight_residual = None
        mem_input_layers = []
        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]
        x = self.expand_streams(x)
        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:
            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None
            if exists(mem):
                mem_input, add_residual = mem_hyper_conn(x)
                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')
                retrieved, next_neural_mem_cache = mem.forward(qkv_mem_input, state=next(neural_mem_caches, None), prev_weights=mem_weight_residual)
                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates
                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)
            attn_in, add_residual = attn_hyper_conn(x)
            mem_input_layers.append(attn_in)
            attn_out, (values, next_kv_cache) = attn(attn_in, value_residual=value_residual, disable_flex_attn=disable_flex_attn, flex_attn_fn=flex_attn_fn, output_gating=attn_out_gates, cache=next(kv_caches, None))
            mem_input_layers.append(attn_out)
            value_residual = default(value_residual, values)
            x = add_residual(attn_out)
            next_kv_caches.append(next_kv_cache)
            if exists(mem):
                next_neural_mem_caches.append(next_neural_mem_cache)
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            mem_input_layers.append(ff_out)
            x = add_ff_residual(ff_out)
        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])
            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]
            kv_cache_length = next_kv_caches.shape[-2]
            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]
            next_cache = (inference_seq_index + 1, next_kv_caches, next_neural_mem_caches)
            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)
            if is_inferencing and is_longterm_mem:
                return None, next_cache
        x = self.reduce_streams(x)
        if not is_inferencing:
            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad=False)
            if exists(self.longterm_mems):
                x, _ = inverse_pack_mems(x)
            x = inverse_segment(x)
            x = x[:, :seq_len]
        x = self.norm(x)
        logits = self.to_logits(x)
        if not return_loss:
            if not return_cache:
                return logits
            return logits, next_cache
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
