from __future__ import annotations
import torch
import torch.nn as nn
from torch import stack, Tensor
from torch.nn import Module, ModuleList
from einops import repeat, rearrange, pack
import torch.nn.functional as F
from copy import deepcopy
from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions

from ..modeling_utils import (
    PreTrainedModel, exists, default, GEGLU, FeedForward, pack_with_inverse, TitansCausalLMOutputWithPast
)
from ...modules import NeuralMemory
from .configuration_mag import TitansMAGConfig
from x_transformers.attend import Attend
from rotary_embedding_torch import RotaryEmbedding

flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, window_size, dim_head=64, heads=8, use_flex_attn=False, num_persist_mem_tokens=0):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        dim_inner = dim_head * heads
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal=True)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.window_size = window_size
        self.use_flex_attn = use_flex_attn
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.heads = heads
        self.dim_head = dim_head

    def forward(self, seq, disable_flex_attn=False, output_gating=None):
        batch, seq_len = seq.shape[:2]
        seq = self.norm(seq)
        
        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn and flex_attention is not None:
            def create_mag_mask(_, __, q_idx, kv_idx):
                is_persist = kv_idx < self.num_persist_mem_tokens
                causal = q_idx >= kv_idx
                sliding = (q_idx - kv_idx) <= self.window_size
                return causal & (is_persist | sliding)
                
            block_mask = create_block_mask(create_mag_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            idx = torch.arange(seq_len, device=seq.device)
            q_idx = rearrange(idx, 'i -> i 1')
            k_idx = rearrange(idx, 'j -> 1 j')
            
            dist = q_idx - k_idx
            causal_mask = dist >= 0
            is_persist = k_idx < self.num_persist_mem_tokens
            sliding_mask = dist <= self.window_size
            
            mask = causal_mask & (is_persist | sliding_mask)
            mask = repeat(mask, 'i j -> b 1 i j', b=batch)
            
            out, _ = self.attend(q, k, v, mask=mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out


class TitansMAGModel(PreTrainedModel):
    def __init__(self, config: TitansMAGConfig, neural_memory_model: Module | None = None, **neural_memory_kwargs):
        super().__init__(config)
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=config.hidden_size, num_axial_dims=1)
        
        self.num_persist_mem_tokens = config.num_persist_mem_tokens
        if self.num_persist_mem_tokens > 0:
            self.persist_mems = nn.Parameter(torch.randn(config.num_persist_mem_tokens, config.hidden_size) * 0.02)
        else:
            self.register_parameter('persist_mems', None)

        init_hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(
            config.num_residual_streams, dim=config.hidden_size, add_stream_embed=True, disable=config.num_residual_streams == 1
        )

        self.layers = ModuleList([])
        
        layers = tuple(range(1, config.num_hidden_layers + 1))
        neural_memory_layers = config.neural_memory_layers if config.neural_memory_layers else layers
        self.neural_mem_weight_residual = config.neural_mem_weight_residual
        is_first_neural_mem = True
        
        for layer in layers:
            attn = SlidingWindowAttention(
                dim=config.hidden_size,
                dim_head=config.dim_head,
                heads=config.num_attention_heads,
                window_size=config.window_size,
                use_flex_attn=config.use_flex_attn,
                num_persist_mem_tokens=config.num_persist_mem_tokens
            )
            
            mem = None
            mem_hyper_conn = None
            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual=False) 
                
                mem = NeuralMemory(
                    dim=config.hidden_size,
                    chunk_size=config.neural_memory_segment_len,
                    batch_size=config.neural_memory_batch_size,
                    model=deepcopy(neural_memory_model) if neural_memory_model else None,
                    qkv_receives_diff_views=False,
                    accept_weight_residual=config.neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )
                is_first_neural_mem = False
                
            ff = FeedForward(dim=config.hidden_size, mult=config.intermediate_size // config.hidden_size)
            
            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(), # attn
                init_hyper_conn(), # ff
                mem,
                attn,
                ff,
            ]))

        self.norm = nn.RMSNorm(config.hidden_size)
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, return_loss=False, disable_flex_attn=False, labels=None):
        x = input_ids
        if return_loss and labels is None:
            x, labels = x[:, :-1], x[:, 1:]
            
        batch, seq_len = x.shape[:2]
        
        x = self.token_emb(x)
        # Note: applying axial positional embedding to the sequence before prepending persist mems
        pos_emb = self.axial_pos_emb((seq_len,))
        x = x + pos_emb

        if exists(self.persist_mems):
            mems = repeat(self.persist_mems, 'n d -> b n d', b=batch)
            x, inverse_pack_mems = pack_with_inverse((mems, x), 'b * d')

        mem_weight_residual = None

        x = self.expand_streams(x)
        
        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem, attn, ff in self.layers:
            attn_out_gates = None
            
            if exists(mem):
                mem_input, _ = mem_hyper_conn(x)
                retrieved, next_neural_mem_cache = mem(mem_input, prev_weights=mem_weight_residual)
                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates
                
                attn_out_gates = retrieved.sigmoid()
                
            attn_in, add_attn_residual = attn_hyper_conn(x)
            attn_out = attn(attn_in, disable_flex_attn=disable_flex_attn, output_gating=attn_out_gates)
            x = add_attn_residual(attn_out)
            
            ff_in, add_ff_residual = ff_hyper_conn(x)
            ff_out = ff(ff_in)
            x = add_ff_residual(ff_out)
            
        x = self.reduce_streams(x)
        
        if exists(self.persist_mems):
            _, x = inverse_pack_mems(x)
            
        x = self.norm(x)
        logits = self.to_logits(x)
        
        if not return_loss:
            return TitansCausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None
            )
            
        loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
        return TitansCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None
        )
