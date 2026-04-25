from __future__ import annotations
from typing import Callable, Optional
from functools import partial
import tqdm

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend

from ...modules import NeuralMemory
from ...generation import TitansGenerationMixin
from ..modeling_utils import (
    PreTrainedModel,
    exists,
    default,
    FeedForward,
    TitansCausalLMOutputWithPast,
    min_p_filter,
    gumbel_sample
)
from .configuration_mac import TitansMACConfig

LinearNoBias = partial(nn.Linear, bias=False)

class MACAttention(nn.Module):
    def __init__(self, config: TitansMACConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.dim_head = config.dim_head
        self.segment_len = config.segment_len
        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        self.total_segment_len = self.segment_len + self.num_longterm_mem_tokens
        self.num_persist_mem_tokens = config.num_persist_mem_tokens
        
        dim_inner = self.dim_head * self.heads
        
        self.norm = nn.RMSNorm(self.dim)
        self.to_qkv = LinearNoBias(self.dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, self.dim)
        
        # Disable default causal masking to enforce the custom MAC unrestricted prefix attention mask
        self.attend = Attend(causal=False)
        self.rotary_emb = RotaryEmbedding(self.dim_head)
        
        if self.num_persist_mem_tokens > 0:
            self.persistent_memory = nn.Parameter(torch.zeros(2, self.heads, self.num_persist_mem_tokens, self.dim_head))
        else:
            self.persistent_memory = None

    def forward(self, x, cache=None):
        b, seq_len, d = x.shape
        
        x = self.norm(x)
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        
        c = self.total_segment_len
        w = seq_len // c
        q = rearrange(q, 'b h (w c) d -> (b w) h c d', w=w, c=c)
        k = rearrange(k, 'b h (w c) d -> (b w) h c d', w=w, c=c)
        v = rearrange(v, 'b h (w c) d -> (b w) h c d', w=w, c=c)
        
        if self.persistent_memory is not None:
            pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv bw h n d', bw=q.shape[0])
            k = torch.cat((pmk, k), dim=-2)
            v = torch.cat((pmv, v), dim=-2)
            
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        
        p_len = self.num_persist_mem_tokens
        m_len = self.num_longterm_mem_tokens
        prefix_len_k = p_len + m_len
        prefix_len_q = m_len
        
        # Build the specific MAC mask: Unrestricted Prefix Attention
        attn_bias = torch.full((q_len, k_len), -torch.finfo(q.dtype).max, device=q.device)
        
        # 1. Prefix queries (Long-term Memory) can attend to ALL Persistent and Long-term Memory keys bidirectionally.
        if prefix_len_q > 0:
            attn_bias[:prefix_len_q, :prefix_len_k] = 0
            
        # 2. Sequence queries can attend to ALL prefix keys, and causally to Sequence keys within the same segment.
        if q_len > prefix_len_q:
            attn_bias[prefix_len_q:, :prefix_len_k] = 0
            seq_len_q = q_len - prefix_len_q
            causal_mask = torch.ones((seq_len_q, seq_len_q), dtype=torch.bool, device=q.device).tril()
            attn_bias[prefix_len_q:, prefix_len_k:][causal_mask] = 0
            
        out, _ = self.attend(q, k, v, attn_bias=attn_bias)
        
        out = rearrange(out, '(b w) h c d -> b (w c) (h d)', b=b, w=w)
        out = self.to_out(out)
        
        return out

class MACBlock(nn.Module):
    def __init__(self, config: TitansMACConfig, use_neural_memory: bool, neural_memory_model: Optional[nn.Module] = None, **neural_memory_kwargs):
        super().__init__()
        self.use_neural_memory = use_neural_memory
        
        if use_neural_memory:
            self.neural_memory = NeuralMemory(
                dim=config.hidden_size,
                chunk_size=config.segment_len + config.num_longterm_mem_tokens,
                model=neural_memory_model,
                **neural_memory_kwargs
            )
        else:
            self.neural_memory = None
            
        self.attention = MACAttention(config)
        self.ff = FeedForward(dim=config.hidden_size, mult=config.intermediate_size // config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size)
        
    def forward(self, x, mem_state=None):
        if self.use_neural_memory:
            retrieved, next_mem_state = self.neural_memory(x, state=mem_state)
            
            mems_len = self.attention.num_longterm_mem_tokens
            if mems_len > 0:
                c_total = self.attention.total_segment_len
                w = x.shape[1] // c_total
                x_chunks = rearrange(x, 'b (w c) d -> b w c d', c=c_total)
                retrieved_chunks = rearrange(retrieved, 'b (w c) d -> b w c d', c=c_total)
                
                # Add retrieved memory ONLY to the longterm_mems portion
                x_mems = x_chunks[:, :, :mems_len] + retrieved_chunks[:, :, :mems_len]
                x_si = x_chunks[:, :, mems_len:]
                
                x_chunks = torch.cat((x_mems, x_si), dim=2)
                x = rearrange(x_chunks, 'b w c d -> b (w c) d')
        else:
            next_mem_state = None
            
        x = x + self.attention(x)
        x = x + self.ff(self.norm(x))
        
        return x, next_mem_state

class TitansMACModel(TitansGenerationMixin, PreTrainedModel):
    def __init__(self, config: TitansMACConfig, neural_memory_model: Optional[nn.Module] = None, **neural_memory_kwargs):
        super().__init__(config)
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=config.hidden_size, num_axial_dims=2)
        
        self.segment_len = config.segment_len
        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        
        if self.num_longterm_mem_tokens > 0:
            self.longterm_mems = nn.Parameter(torch.randn(config.num_longterm_mem_tokens, config.hidden_size) * 0.02)
        else:
            self.register_parameter('longterm_mems', None)
            
        self.layers = nn.ModuleList()
        neural_memory_layers = config.neural_memory_layers if config.neural_memory_layers else []
        
        for layer_idx in range(1, config.num_hidden_layers + 1):
            use_neural_memory = layer_idx in neural_memory_layers
            self.layers.append(MACBlock(config, use_neural_memory, neural_memory_model, **neural_memory_kwargs))
            
        self.norm = nn.RMSNorm(config.hidden_size)
        self.to_logits = LinearNoBias(config.hidden_size, config.vocab_size) 
        
    def _get_num_layers(self) -> int:
        return len(self.layers)

    def _uses_atlas_cache(self) -> bool:
        return False

    def forward(self, input_ids, return_loss=False, return_loss_breakdown=False, disable_flex_attn=False, cache=None, return_cache=False, factorized_pos_emb=None, labels=None, attention_mask=None):
        x = input_ids
        if return_loss and labels is None:
            x, labels = x[:, :-1], x[:, 1:]
            
        b, seq_len = x.shape
        x = self.token_emb(x)
        
        pad_len = (self.segment_len - (seq_len % self.segment_len)) % self.segment_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            
        seq_len_padded = seq_len + pad_len
        x = x + self.axial_pos_emb.forward_with_seq_len(seq_len_padded, (self.segment_len,))
        
        if self.longterm_mems is not None:
            c = self.segment_len
            w = x.shape[1] // c
            x_chunks = rearrange(x, 'b (w c) d -> b w c d', w=w, c=c)
            mems = repeat(self.longterm_mems, 'm d -> b w m d', b=b, w=w)
            x_interleaved = torch.cat((mems, x_chunks), dim=2)
            x = rearrange(x_interleaved, 'b w c_total d -> b (w c_total) d')
            
        next_caches = []
        for i, layer in enumerate(self.layers):
            mem_state = cache[i] if cache is not None else None
            x, next_mem_state = layer(x, mem_state=mem_state)
            next_caches.append(next_mem_state)
            
        if self.longterm_mems is not None:
            c_total = self.segment_len + self.num_longterm_mem_tokens
            w = x.shape[1] // c_total
            x_chunks = rearrange(x, 'b (w c_total) d -> b w c_total d', c_total=c_total)
            x_out = x_chunks[:, :, self.num_longterm_mem_tokens:]
            x = rearrange(x_out, 'b w c d -> b (w c) d')
            
        if pad_len > 0:
            x = x[:, :-pad_len]
            
        x = self.norm(x)
        logits = self.to_logits(x)
        
        if not return_loss:
            if not return_cache:
                return TitansCausalLMOutputWithPast(loss=None, logits=logits, past_key_values=None)
            return TitansCausalLMOutputWithPast(loss=None, logits=logits, past_key_values=next_caches)
            
        loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
        return TitansCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=next_caches)

