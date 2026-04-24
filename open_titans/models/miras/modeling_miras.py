from __future__ import annotations

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from ...modules import NeuralMemory
from ...modules.attention import AttentionalBias
from ..modeling_utils import (
    PreTrainedModel,
    FeedForward,
    TitansCausalLMOutputWithPast,
)
from .configuration_miras import MirasConfig

LinearNoBias = partial(nn.Linear, bias=False)


VARIANT_CONFIG = {
    "yaad": {"bias_type": "huber"},
    "moneta": {"bias_type": "lp"},
    "memora": {"bias_type": "kl"},
}


def _build_bias_fn(bias_type: str, config: MirasConfig):
    kwargs = {}
    if bias_type == "huber":
        kwargs["delta"] = config.huber_delta
    elif bias_type == "lp":
        kwargs["p"] = config.lp_norm
    return AttentionalBias(bias_type=bias_type, **kwargs)


class MirasLayer(nn.Module):
    def __init__(self, config: MirasConfig, neural_memory_model: nn.Module | None = None):
        super().__init__()
        dim = config.hidden_size
        heads = config.num_attention_heads
        dim_head = config.dim_head
        dim_inner = heads * dim_head

        self.heads = heads
        self.dim_head = dim_head

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        variant_cfg = VARIANT_CONFIG[config.variant]
        bias_fn = _build_bias_fn(variant_cfg["bias_type"], config)

        self.neural_memory = NeuralMemory(
            dim=dim,
            heads=config.mem_heads,
            chunk_size=config.chunk_size,
            store_memory_loss_fn=bias_fn,
            model=deepcopy(neural_memory_model) if neural_memory_model else None,
            momentum=config.momentum,
            momentum_order=config.momentum_order,
            pre_rmsnorm=config.pre_rmsnorm,
        )

        self.mem_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.ff = FeedForward(dim=dim, mult=config.intermediate_size / dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None, mem_state=None):
        residual = x
        x_normed = self.norm1(x)

        attn_out = self._attention(x_normed, attention_mask=attention_mask)

        retrieved, next_mem_state = self.neural_memory(x_normed, state=mem_state)
        gate = self.mem_gate(x_normed)
        x = residual + attn_out + gate * retrieved

        x = x + self.ff(self.norm2(x))
        return x, next_mem_state

    def _attention(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        scale = self.dim_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1,
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        
        if attention_mask is not None:
            pad_mask = ~attention_mask.view(b, 1, 1, n).bool()
            attn = attn.masked_fill(pad_mask, float("-inf"))
            
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MirasModel(PreTrainedModel):
    def __init__(self, config: MirasConfig, neural_memory_model: nn.Module | None = None):
        super().__init__(config)
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)

        self.layers = nn.ModuleList([
            MirasLayer(config, neural_memory_model=neural_memory_model)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.RMSNorm(config.hidden_size)
        self.to_logits = LinearNoBias(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None, cache=None):
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        next_caches = []
        for i, layer in enumerate(self.layers):
            mem_state = cache[i] if cache is not None else None
            x, next_mem_state = layer(x, attention_mask=attention_mask, mem_state=mem_state)
            next_caches.append(next_mem_state)

        x = self.norm(x)
        logits = self.to_logits(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b n v -> b v n"), labels,
            )

        return TitansCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=next_caches,
        )
