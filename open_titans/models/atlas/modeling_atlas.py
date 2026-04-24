import torch
import torch.nn as nn
import torch.nn.functional as F

from open_titans.modules.memory.retrospective import RetrospectiveMemoryBuffer
from open_titans.optim.muon import newton_schulz5
from .configuration_atlas import TitansAtlasConfig
from ..modeling_utils import PreTrainedModel, TitansCausalLMOutputWithPast


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = config.retrospective_window

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.size()
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.ones(s, s, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)
        window_mask = torch.triu(mask, diagonal=-self.window_size + 1)
        
        attn_weights = attn_weights.masked_fill(~window_mask, float('-inf'))
        if attention_mask is not None:
            pad_mask = ~attention_mask.view(b, 1, 1, s).bool()
            attn_weights = attn_weights.masked_fill(pad_mask, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class AtlasLayerBlock(nn.Module):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.chunk_size = config.chunk_size
        self.ns_steps = config.muon_ns_steps
        
        self.norm_in = nn.RMSNorm(self.hidden_size)
        self.norm_kq = nn.RMSNorm(self.hidden_size)
        self.norm_out = nn.RMSNorm(self.hidden_size)
        
        self.to_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.to_gates = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.to_bypass = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.conv_k = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, groups=self.hidden_size)
        self.conv_q = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, groups=self.hidden_size)
        
        self.retro_buffer = RetrospectiveMemoryBuffer(config.retrospective_window, self.hidden_size)

    def forward(self, x: torch.Tensor, mem_state: torch.Tensor = None, past_buffer_k: torch.Tensor = None, past_buffer_v: torch.Tensor = None) -> tuple:
        b, s, d = x.size()
        
        if mem_state is None:
            memory_state = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        else:
            memory_state = mem_state
            
        x_norm = self.norm_in(x)
        
        k_raw = self.to_k(x_norm)
        q_raw = self.to_q(x_norm)
        v = F.silu(self.to_v(x_norm))
        
        k = F.silu(self.conv_k(k_raw.transpose(1, 2)).transpose(1, 2))
        q = F.silu(self.conv_q(q_raw.transpose(1, 2)).transpose(1, 2))
        
        k = self.norm_kq(k)
        q = self.norm_kq(q)
        
        gates = F.silu(self.to_gates(x_norm))
        gamma, eta, alpha = gates.chunk(3, dim=-1)
        
        bypass = F.silu(self.to_bypass(x_norm))
        
        out_chunks = []
        chunks_k = k.split(self.chunk_size, dim=1)
        chunks_q = q.split(self.chunk_size, dim=1)
        chunks_v = v.split(self.chunk_size, dim=1)
        chunks_eta = eta.split(self.chunk_size, dim=1)
        chunks_alpha = alpha.split(self.chunk_size, dim=1)
        
        for c_k, c_q, c_v, c_eta, c_alpha in zip(chunks_k, chunks_q, chunks_v, chunks_eta, chunks_alpha):
            ctx_v, past_buffer_v = self.retro_buffer(c_v, past_buffer_v)
            ctx_k, past_buffer_k = self.retro_buffer(c_k, past_buffer_k)
            
            chunk_len = c_k.size(1)
            ctx_len = ctx_k.size(1)
            
            pred_v = torch.matmul(ctx_k, memory_state)
            err = pred_v - ctx_v
            
            grad = torch.matmul(ctx_k.transpose(1, 2), err)
            orth_grad = newton_schulz5(grad, steps=self.ns_steps)
            
            c_eta_mean = c_eta.mean(dim=1).unsqueeze(1)
            c_alpha_mean = c_alpha.mean(dim=1).unsqueeze(1)
            
            memory_state = memory_state * c_alpha_mean - orth_grad * c_eta_mean
            
            c_out = torch.matmul(c_q, memory_state)
            out_chunks.append(c_out)
            
        out = torch.cat(out_chunks, dim=1)
        out = self.norm_out(out)
        out = out * bypass * gamma
        
        return out, memory_state, past_buffer_k, past_buffer_v


class AtlasDeepTransformersLayer(nn.Module):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__()
        self.atlas = AtlasLayerBlock(config)
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask=None, mem_state=None, past_buffer_k=None, past_buffer_v=None):
        mem_out, next_mem_state, next_buffer_k, next_buffer_v = self.atlas(x, mem_state, past_buffer_k, past_buffer_v)
        x = x + mem_out
        x = x + self.mlp(self.norm2(self.norm1(x)))
        return x, next_mem_state, next_buffer_k, next_buffer_v


class AtlasMAGLayer(nn.Module):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__()
        self.atlas = AtlasLayerBlock(config)
        self.swa1 = SlidingWindowAttention(config)
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.swa2 = SlidingWindowAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask=None, mem_state=None, past_buffer_k=None, past_buffer_v=None):
        mem_out, next_mem_state, next_buffer_k, next_buffer_v = self.atlas(x, mem_state, past_buffer_k, past_buffer_v)
        mem_out = self.norm1(mem_out)
        
        attn_out = self.swa1(x, attention_mask=attention_mask)
        attn_out = self.norm2(attn_out)
        
        fused = mem_out + attn_out
        
        out = fused + self.mlp(self.swa2(fused, attention_mask=attention_mask))
        return out, next_mem_state, next_buffer_k, next_buffer_v


class AtlasMALLayer(nn.Module):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__()
        self.atlas = AtlasLayerBlock(config)
        self.mlp1 = SwiGLU(config.hidden_size, config.intermediate_size)
        self.swa = SlidingWindowAttention(config)
        self.mlp2 = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask=None, mem_state=None, past_buffer_k=None, past_buffer_v=None):
        mem_out, next_mem_state, next_buffer_k, next_buffer_v = self.atlas(x, mem_state, past_buffer_k, past_buffer_v)
        x = x + self.mlp1(mem_out)
        attn_out = self.swa(x, attention_mask=attention_mask)
        x = x + self.mlp2(attn_out)
        return x, next_mem_state, next_buffer_k, next_buffer_v


class AtlasModel(PreTrainedModel):
    def __init__(self, config: TitansAtlasConfig):
        super().__init__(config)
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        
        if config.variant == "deep_transformers":
            layer_cls = AtlasDeepTransformersLayer
        elif config.variant == "mag":
            layer_cls = AtlasMAGLayer
        elif config.variant == "mal":
            layer_cls = AtlasMALLayer
        else:
            raise ValueError(f"Unknown variant: {config.variant}")
            
        self.layers = nn.ModuleList([
            layer_cls(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = nn.RMSNorm(config.hidden_size)
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None, cache=None):
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        next_caches = []
        for i, layer in enumerate(self.layers):
            if cache is not None and cache[i] is not None:
                mem_state, past_buffer_k, past_buffer_v = cache[i]
            else:
                mem_state, past_buffer_k, past_buffer_v = None, None, None
                
            x, next_mem_state, next_buffer_k, next_buffer_v = layer(x, attention_mask=attention_mask, mem_state=mem_state, past_buffer_k=past_buffer_k, past_buffer_v=past_buffer_v)
            next_caches.append((next_mem_state, next_buffer_k, next_buffer_v))
            
        x = self.norm(x)
        logits = self.to_logits(x)
        
        loss = None
        if labels is not None:
            # Flatten to compute Cross Entropy Loss
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return TitansCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=next_caches,
        )
