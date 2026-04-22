import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from einops import rearrange, pack, unpack

class PreTrainedModel(nn.Module):
    """Base class for all models."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, path, config):
        model = cls(config)
        # Placeholder for weight loading logic
        return model

    def save_pretrained(self, path):
        # Placeholder for weight saving logic
        pass



def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)
    def inverse(out, inv_pattern=None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))
    return packed, inverse

def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)

def pad_and_segment_with_inverse(seq, segment_len, fold_into_batch=True, inverse_remove_pad=True):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)
    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0
    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))
    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n=segment_len)
    def inverse(out):
        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b=batch)
        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]
        return out
    return seq, inverse

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim=-1, keepdim=True)

def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)



class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )
