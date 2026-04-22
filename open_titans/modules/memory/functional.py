from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, pack, unpack
from tensordict import TensorDict


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def divisible_by(num, den):
    return (num % den) == 0


def round_down_multiple(seq, mult):
    return seq // mult * mult


def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult


def safe_cat(inputs, dim=-2):
    inputs = tuple(filter(exists, inputs))
    if len(inputs) == 0:
        return None
    if len(inputs) == 1:
        return inputs[0]
    return torch.cat(inputs, dim=dim)


def is_empty_tensor(t):
    return t.numel() == 0


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))


def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))


def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]


def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return nn.Identity()
    if len(modules) == 1:
        return modules[0]
    return nn.Sequential(*modules)


def softclamp_max(t, max_value):
    half = max_value / 2
    return ((t / half).tanh() * half) + half


def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t
    t, inverse = pack_one_with_inverse(t, "bn *")
    norm = t.norm(dim=-1, keepdim=True)
    clamped_norm = softclamp_max(norm, max_value)
    t = t * (clamped_norm / norm)
    return inverse(t)


def newtonschulz5(t, steps=5, eps=1e-7, coefs=(3.4445, -4.7750, 2.0315)):
    if t.ndim <= 3:
        return t
    shape = t.shape
    should_transpose = shape[-2] > shape[-1]
    if should_transpose:
        t = t.transpose(-1, -2)
    t, inv_pack = pack_one_with_inverse(t, "* i j")
    t = t / t.norm(dim=(-1, -2), keepdim=True).clamp(min=eps)
    a, b, c = coefs
    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t
    if should_transpose:
        t = t.transpose(-1, -2)
    return inv_pack(t)


def default_adaptive_step_transform(adaptive_step, max_lr=1e-2):
    return adaptive_step.sigmoid() * max_lr


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim=-1)


class MultiheadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.0)


class AveragePool(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x, chunk_size=None):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, "b (n c) d -> b n d", "mean", c=chunk_size)


class AttentionPool(nn.Module):
    def __init__(self, dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(self, x, chunk_size=None):
        chunk_size = default(chunk_size, self.chunk_size)
        x = rearrange(x, "b (n c) d -> b n c d", c=chunk_size)
        attn = self.to_attn_logits(x).softmax(dim=-2)
        return reduce(x * attn, "b n c d -> b n d", "sum")
