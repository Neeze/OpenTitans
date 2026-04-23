from __future__ import annotations

from enum import Enum
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BiasType(Enum):
    L2 = "l2"
    HUBER = "huber"
    LP = "lp"
    KL = "kl"


def l2_bias(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).pow(2).mean(dim=-1)


def huber_bias(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    return loss.mean(dim=-1)


def lp_bias(pred: Tensor, target: Tensor, p: float = 3.0, eps: float = 1e-8) -> Tensor:
    return (pred - target).abs().clamp(min=eps).pow(p).mean(dim=-1)


def kl_bias(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_log_softmax = F.log_softmax(pred, dim=-1)
    target_softmax = F.softmax(target, dim=-1)
    return F.kl_div(pred_log_softmax, target_softmax, reduction="none").sum(dim=-1)


BIAS_REGISTRY = {
    BiasType.L2: l2_bias,
    BiasType.HUBER: huber_bias,
    BiasType.LP: lp_bias,
    BiasType.KL: kl_bias,
}


class AttentionalBias(nn.Module):
    def __init__(
        self,
        bias_type: str | BiasType = BiasType.L2,
        **kwargs,
    ):
        super().__init__()

        if isinstance(bias_type, str):
            bias_type = BiasType(bias_type)

        self.bias_type = bias_type

        if kwargs:
            self.bias_fn = partial(BIAS_REGISTRY[bias_type], **kwargs)
        else:
            self.bias_fn = BIAS_REGISTRY[bias_type]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.bias_fn(pred, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bias_type={self.bias_type.value})"
