from __future__ import annotations

from enum import Enum
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RetentionType(Enum):
    QUADRATIC = "quadratic"
    BREGMAN = "bregman"
    ELASTIC_NET = "elastic_net"
    F_DIVERGENCE = "f_divergence"


def quadratic_local(w: Tensor, w_prev: Tensor) -> Tensor:
    return 0.5 * (w - w_prev).pow(2).flatten(-2).mean(dim=-1)


def quadratic_global(w: Tensor, weight_decay: float = 1e-4) -> Tensor:
    return 0.5 * weight_decay * w.pow(2).flatten(-2).mean(dim=-1)


def bregman_local(w: Tensor, w_prev: Tensor, eps: float = 1e-8) -> Tensor:
    phi_w = w.pow(2)
    phi_w_prev = w_prev.pow(2)
    grad_phi = 2.0 * w_prev
    divergence = phi_w - phi_w_prev - grad_phi * (w - w_prev)
    return divergence.flatten(-2).mean(dim=-1)


def bregman_global(w: Tensor, weight_decay: float = 1e-4) -> Tensor:
    return weight_decay * w.pow(2).flatten(-2).mean(dim=-1)


def elastic_net_local(
    w: Tensor, w_prev: Tensor, alpha: float = 0.5
) -> Tensor:
    l1 = (w - w_prev).abs().flatten(-2).mean(dim=-1)
    l2 = 0.5 * (w - w_prev).pow(2).flatten(-2).mean(dim=-1)
    return alpha * l1 + (1.0 - alpha) * l2


def elastic_net_global(
    w: Tensor, l1_decay: float = 1e-5, l2_decay: float = 1e-4
) -> Tensor:
    l1 = l1_decay * w.abs().flatten(-2).mean(dim=-1)
    l2 = 0.5 * l2_decay * w.pow(2).flatten(-2).mean(dim=-1)
    return l1 + l2


def f_divergence_local(w: Tensor, w_prev: Tensor, eps: float = 1e-8) -> Tensor:
    p = F.softmax(w.flatten(-2), dim=-1)
    q = F.softmax(w_prev.flatten(-2), dim=-1)
    return F.kl_div(q.clamp(min=eps).log(), p, reduction="none").sum(dim=-1)


def f_divergence_global(w: Tensor, eps: float = 1e-8) -> Tensor:
    p = F.softmax(w.flatten(-2), dim=-1)
    uniform = torch.ones_like(p) / p.shape[-1]
    return F.kl_div(uniform.clamp(min=eps).log(), p, reduction="none").sum(dim=-1)


LOCAL_REGISTRY = {
    RetentionType.QUADRATIC: quadratic_local,
    RetentionType.BREGMAN: bregman_local,
    RetentionType.ELASTIC_NET: elastic_net_local,
    RetentionType.F_DIVERGENCE: f_divergence_local,
}

GLOBAL_REGISTRY = {
    RetentionType.QUADRATIC: quadratic_global,
    RetentionType.BREGMAN: bregman_global,
    RetentionType.ELASTIC_NET: elastic_net_global,
    RetentionType.F_DIVERGENCE: f_divergence_global,
}


class RetentionRegularization(nn.Module):
    def __init__(
        self,
        retention_type: str | RetentionType = RetentionType.QUADRATIC,
        lambda_local: float = 1.0,
        lambda_global: float = 0.01,
        learnable_lambda: bool = False,
        local_kwargs: dict | None = None,
        global_kwargs: dict | None = None,
    ):
        super().__init__()

        if isinstance(retention_type, str):
            retention_type = RetentionType(retention_type)

        self.retention_type = retention_type

        if learnable_lambda:
            self._lambda_local = nn.Parameter(torch.tensor(lambda_local))
            self._lambda_global = nn.Parameter(torch.tensor(lambda_global))
        else:
            self.register_buffer("_lambda_local", torch.tensor(lambda_local))
            self.register_buffer("_lambda_global", torch.tensor(lambda_global))

        local_fn = LOCAL_REGISTRY[retention_type]
        global_fn = GLOBAL_REGISTRY[retention_type]
        self.local_fn = partial(local_fn, **local_kwargs) if local_kwargs else local_fn
        self.global_fn = partial(global_fn, **global_kwargs) if global_kwargs else global_fn

    @property
    def lambda_local(self) -> Tensor:
        return self._lambda_local.abs()

    @property
    def lambda_global(self) -> Tensor:
        return self._lambda_global.abs()

    def forward(
        self,
        w: Tensor,
        w_prev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_loss = self.local_fn(w, w_prev)
        global_loss = self.global_fn(w)
        total = self.lambda_local * local_loss + self.lambda_global * global_loss
        return total, local_loss, global_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.retention_type.value}, "
            f"λ_local={self._lambda_local.item():.4f}, "
            f"λ_global={self._lambda_global.item():.4f})"
        )
