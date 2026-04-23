from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UpdateRuleType(Enum):
    LINEAR = "linear"
    YAAD = "yaad"
    MEMORA = "memora"
    MONETA = "moneta"


def linear_update(
    w_prev: Tensor,
    grad: Tensor,
    alpha: Tensor,
    eta: Tensor,
) -> Tensor:
    return alpha * w_prev - eta * grad


def yaad_update(
    w_prev: Tensor,
    grad: Tensor,
    alpha: Tensor,
    eta: Tensor,
    error_norm: Tensor,
    delta: float = 1.0,
) -> Tensor:
    scale = torch.where(
        error_norm <= delta,
        torch.ones_like(error_norm),
        delta / error_norm.clamp(min=1e-8),
    )
    while scale.ndim < grad.ndim:
        scale = scale.unsqueeze(-1)
    return alpha * w_prev - eta * scale * grad


def memora_update(
    w_prev: Tensor,
    grad: Tensor,
    alpha: Tensor,
    eta: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    log_w = (w_prev.clamp(min=eps)).log()
    logits = alpha * log_w - eta * grad
    flat = logits.flatten(-2)
    normed = F.softmax(flat, dim=-1)
    return normed.view_as(w_prev)


def moneta_update(
    w_prev: Tensor,
    grad: Tensor,
    alpha: Tensor,
    eta: Tensor,
    l1_strength: float = 0.01,
) -> Tensor:
    raw = alpha * w_prev - eta * grad
    return torch.sign(raw) * F.relu(raw.abs() - l1_strength)


UPDATE_REGISTRY = {
    UpdateRuleType.LINEAR: linear_update,
    UpdateRuleType.YAAD: yaad_update,
    UpdateRuleType.MEMORA: memora_update,
    UpdateRuleType.MONETA: moneta_update,
}


class MemoryUpdateRule(nn.Module):
    def __init__(
        self,
        dim: int,
        rule_type: str | UpdateRuleType = UpdateRuleType.LINEAR,
        learnable_eta: bool = True,
        learnable_alpha: bool = True,
        init_eta: float = 0.01,
        init_alpha: float = 0.9,
        **rule_kwargs,
    ):
        super().__init__()

        if isinstance(rule_type, str):
            rule_type = UpdateRuleType(rule_type)

        self.rule_type = rule_type
        self.rule_kwargs = rule_kwargs

        if learnable_eta:
            self._eta_logit = nn.Parameter(torch.tensor(_inv_sigmoid(init_eta)))
        else:
            self.register_buffer("_eta_logit", torch.tensor(_inv_sigmoid(init_eta)))

        if learnable_alpha:
            self._alpha_logit = nn.Parameter(torch.tensor(_inv_sigmoid(init_alpha)))
        else:
            self.register_buffer("_alpha_logit", torch.tensor(_inv_sigmoid(init_alpha)))

    @property
    def eta(self) -> Tensor:
        return self._eta_logit.sigmoid()

    @property
    def alpha(self) -> Tensor:
        return self._alpha_logit.sigmoid()

    def forward(
        self,
        w_prev: Tensor,
        grad: Tensor,
        error_norm: Tensor | None = None,
    ) -> Tensor:
        eta = self.eta
        alpha = self.alpha

        while eta.ndim < grad.ndim:
            eta = eta.unsqueeze(-1)
        while alpha.ndim < grad.ndim:
            alpha = alpha.unsqueeze(-1)

        if self.rule_type == UpdateRuleType.YAAD:
            if error_norm is None:
                raise ValueError("YAAD update requires error_norm")
            return yaad_update(w_prev, grad, alpha, eta, error_norm, **self.rule_kwargs)
        elif self.rule_type == UpdateRuleType.MEMORA:
            return memora_update(w_prev, grad, alpha, eta, **self.rule_kwargs)
        elif self.rule_type == UpdateRuleType.MONETA:
            return moneta_update(w_prev, grad, alpha, eta, **self.rule_kwargs)
        else:
            return linear_update(w_prev, grad, alpha, eta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rule={self.rule_type.value}, "
            f"η={self.eta.item():.4f}, "
            f"α={self.alpha.item():.4f})"
        )


def _inv_sigmoid(x: float, eps: float = 1e-6) -> float:
    x = max(min(x, 1.0 - eps), eps)
    return torch.tensor(x / (1.0 - x)).log().item()
