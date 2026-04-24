from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import zip_longest
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from assoc_scan import AssocScan
from open_titans.modules.memory.functional import exists, Sequential, newtonschulz5


class UpdateRuleType(Enum):
    MOMENTUM = "momentum"
    LINEAR = "linear"
    YAAD = "yaad"
    MEMORA = "memora"
    MONETA = "moneta"
    EXPRESSIVE = "expressive"


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
        W_t: Tensor,
        x_t: Tensor,
        grad_l_in: Tensor,
        **kwargs,
    ) -> Tensor:
        w_prev = W_t
        grad = grad_l_in
        error_norm = kwargs.get("error_norm", None)

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


def sherman_morrison_step(
    W_t: Tensor,
    x_t: Tensor,
    grad_l_in: Tensor,
    eta_t: Tensor,
    lambda_sq: Tensor,
) -> Tensor:
    scale = 1.0 / (lambda_sq + eta_t)
    if grad_l_in.ndim == 3: # Parameter gradient
        x_x_t = torch.bmm(x_t.unsqueeze(-1), x_t.unsqueeze(1))
        update = torch.bmm(W_t, x_x_t) - grad_l_in
    else: # Output gradient
        y_hat = torch.bmm(W_t, x_t.unsqueeze(-1)).squeeze(-1)
        error = y_hat - grad_l_in
        update = torch.bmm(error.unsqueeze(-1), x_t.unsqueeze(1))
    return W_t - scale.view(-1, 1, 1) * update


class ExpressiveUpdateRule(nn.Module):
    def __init__(
        self,
        dim_in: int,
        init_eta: float = 1e-3,
    ):
        super().__init__()
        self.eta_proj = nn.Linear(dim_in, 1, bias=True)
        nn.init.xavier_uniform_(self.eta_proj.weight)
        nn.init.constant_(self.eta_proj.bias, init_eta)
        self.register_buffer("lambda_sq", torch.tensor([float(dim_in)]))

    def forward(
        self,
        W_t: Tensor,
        x_t: Tensor,
        grad_l_in: Tensor,
        **kwargs,
    ) -> Tensor:
        eta_t = F.softplus(self.eta_proj(x_t))
        if W_t.ndim == 2:
            # Fallback for biases: simple scaled gradient descent
            scale = 1.0 / (self.lambda_sq + eta_t)
            return W_t - scale.view(-1, 1) * grad_l_in
        return sherman_morrison_step(W_t, x_t, grad_l_in, eta_t, self.lambda_sq)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dim_in={self.lambda_sq.item():.0f}, "
            f"init_bias={self.eta_proj.bias.item():.4f})"
        )

class MomentumUpdateRule(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        momentum: bool = True,
        momentum_order: int = 1,
        learned_momentum_combine: bool = False,
        learned_combine_include_zeroth: bool = False,
        use_accelerated_scan: bool = False,
        spectral_norm_surprises: bool = False,
        init_momentum_bias: float | None = None,
        init_decay_bias: float | None = None,
    ):
        super().__init__()
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        self.momentum_order = momentum_order
        self.learned_combine_include_zeroth = learned_combine_include_zeroth
        self.has_momentum = momentum
        self.learned_combine = learned_momentum_combine
        self.spectral_norm_surprises = spectral_norm_surprises

        self.to_momentum = (
            Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange("b n (h o) -> o (b h) n 1", o=momentum_order),
            )
            if momentum
            else None
        )
        
        self.to_learned_momentum_combine = None
        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1
            effective_order = momentum_order + 1 if learned_combine_include_zeroth else momentum_order
            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * effective_order),
                Rearrange("b n (h o) -> o (b h) n", h=heads),
                nn.Softmax(dim=0),
            )

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange("b n h -> (b h) n 1"),
        )
        
        if exists(init_momentum_bias) and self.to_momentum is not None:
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

    def precompute(self, chunked_seq):
        adaptive_momentum = self.to_momentum(chunked_seq).sigmoid() if self.has_momentum else None
        combine_momentums = self.to_learned_momentum_combine(chunked_seq) if self.learned_combine else None
        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()
        return adaptive_momentum, combine_momentums, decay_factor

    def forward(self, W_t, x_t, grad_l_in, last_momentum=None, adaptive_momentum=None, combine_momentums=None, decay_factor=None):
        surprise = grad_l_in
        update = surprise
        next_last_momentum = None

        if self.has_momentum:
            momentum = surprise
            momentums = []
            
            for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev=one_last_momentum)
                momentums.append(momentum)

            momentums = torch.stack(momentums)
            next_last_momentum = momentums[:, :, -1]

            if self.learned_combine and self.learned_combine_include_zeroth:
                momentums = torch.cat((rearrange(surprise, "... -> 1 ..."), momentums), dim=0)

            if not self.learned_combine:
                update = momentums[-1]
            else:
                update = einsum(combine_momentums, momentums, "o b n, o b n ... -> b n ...")

        if self.spectral_norm_surprises:
            update = newtonschulz5(update)

        update = self.assoc_scan(1.0 - decay_factor, update, prev=W_t, remove_prev=False)
        return update, next_last_momentum
