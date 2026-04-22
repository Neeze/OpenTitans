from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F


class MemoryMLP(Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        expansion_factor: float = 4.0,
    ):
        super().__init__()

        hidden_dim = int(dim * expansion_factor)

        layers: list[Module] = []
        in_dim = dim

        for i in range(depth):
            out_dim = hidden_dim if i < depth - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < depth - 1:
                layers.append(nn.GELU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualNorm(Module):
    def __init__(self, dim: int, model: Module):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return x + self.norm(self.model(x))
