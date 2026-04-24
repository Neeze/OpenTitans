from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def nested_train_step(
    model: torch.nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn=F.cross_entropy,
) -> float:
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
