import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for approximate second-order (Hessian) dynamics.
    Computes an approximate orthonormalization of the gradient matrix.
    """
    if G.ndim < 2:
        return G
    
    # Store original shape and transpose if needed to make rows <= cols
    original_shape = G.shape
    should_transpose = original_shape[-2] > original_shape[-1]
    
    if should_transpose:
        G = G.transpose(-1, -2)
        
    # Flatten extra dimensions if any
    G = G.view(-1, G.size(-2), G.size(-1))
    
    # Normalize
    norm = G.norm(dim=(-1, -2), keepdim=True).clamp(min=eps)
    X = G / norm
    
    # Newton-Schulz coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if should_transpose:
        X = X.transpose(-1, -2)
        
    return X.view(original_shape)


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonalized) Optimizer for ATLAS Gradient Engine.
    Employs approximate second-order (Hessian) optimization dynamics using 
    Newton-Schulz iterations. This optimally packs information into the memory 
    matrix much faster and more stably than standard SGD.
    """
    def __init__(
        self, 
        params, 
        lr: float = 0.02, 
        momentum: float = 0.95, 
        nesterov: bool = True,
        ns_steps: int = 5
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state['momentum_buffer']

                # Apply Newton-Schulz 5 iteration to approximate Hessian projection
                orthogonal_grad = newton_schulz5(grad, steps=ns_steps)

                # Momentum update
                buf.mul_(momentum).add_(orthogonal_grad)

                if nesterov:
                    update = orthogonal_grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Parameter update
                p.add_(update, alpha=-lr)

        return loss
