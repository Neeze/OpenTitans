import torch.optim as optim

class TestTimeOptimizer(optim.Optimizer):
    """Optimizer for background test-time memorization (ATLAS)."""
    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
