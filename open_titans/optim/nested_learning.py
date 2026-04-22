import torch.optim as optim

class NestedLearningOptimizer(optim.Optimizer):
    """Implementation of Nested Learning (NL) algorithm."""
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
