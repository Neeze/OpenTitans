import torch
import torch.nn as nn

class PreTrainedModel(nn.Module):
    """Base class for all models."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, path, config):
        model = cls(config)
        # Placeholder for weight loading logic
        return model

    def save_pretrained(self, path):
        # Placeholder for weight saving logic
        pass
