import torch

def track_memory():
    """Monitor VRAM usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0
