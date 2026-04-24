from typing import Optional


class TitansAtlasConfig:
    """
    Configuration for ATLAS architectures (DeepTransformers, MAG, MAL).
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_len: int = 2048,
        chunk_size: int = 64,             # C: For Chunk-wise Parallel Test-Time Update
        retrospective_window: int = 256,  # c: Sliding window for Omega Rule
        muon_lr: float = 0.02,            # eta: Learning rate for Muon Test-Time Optimizer
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        variant: str = "deep_transformers",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        
        self.chunk_size = chunk_size
        self.retrospective_window = retrospective_window
        
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_ns_steps = muon_ns_steps
        
        self.variant = variant
        
        for k, v in kwargs.items():
            setattr(self, k, v)
