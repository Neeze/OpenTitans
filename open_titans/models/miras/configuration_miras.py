from __future__ import annotations


class MirasConfig:
    def __init__(
        self,
        variant: str = "yaad",
        vocab_size: int = 50257,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        dim_head: int = 64,
        intermediate_size: int = 512,
        chunk_size: int = 8,
        mem_heads: int = 2,
        momentum: bool = True,
        momentum_order: int = 1,
        pre_rmsnorm: bool = True,
        huber_delta: float = 1.0,
        lp_norm: float = 3.0,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        assert variant in ("yaad", "moneta", "memora"), (
            f"Unknown variant: {variant}. Must be one of: yaad, moneta, memora"
        )

        self.variant = variant
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.intermediate_size = intermediate_size
        self.chunk_size = chunk_size
        self.mem_heads = mem_heads
        self.momentum = momentum
        self.momentum_order = momentum_order
        self.pre_rmsnorm = pre_rmsnorm
        self.huber_delta = huber_delta
        self.lp_norm = lp_norm
        self.max_seq_len = max_seq_len

        for key, value in kwargs.items():
            setattr(self, key, value)
