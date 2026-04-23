class TitansMALConfig:
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=6,
        window_size=128,
        num_persist_mem_tokens=16,
        neural_memory_segment_len=None,
        neural_memory_batch_size=None,
        num_attention_heads=8,
        dim_head=64,
        intermediate_size=2048,
        num_residual_streams=4,
        neural_memory_layers=None,
        use_flex_attn=False,
        neural_mem_weight_residual=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.neural_memory_segment_len = neural_memory_segment_len or window_size
        self.neural_memory_batch_size = neural_memory_batch_size
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.intermediate_size = intermediate_size
        self.num_residual_streams = num_residual_streams
        self.neural_memory_layers = neural_memory_layers if neural_memory_layers is not None else []
        self.use_flex_attn = use_flex_attn
        self.neural_mem_weight_residual = neural_mem_weight_residual
        for key, value in kwargs.items():
            setattr(self, key, value)
