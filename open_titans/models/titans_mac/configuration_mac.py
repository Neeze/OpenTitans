class TitansMACConfig:
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=6,
        segment_len=128,
        neural_memory_segment_len=None,
        neural_mem_gate_attn_output=False,
        neural_memory_add_value_residual=False,
        num_longterm_mem_tokens=0,
        num_persist_mem_tokens=0,
        neural_memory_batch_size=None,
        neural_memory_qkv_receives_diff_views=False,
        num_attention_heads=8,
        dim_head=64,
        intermediate_size=2048,
        num_residual_streams=4,
        neural_memory_layers=None,
        use_flex_attn=False,
        sliding_window_attn=False,
        neural_mem_weight_residual=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.segment_len = segment_len
        self.neural_memory_segment_len = neural_memory_segment_len
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output
        self.neural_memory_add_value_residual = neural_memory_add_value_residual
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.neural_memory_batch_size = neural_memory_batch_size
        self.neural_memory_qkv_receives_diff_views = neural_memory_qkv_receives_diff_views
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.intermediate_size = intermediate_size
        self.num_residual_streams = num_residual_streams
        self.neural_memory_layers = neural_memory_layers if neural_memory_layers is not None else []
        self.use_flex_attn = use_flex_attn
        self.sliding_window_attn = sliding_window_attn
        self.neural_mem_weight_residual = neural_mem_weight_residual
        for key, value in kwargs.items():
            setattr(self, key, value)
