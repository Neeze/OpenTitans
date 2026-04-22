from .configuration_utils import PretrainedConfig

class TitansConfig(PretrainedConfig):
    """Configuration for Titans core architecture."""
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        memory_size=1024,
        chunk_size=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.memory_size = memory_size
        self.chunk_size = chunk_size
