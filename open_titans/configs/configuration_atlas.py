from .configuration_utils import PretrainedConfig

class AtlasConfig(PretrainedConfig):
    """Configuration for ATLAS (Test-time memory)."""
    def __init__(
        self,
        memory_warmup_steps=100,
        test_time_learning_rate=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory_warmup_steps = memory_warmup_steps
        self.test_time_learning_rate = test_time_learning_rate
