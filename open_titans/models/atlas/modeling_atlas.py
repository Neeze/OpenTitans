from ..modeling_utils import PreTrainedModel

class AtlasModel(PreTrainedModel):
    """Model architecture with Test-Time Memorization."""
    def __init__(self, config):
        super().__init__(config)
        pass

    def forward(self, input_ids, attention_mask=None):
        pass
