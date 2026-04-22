from dataclasses import dataclass

@dataclass
class TrainingArguments:
    """Special parameters for Memory Models."""
    learning_rate: float = 5e-5
    batch_size: int = 8
    bptt_steps: int = 128
