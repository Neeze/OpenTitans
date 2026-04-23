from __future__ import annotations

from .configuration_miras import MirasConfig
from .modeling_miras import MirasModel


MIRAS_REGISTRY = {
    "yaad": {
        "description": "Robust Memory — Huber loss + NeuralMemory with momentum",
        "defaults": dict(
            variant="yaad",
            huber_delta=1.0,
        ),
    },
    "moneta": {
        "description": "Sparse Memory — Lp norm + NeuralMemory with momentum",
        "defaults": dict(
            variant="moneta",
            lp_norm=3.0,
        ),
    },
    "memora": {
        "description": "Stable Memory — KL divergence + NeuralMemory with momentum",
        "defaults": dict(
            variant="memora",
        ),
    },
}


def create_miras_model(
    variant: str,
    neural_memory_model=None,
    **overrides,
) -> MirasModel:
    assert variant in MIRAS_REGISTRY, (
        f"Unknown variant '{variant}'. Available: {list(MIRAS_REGISTRY.keys())}"
    )

    defaults = MIRAS_REGISTRY[variant]["defaults"].copy()
    defaults.update(overrides)
    config = MirasConfig(**defaults)
    return MirasModel(config, neural_memory_model=neural_memory_model)


def list_variants() -> dict[str, str]:
    return {name: info["description"] for name, info in MIRAS_REGISTRY.items()}
