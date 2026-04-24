from typing import Any

from .configuration_atlas import TitansAtlasConfig
from .modeling_atlas import AtlasModel


ATLAS_REGISTRY = {
    "deep_transformers": {
        "description": "Pure ATLAS Architecture (DeepTransformers). Replaces self-attention with active memory.",
        "defaults": dict(
            variant="deep_transformers",
        ),
    },
    "mag": {
        "description": "Memory as a Gate (MAG). Parallel hybrid processing long-term and short-term context.",
        "defaults": dict(
            variant="mag",
        ),
    },
    "mal": {
        "description": "Memory as a Layer (MAL). Sequential hybrid pipeline (ATLAS Layer -> SWA).",
        "defaults": dict(
            variant="mal",
        ),
    },
}


def create_atlas_model(
    variant: str,
    **overrides: Any,
) -> AtlasModel:
    """
    Factory function to create an ATLAS model variant.
    """
    if variant not in ATLAS_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(ATLAS_REGISTRY.keys())}")

    entry = ATLAS_REGISTRY[variant]
    defaults = entry["defaults"].copy()
    defaults.update(overrides)
    
    config = TitansAtlasConfig(**defaults)
    
    return AtlasModel(config)


def list_variants() -> dict[str, str]:
    """
    List available ATLAS variants.
    """
    return {name: info["description"] for name, info in ATLAS_REGISTRY.items()}
