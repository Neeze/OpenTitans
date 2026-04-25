# ATLAS: Learning to Optimally Memorize the Context at Test Time

**ATLAS** is a neural architecture framework designed to optimally memorize and retrieve long-term context by treating memory updates as a highly efficient **Test-Time Optimization** process. Unlike traditional linear attention or static RNNs, ATLAS computes gradients on the fly to "learn" the local context into its persistent memory weights.

## Core Principle: The Omega Rule

ATLAS operates on a chunk-wise parallel update mechanism using the **Omega Rule**. Instead of simple additive updates, ATLAS uses second-order gradient information to minimize a local prediction error:

$$\mathcal{L}_{chunk} = \| \text{Mem}(k) - v \|^2$$

1.  **Retrospective Memory Buffer**: Maintains a sliding window of recent tokens to compute accurate error signals.
2.  **Newton-Schulz Orthogonalization**: Uses a 5-step Newton-Schulz iteration (via the `Muon` optimizer) to approximate the Hessian inverse, providing stable second-order updates without the $O(d^3)$ cost of full matrix inversion.

## ATLAS Variants

ATLAS provides three distinct architectural topologies to balance long-term memorization with local syntactic processing:

### 1. DeepTransformers (Pure ATLAS)
A "transformer-less" architecture that replaces standard self-attention entirely with ATLAS memory layers.
-   **Best for**: Maximum context compression and ultra-long dependencies.
-   **Structure**: `ATLAS Layer -> SwiGLU`.

### 2. MAG (Memory as a Gate)
A parallel hybrid architecture that processes the sequence through two concurrent branches.
-   **Best for**: Balanced performance where local structure and historical retrieval are equally critical.
-   **Structure**: `(ATLAS Layer || SlidingWindowAttention) -> Gated Fusion -> SwiGLU`.

### 3. MAL (Memory as a Layer)
A sequential pipeline where the ATLAS layer acts as a long-term "pre-processor" for a local attention layer.
-   **Best for**: Tasks requiring deep reasoning over retrieved context.
-   **Structure**: `ATLAS Layer -> SlidingWindowAttention -> SwiGLU`.

## Usage

Instantiate an ATLAS model using the unified registry:

```python
import torch
from open_titans.models.atlas import create_atlas_model

# Create an ATLAS-MAG variant
model = create_atlas_model(
    variant="mag",
    hidden_size=512,
    vocab_size=50257,
    num_hidden_layers=12
)

# Forward pass with attention mask support
input_ids = torch.randint(0, 50257, (1, 128))
mask = torch.ones((1, 128), dtype=torch.bool)

output = model(input_ids, attention_mask=mask)
print(output.logits.shape) # (1, 128, 50257)
```

## Components

ATLAS is built from highly modular components available for independent use:

-   **Memory Buffer**: `open_titans.modules.memory.RetrospectiveMemoryBuffer`
-   **Optimizer**: `open_titans.optim.Muon`
-   **Core Layer**: `open_titans.models.atlas.modeling_atlas.AtlasLayerBlock`
-   **Newton-Schulz**: `open_titans.optim.muon.newton_schulz5`

## Comparison with Standard Transformers

| Feature | Transformer | ATLAS |
| :--- | :--- | :--- |
| **KV Cache** | $O(N)$ (Grows with length) | $O(1)$ (Fixed state size) |
| **Updates** | None (Static weights) | Dynamic (Test-time gradients) |
| **Inference Cost** | Quadratic $O(N^2)$ | Linear $O(N)$ |
| **Complexity** | Memory-bound | Computation-bound |
