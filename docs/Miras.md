# MIRAS: Online Optimization Framework

**MIRAS** (Memory via Internal Regularized Attentional Surprise) is a unifying framework that recasts sequence modeling as a continuous **Online Optimization** problem. Instead of using static weight updates or heuristic gates, MIRAS treats the forward pass of a model as an active learning loop.

## Core Principle: Learning-Retaining

In the MIRAS framework, the memory state $W_t$ is updated by solving an optimization problem at every time step:

$$W_t = \arg \min_W \left[ \ell(W; k_t, v_t) + \text{Ret}_t(W, W_{t-1}) \right]$$

1.  **Attentional Bias ($\ell$):** An internal loss function that pushes the memory to map keys $k_t$ to values $v_t$ (Learning).
2.  **Retention Regularizer ($\text{Ret}$):** A penalty function that prevents the memory from deviating too far from its previous state (Retaining).

## Miras Variants

MIRAS provides three specialized model architectures, each derived from different optimization objectives:

### 1. YAAD (Robust Memory)
Designed for sequences with noise or outliers.
-   **Bias**: **Huber Loss** (Hybrid $L_2$ and $L_1$ penalty).
-   **Regularizer**: **Bregman Divergence** (Geometry-aware retention).
-   **Effect**: Small errors are updated quadratically, while large "surprising" errors are capped linearly, preventing memory corruption from outliers.

### 2. MONETA (Sparse Memory)
Designed for extreme context compression.
-   **Bias**: **Generalized Norms** ($L_p$).
-   **Regularizer**: **Elastic Net** ($L_1 + L_2$).
-   **Effect**: The $L_1$ penalty induces **sparsity** via soft-thresholding, forcing unimportant memory weights exactly to zero.

### 3. MEMORA (Stable Memory)
Designed for ultra-long context (1M+ tokens).
-   **Bias**: **KL-Divergence**.
-   **Regularizer**: **f-Divergence** on a probability simplex.
-   **Effect**: Mathematically constrains memory weights to stay normalized, preventing gradient explosion or state saturation across infinite horizons.

## Usage

You can easily instantiate any MIRAS variant using the provided registry.

```python
import torch
from open_titans.models.miras import create_miras_model

# Create a YAAD model
model = create_miras_model("yaad", hidden_size=512)

# Run forward pass
input_ids = torch.randint(0, 50257, (1, 128))
output = model(input_ids)
print(output.logits.shape) # (1, 128, 50257)
```

### Customizing the Memory Model

Just like other Titans models in the framework, you can pass a custom neural network to act as the associative memory.

```python
import torch.nn as nn
from open_titans.models.miras import create_miras_model

custom_mlp = nn.Sequential(
    nn.Linear(64, 128),
    nn.GELU(),
    nn.Linear(128, 64)
)

model = create_miras_model("moneta", neural_memory_model=custom_mlp)
```

## Components

If you are a researcher, you can use the modular components of MIRAS independently:

-   **Attentional Bias**: `open_titans.modules.attention.AttentionalBias`
-   **Retention Regularization**: `open_titans.modules.gates.RetentionRegularization`
-   **Update Rules**: `open_titans.modules.memory.MemoryUpdateRule`
