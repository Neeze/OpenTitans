# Neural Memory & Custom Models

The **Neural Memory** module is the heart of the OpenTitans framework. It implements the "fast-weight" update mechanism where the model learns to store and retrieve information at test time.

## How it Works

Unlike standard attention which uses a KV-cache, Neural Memory uses a small neural network whose weights are updated during the forward pass. 
- **Storage**: When the model sees new data, it performs a small number of gradient descent steps on the neural memory's weights to "memorize" the new information.
- **Retrieval**: To recall information, the model passes a query through this same neural network.

## Available Memory Models

OpenTitans provides several built-in architectures for the neural memory:

| Model | Description |
| :--- | :--- |
| `MemoryMLP` | A standard multi-layer perceptron using manual matrix multiplications for efficient updates. |
| `GatedResidualMemoryMLP` | An MLP with gated residual connections and final projection for increased capacity. |
| `FactorizedMemoryMLP` | Uses factorized weights to reduce the number of parameters per chunk. |
| `MemorySwiGluMLP` | An MLP variant using the SwiGLU activation function, common in modern LLMs. |
| `MemoryAttention` | Implements an internal attention mechanism within the memory module itself. |

### Wrappers

- `ResidualNorm`: A wrapper that adds a residual connection and `LayerNorm` around any memory model.
- `LayerNorm`: A custom normalization layer designed specifically for neural memory updates.

## Configuration

You can configure the neural memory in two ways when initializing a Titans model:

### 1. Using Default MLP with Kwargs
By default, `MemoryMLP` is used. You can configure its depth and expansion factor via `neural_memory_kwargs`:

```python
model = TitansMACModel(
    config,
    neural_memory_kwargs={
        "default_model_kwargs": {
            "depth": 3,
            "expansion_factor": 4.0
        }
    }
)
```

### 2. Passing a Custom Model Instance
For more advanced architectures, instantiate the memory model and pass it as `neural_memory_model`:

```python
from open_titans.modules.memory import GatedResidualMemoryMLP

# Important: Use dim_head from your config
custom_mem = GatedResidualMemoryMLP(dim=64, depth=2, expansion_factor=4.0)

model = TitansMACModel(
    config,
    neural_memory_model=custom_mem
)
```

## Defining a Custom Memory Model

You can provide your own architecture for the neural memory. This allows you to experiment with different capacities and update dynamics.

### Requirements

To be compatible with `NeuralMemory`, your custom model must:
1.  Be a `torch.nn.Module`.
2.  **Not** contain any `buffers` (only `parameters` are allowed).
3.  Accept an input tensor of shape `(batch, seq, dim_head)`.
4.  Return an output tensor of shape `(batch, seq, dim_head)`.

### Example: Deep MLP Memory

```python
import torch.nn as nn

class DeepMemoryMLP(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)
```

## Memory Write Operation (Update Rules)

The mechanism used to update the neural memory weights during the forward pass (the "write" operation) is fully decoupled. OpenTitans provides different update rules that govern how gradients ("surprises") translate into parameter updates.

### Available Update Rules

| Rule | Description |
| :--- | :--- |
| `MomentumUpdateRule` | The default update rule from the Titans paper. Uses chunk-wise gradient descent with momentum and a forgetting (decay) mechanism. Supports fast intra-chunk parallelization via associative scans. |
| `ExpressiveUpdateRule` | Implements the Sherman-Morrison rank-1 $\mathcal{O}(d^2)$ update from the *Nested Learning* paper. Employs a self-referential gating mechanism to predict data-dependent learning rates per token. |
| `MemoryUpdateRule` | Specialized update rule designed specifically for the MIRAS model architecture and its variants (`YAAD`, `MEMORA`, `MONETA`). Provides learnable step size ($\eta$) and decay rate ($\alpha$). |

### Configuring the Update Rule

You can specify the update rule by passing it to `update_rule` via `neural_memory_kwargs`. If not specified, the model defaults to `MomentumUpdateRule` to preserve backward compatibility.

#### Example: Using Expressive Update Rule

To use the `ExpressiveUpdateRule` for an exact $\mathcal{O}(d^2)$ update, pass the instantiated rule. *Note: The `ExpressiveUpdateRule` is mathematically defined for a single weight matrix, so you should limit the memory MLP depth to 1.*

```python
from open_titans.modules.memory.update_rule import ExpressiveUpdateRule

# Important: dim_in should match your model's hidden dimension
expressive_rule = ExpressiveUpdateRule(dim_in=64)

model = TitansMACModel(
    config,
    neural_memory_kwargs={
        "update_rule": expressive_rule,
        "default_model_kwargs": {
            "depth": 1,
            "expansion_factor": 1.0
        }
    }
)
```

#### Example: Customizing Momentum Rule

You can explicitly pass a custom-configured `MomentumUpdateRule`:

```python
from open_titans.modules.memory.update_rule import MomentumUpdateRule

momentum_rule = MomentumUpdateRule(
    dim=64, 
    heads=4, 
    momentum_order=2, 
    learned_momentum_combine=True
)

model = TitansMACModel(
    config,
    neural_memory_kwargs={
        "update_rule": momentum_rule
    }
)
```

## Advanced Hyperparameters

Additional parameters available in `neural_memory_kwargs`:

- `momentum`: enables momentum-based updates for the default rule (default: `True`).
- `momentum_order`: order of momentum for the default rule (default: `1`).
- `max_grad_norm`: clips gradients during updates to ensure stability.
- `per_parameter_lr_modulation`: allows the model to learn a per-parameter learning rate.
- `chunk_size`: controls the frequency of chunk-wise parameter updates.
