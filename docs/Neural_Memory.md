# 🧠 Neural Memory & Custom Models

The **Neural Memory** module is the heart of the OpenTitans framework. It implements the "fast-weight" update mechanism where the model learns to store and retrieve information at test time.

## 🛠️ How it Works

Unlike standard attention which uses a KV-cache, Neural Memory uses a small neural network whose weights are updated during the forward pass. 
- **Storage**: When the model sees new data, it performs a small number of gradient descent steps on the neural memory's weights to "memorize" the new information.
- **Retrieval**: To recall information, the model passes a query through this same neural network.

## 🏛️ Available Memory Models

OpenTitans provides several built-in architectures for the neural memory:

| Model | Description |
| :--- | :--- |
| `MemoryMLP` | A standard multi-layer perceptron using manual matrix multiplications for efficient updates. |
| `GatedResidualMemoryMLP` | An MLP with gated residual connections and final projection for increased capacity. |
| `FactorizedMemoryMLP` | Uses factorized weights to reduce the number of parameters per chunk. |
| `MemorySwiGluMLP` | An MLP variant using the SwiGLU activation function, common in modern LLMs. |
| `MemoryAttention` | Implements an internal attention mechanism within the memory module itself. |

### 🛠️ Wrappers

- `ResidualNorm`: A wrapper that adds a residual connection and `LayerNorm` around any memory model.
- `LayerNorm`: A custom normalization layer designed specifically for neural memory updates.

## ⚙️ Configuration

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

## 🎨 Defining a Custom Memory Model

You can provide your own architecture for the neural memory. This allows you to experiment with different capacities and update dynamics.

### 📋 Requirements

To be compatible with `NeuralMemory`, your custom model must:
1.  Be a `torch.nn.Module`.
2.  **Not** contain any `buffers` (only `parameters` are allowed).
3.  Accept an input tensor of shape `(batch, seq, dim_head)`.
4.  Return an output tensor of shape `(batch, seq, dim_head)`.

### 💡 Example: Deep MLP Memory

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

## 🚀 Advanced Hyperparameters

Additional parameters available in `neural_memory_kwargs`:

- `momentum`: enables momentum-based updates (default: `True`).
- `momentum_order`: order of momentum (default: `1`).
- `max_grad_norm`: clips gradients during updates to ensure stability.
- `per_parameter_lr_modulation`: allows the model to learn a per-parameter learning rate.
- `chunk_size`: controls the frequency of weight updates.
