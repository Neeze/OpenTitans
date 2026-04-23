# 🧠 Neural Memory & Custom Models

The **Neural Memory** module is the heart of the OpenTitans framework. It implements the "fast-weight" update mechanism where the model learns to store and retrieve information at test time.

## 🛠️ How it Works

Unlike standard attention which uses a KV-cache, Neural Memory uses a small neural network (an MLP) whose weights are updated during the forward pass. 
- **Storage**: When the model sees new data, it performs a small number of gradient descent steps on the neural memory's weights to "memorize" the new information.
- **Retrieval**: To recall information, the model passes a query through this same neural network.

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

# Usage:
from open_titans.models.titans_mac import TitansMACModel, TitansMACConfig

config = TitansMACConfig(hidden_size=256, dim_head=64)
custom_mem = DeepMemoryMLP(dim=64) # Use dim_head

model = TitansMACModel(config, neural_memory_model=custom_mem)
```

## ⚙️ Advanced Configuration

When instantiating a Titans model, you can pass additional kwargs that will be forwarded to the `NeuralMemory` module:

```python
model = TitansMACModel(
    config,
    neural_memory_model=custom_mem,
    momentum=True,               # Use momentum for weight updates
    momentum_order=2,            # Second-order momentum
    learned_momentum_combine=True # Learn how to combine momentum terms
)
```

### Key Parameters:
- `momentum`: Enables momentum-based updates for the neural memory weights.
- `adaptive_step_transform`: A function to transform the predicted learning rate for updates.
- `max_grad_norm`: Clips the gradients during the test-time update to ensure stability.
