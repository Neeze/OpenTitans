# Quickstart

Get up and running with OpenTitans in just a few lines of code. This guide shows how to initialize a model and perform a basic forward pass.

## Basic Usage Example

The following example demonstrates how to use the `TitansMACModel` (Memory as Context).

```python
import torch
import torch.nn as nn
from open_titans.models.titans_mac import TitansMACModel, TitansMACConfig

# 1. Define the configuration
config = TitansMACConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=2,
    segment_len=32,
    num_attention_heads=4,
    dim_head=64,
    intermediate_size=512,
    neural_memory_layers=[2], # Layer indices to apply neural memory
    num_longterm_mem_tokens=8,
)

# 2. (Optional) Define a custom neural memory model
# This model will be used inside the memory modules for weight updates
neural_memory_model = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),
    nn.Linear(512, 256)
)

# 3. Instantiate the model
model = TitansMACModel(config, neural_memory_model=neural_memory_model)

# 4. Prepare input data
batch_size = 2
seq_len = 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# 5. Perform a forward pass
outputs = model(input_ids)

# The output is a Tensor of shape (batch_size, seq_len, hidden_size)
print(f"Output shape: {outputs.shape}")
```

## Moving to GPU

OpenTitans is optimized for CUDA. To run on a GPU, simply move the model and tensors:

```python
if torch.cuda.is_available():
    model = model.cuda()
    input_ids = input_ids.cuda()
    outputs = model(input_ids)
    print("Inference completed on GPU.")
```

## Next Steps

-   Learn how to **[Define Custom Memory Models](./Neural_Memory.md)**.
-   Explore different **Titans Variants** in the [Titans Variants](./Titans_Variants.md) guide.
-   Check out the `tests/` directory for more advanced usage examples and benchmarking scripts.
