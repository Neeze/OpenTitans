# Nested Learning (Self-Referential Titans)

**Nested Learning** (also known as Self-Referential Learning) is a paradigm introduced to dynamically learn memory during inference via **bi-level optimization**. The memory mechanism in OpenTitans supports this implicitly—allowing standard causal language modeling to trigger a meta-learning process under the hood.

---

## The Bi-Level Optimization Paradigm

The core idea of Nested Learning is that the neural memory acts as a dynamic parameter matrix $\mathbf{W}_t$ that is continuously optimized *during the forward pass*.

1. **Inner Optimizer (Fast Weights):** Inside the `NeuralMemory` module, the memory matrix $\mathbf{W}_t$ is updated token-by-token (or chunk-by-chunk) using the **Sherman-Morrison** rank-1 update. This rule perfectly minimizes an $L_2$ error function incrementally at exact $\mathcal{O}(d^2)$ complexity without backpropagation.
2. **Outer Optimizer (Meta-Parameters):** Standard optimization (like AdamW) updates the static "meta-parameters" of the model. Crucially, the model learns *how* to update its memory by learning the gating mechanism (`eta_proj`) that outputs a dynamic, data-dependent inner learning rate $\eta_t$ for each token.

---

## Native Implementation in OpenTitans

In OpenTitans, Nested Learning is achieved by injecting the `ExpressiveUpdateRule` into any Titans variant (MAC, MAG, or MAL). Because the `NeuralMemory` tracks gradients through the recurrent update sequence, calling standard PyTorch `.backward()` computes the gradients for the Outer Optimizer automatically!

### 1. Requirements

Because the exact Sherman-Morrison update relies on analytical properties of linear matrices, the inner memory model **must** be a single linear transformation (i.e., `nn.Linear` with no bias, not a multi-layer MLP).

### 2. Code Example

```python
import torch
import torch.nn as nn
from open_titans.models.titans_mac import TitansMACConfig, TitansMACModel
from open_titans.modules.memory.update_rule import ExpressiveUpdateRule

# 1. Standard Titans Architecture Configuration
config = TitansMACConfig(
    vocab_size=50257,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    segment_len=128,
    neural_memory_layers=[3, 4, 5], # Apply memory to specific layers
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Inject the Sherman-Morrison update rule
update_rule = ExpressiveUpdateRule(dim_in=config.hidden_size).to(device)

# 3. Restrict the inner memory model to a single linear matrix
memory_model = nn.Linear(config.hidden_size, config.hidden_size, bias=False).to(device)

# 4. Instantiate the native architecture
model = TitansMACModel(
    config, 
    update_rule=update_rule, 
    neural_memory_model=memory_model
).to(device)

# 5. Outer Optimizer for Meta-Learning
outer_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 6. Training Loop (Standard PyTorch)
inputs = torch.randint(0, 50257, (2, 256)).to(device)
labels = inputs.clone()

# The Forward Pass executes the Inner Optimizer implicitly!
outputs = model(inputs, return_loss=True, labels=labels)
loss = outputs.loss

# Backpropagation Through Time (BPTT) updates the meta-parameters
loss.backward()
outer_optimizer.step()
```

---

## The "Self-Referential" Magic

Inside the `ExpressiveUpdateRule`, a meta-parameter weight layer named `eta_proj` actively observes the input token and outputs a scalar $\eta_t$:

$$ \eta_t = \text{Softplus}( \mathbf{W}_\eta \cdot \mathbf{x}_t ) $$

By doing this, the outer model actively decides whether to aggressively overwrite memory ($\eta_t$ is large) or ignore noise ($\eta_t \approx 0$). This dynamic step size gives the memory its "self-referential" property.
