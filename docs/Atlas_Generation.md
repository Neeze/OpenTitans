# ATLAS Generation

ATLAS generation is more complex than standard Titans generation because the inner optimizer requires gradient computation at inference time. This document covers the `AtlasGenerationMixin` and how it handles **Test-Time Training (TTT)** during auto-regressive decoding.

## The Challenge

Standard generation runs under `@torch.no_grad()`. However, the ATLAS update rule computes the gradient of a local prediction loss to update the memory:

$$\nabla_{W_t} \mathcal{L}_{chunk} = \nabla_{W_t} \| W_t \cdot k_t - v_t \|^2$$

This gradient is then orthogonalized via Newton-Schulz iteration (Muon) to produce the memory update:

$$W_{t+1} = W_t \cdot \alpha - \text{NS}_5(\nabla_{W_t} \mathcal{L}_{chunk}) \cdot \eta$$

The outer model parameters (embeddings, projections, attention weights) must remain frozen. Only the memory matrix $W_t$ receives updates.

## `AtlasGenerationMixin`

`AtlasGenerationMixin` extends `TitansGenerationMixin` with two key differences:

1. **Gradient context control**: The `enable_ttt_grad` parameter wraps forward passes in `torch.enable_grad()` instead of `torch.no_grad()`.
2. **Strict gradient isolation**: After each step, `cache.detach()` ensures the inner gradient does not leak into the outer parameters.

## Usage

### Standard Generation (No TTT)

```python
import torch
from open_titans.models.atlas.configuration_atlas import TitansAtlasConfig
from open_titans.models.atlas.modeling_atlas import AtlasModel

config = TitansAtlasConfig(
    vocab_size=50257,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_seq_len=2048,
    chunk_size=64,
    retrospective_window=128,
    muon_ns_steps=5,
    variant="deep_transformers",
)

model = AtlasModel(config)
model.eval()

prompt = torch.randint(0, 50257, (1, 64))

output = model.generate(
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
)

print(output.shape)  # (1, 192)
```

### Generation with Test-Time Training

Enable gradient computation for the inner optimizer during generation:

```python
output = model.generate(
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    enable_ttt_grad=True,
)
```

With `enable_ttt_grad=True`:
- The inner optimizer computes $\nabla \mathcal{L}_{in}$ normally
- Memory updates are computed with full gradient support
- `cache.detach()` prevents gradients from propagating to outer parameters
- No outer parameter accumulates `.grad`

## How Gradient Isolation Works

The generation loop uses two mechanisms to isolate gradients:

### 1. Context Manager Switching

```
enable_ttt_grad=False:  @torch.no_grad() → forward() → no gradients anywhere
enable_ttt_grad=True:   @torch.enable_grad() → forward() → gradients flow inside NeuralMemory
```

### 2. State Detaching

After every generation step:

$$\mathcal{S}_{t+1} \leftarrow \text{detach}(\mathcal{S}_{t+1})$$

This ensures:
- $W_{t+1}$ becomes a leaf tensor (no `grad_fn`)
- The computation graph does not grow across steps
- Outer parameters never accumulate gradients

## ATLAS Cache

`AtlasCache` wraps per-layer state tuples:

```python
from open_titans.generation import AtlasCache

# Each layer's state is (mem_state, buffer_k, buffer_v)
cache = AtlasCache.from_layer_states(
    [(mem_state, buffer_k, buffer_v) for _ in range(num_layers)],
    seen_tokens=64,
)
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `mem_state` | `Tensor (B, D, D)` | The memory matrix $W_t$ |
| `buffer_k` | `Tensor (B, window, D)` | Retrospective key buffer |
| `buffer_v` | `Tensor (B, window, D)` | Retrospective value buffer |

## ATLAS Variant Behavior During Generation

| Variant | Layers | Generation Behavior |
| :--- | :--- | :--- |
| **DeepTransformers** | ATLAS → SwiGLU | Pure memory-based generation, no attention overhead |
| **MAG** | (ATLAS \|\| SWA) → Fusion → SwiGLU | Memory gate modulates sliding-window attention output |
| **MAL** | ATLAS → SWA → SwiGLU | Memory pre-processes before local attention decode |

## API Reference

### `AtlasGenerationMixin.generate()`

```python
def generate(
    input_ids: Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    attention_mask: Tensor | None = None,
    enable_ttt_grad: bool = False,
) -> Tensor
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `input_ids` | — | Prompt token IDs `(B, T)` |
| `max_new_tokens` | `128` | Number of tokens to generate |
| `temperature` | `1.0` | Sampling temperature. `0.0` = greedy |
| `top_k` | `0` | Top-K filtering. `0` = disabled |
| `top_p` | `1.0` | Nucleus sampling threshold. `1.0` = disabled |
| `eos_token_id` | `None` | Stop generation when this token is produced |
| `pad_token_id` | `None` | Fill remaining positions after EOS |
| `attention_mask` | `None` | Binary mask for padded inputs |
| `enable_ttt_grad` | `False` | Enable gradient computation for inner optimizer |

**Returns**: `Tensor` of shape `(B, T + max_new_tokens)` containing the prompt followed by generated tokens.
