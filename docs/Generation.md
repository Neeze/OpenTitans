# Generation

Text generation in OpenTitans fundamentally differs from standard Transformers. Instead of appending tokens to a growing KV-Cache, the model continuously updates a fixed-size memory matrix $W_t$ step-by-step. During generation, the memory is not static — it reads the past and writes the new token simultaneously.

## How It Differs from Standard Transformers

| Property | Standard Transformer | OpenTitans |
| :--- | :--- | :--- |
| **State** | KV-Cache (grows $O(N)$) | Memory matrix $W_t$ (fixed $O(d^2)$) |
| **Update** | Concatenate new K, V | $W_{t+1} = \text{UpdateRule}(W_t, x_t, \nabla \mathcal{L}_{in})$ |
| **Inference** | $O(N^2)$ attention per step | $O(d^2)$ memory read + write per step |
| **Cache format** | Tuple of tensors | `TitansCache` / `AtlasCache` wrapper |

## The Generation Loop

The auto-regressive loop follows two distinct phases:

### Phase 1: Prefill (Parallel)

The entire prompt $(x_1, x_2, \dots, x_T)$ is processed in a single forward pass using the `assoc_scan` (Parallel Prefix Scan) to produce the initial state $\mathcal{S}_T$:

$$\mathcal{S}_T, y_T = \text{Model}(x_1, \dots, x_T)$$

### Phase 2: Decode (Sequential)

Each new token $x_t$ is generated one at a time. At each step $t$:

1. **Retrieve**: $\hat{y}_t = W_t \cdot x_t$
2. **Compute inner gradient**: $\nabla \mathcal{L}_{in} = \nabla_{W_t} \| W_t \cdot k_t - v_t \|^2$
3. **Update memory**: $W_{t+1} = \text{UpdateRule}(W_t, x_t, \nabla \mathcal{L}_{in})$
4. **Detach**: $W_{t+1} \leftarrow W_{t+1}.\text{detach}()$
5. **Sample**: $x_{t+1} \sim \text{Categorical}(\text{softmax}(y_t / T))$

Step 4 is critical — without it, the computation graph grows $O(T)$ and causes OOM within ~50 steps.

## Usage

### Titans Models (MAC, MAG, MAL)

All Titans models inherit `TitansGenerationMixin` and expose a `.generate()` method:

```python
import torch
from open_titans.models.titans_mac import TitansMACModel, TitansMACConfig

config = TitansMACConfig(
    vocab_size=50257,
    hidden_size=512,
    num_hidden_layers=6,
    segment_len=64,
    num_attention_heads=8,
    dim_head=64,
    intermediate_size=2048,
    neural_memory_layers=[3, 6],
    num_longterm_mem_tokens=0,
)

model = TitansMACModel(config)
model.eval()

prompt = torch.randint(0, 50257, (1, 32))

output = model.generate(
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    eos_token_id=50256,
    pad_token_id=0,
)

print(output.shape)  # (1, 160)
```

### Greedy Decoding

Set `temperature=0.0` for deterministic, greedy decoding:

```python
output = model.generate(prompt, max_new_tokens=64, temperature=0.0)
```

### Batched Generation

Generation supports batched inputs. Provide an `attention_mask` to handle padding:

```python
prompts = torch.randint(0, 50257, (4, 32))
mask = torch.ones_like(prompts)
mask[2, 20:] = 0  # sequence 2 is shorter

output = model.generate(prompts, max_new_tokens=64, attention_mask=mask)
print(output.shape)  # (4, 96)
```

## Decoding Strategies

The `generate()` method supports three decoding strategies, controlled by `temperature`, `top_k`, and `top_p`:

| Strategy | Parameters | Description |
| :--- | :--- | :--- |
| **Greedy** | `temperature=0.0` | Always pick $\arg\max(y_t)$ |
| **Top-K** | `top_k=50` | Keep only the $K$ highest-probability tokens |
| **Nucleus (Top-P)** | `top_p=0.9` | Keep the smallest set of tokens with cumulative probability $\geq p$ |

All strategies can be combined. The pipeline applies them in order: temperature scaling → Top-K filtering → Top-P filtering → sampling.

## Cache Architecture

### `TitansCache`

Used by MAC, MAG, and MAL models. Wraps a list of per-layer `NeuralMemState` objects:

```python
from open_titans.generation import TitansCache

cache = TitansCache.from_layer_states([state_layer_0, state_layer_1], seen_tokens=32)
cache = cache.detach()  # breaks computation graph
```

### `AtlasCache`

Used by ATLAS models. Wraps a list of per-layer `(mem_state, buffer_k, buffer_v)` tuples:

```python
from open_titans.generation import AtlasCache

cache = AtlasCache.from_layer_states([(mem, buf_k, buf_v)], seen_tokens=32)
cache = cache.detach()
```

Both caches expose the same interface:

| Method | Description |
| :--- | :--- |
| `.detach()` | Returns a new cache with all tensors detached from the computation graph |
| `.get_seq_length()` | Returns the number of tokens processed so far |
| `.update_seen_tokens(n)` | Increments the token counter |

## Critical Implementation Notes

### 1. The Memory Leak Trap (Graph Breaking)

When updating $W_t \to W_{t+1}$ using PyTorch operations, the computation graph grows with each step. Without `.detach()`, this causes OOM within ~50 tokens:

```
Step 1: W_1 = f(W_0)         → graph depth 1
Step 2: W_2 = f(W_1)         → graph depth 2
...
Step N: W_N = f(W_{N-1})     → graph depth N → OOM
```

The fix: always call `cache.detach()` after each step. This treats $W_{t+1}$ as a leaf tensor for the next step.

### 2. Padding Token Corruption

During batched generation, `<pad>` tokens must not update the memory. The `attention_mask` is propagated through to the `ExpressiveUpdateRule`, which sets `adaptive_lr=0` for masked positions:

$$W_{t+1} = \begin{cases} \text{UpdateRule}(W_t, x_t, \nabla) & \text{if mask}_t = 1 \\ W_t & \text{if mask}_t = 0 \end{cases}$$

### 3. Prompt Processing (Prefilling)

Processing a 1000-token prompt one-by-one is extremely slow. The prefill phase processes the entire prompt in parallel using `assoc_scan` to produce the initial $\mathcal{S}_T$, before switching to the sequential decode loop.

## API Reference

### `TitansGenerationMixin`

| Method | Signature |
| :--- | :--- |
| `generate()` | `(input_ids, max_new_tokens=128, temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None, pad_token_id=None, attention_mask=None) → Tensor` |
| `prepare_inputs_for_generation()` | `(input_ids, past_key_values=None, attention_mask=None) → dict` |

### Models with Generation Support

| Model | Mixin | Cache Type |
| :--- | :--- | :--- |
| `TitansMACModel` | `TitansGenerationMixin` | `TitansCache` |
| `TitansMAGModel` | `TitansGenerationMixin` | `TitansCache` |
| `TitansMALModel` | `TitansGenerationMixin` | `TitansCache` |
| `AtlasModel` | `AtlasGenerationMixin` | `AtlasCache` |
