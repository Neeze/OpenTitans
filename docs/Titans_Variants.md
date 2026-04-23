# 🎭 Titans Variants

OpenTitans provides multiple architectural variants of the Titans model, each exploring a different way to integrate neural memory with self-attention.

## 1. Memory as Context (MAC)

The **MAC** variant treats memory tokens as a persistent context that is prepended or interleaved into the input sequence. The model uses a specialized attention mask ("Unrestricted Prefix Attention") to allow sequence tokens to attend to these memory tokens.

### Key Features
-   **Interleaved Memory**: Long-term memory tokens are inserted at the start of each sequence segment.
-   **Contextual Recall**: The model learns to "read" from these tokens using standard attention mechanisms.
-   **Architecture**: `TitansMACModel`

```python
from open_titans.models.titans_mac import TitansMACModel, TitansMACConfig

config = TitansMACConfig(num_longterm_mem_tokens=8, segment_len=32)
model = TitansMACModel(config)
```

---

## 2. Memory as a Gate (MAG)

The **MAG** variant uses neural memory as a parallel branch to the attention mechanism. The output of the neural memory is used to *gate* (modulate) the output of the self-attention layer.

### Key Features
-   **Dual Branch**: Attention and Neural Memory run in parallel (or semi-parallel).
-   **Gated Fusion**: The memory output is passed through a sigmoid function to create a gate that scales the attention features.
-   **Architecture**: `TitansMAGModel`

```python
from open_titans.models.titans_mag import TitansMAGModel, TitansMAGConfig

config = TitansMAGConfig(window_size=32)
model = TitansMAGModel(config)
```

---

## 3. Memory as a Layer (MAL)

The **MAL** variant treats neural memory as a standard layer in the transformer block. It typically follows a sequential pattern: Neural Memory -> Sliding Window Attention -> FeedForward.

### Key Features
-   **Sequential Integration**: Memory acts as a distinct transformation step within each block.
-   **Simple & Powerful**: Directly modifies the residual stream before attention.
-   **Architecture**: `TitansMALModel`

```python
from open_titans.models.titans_mal import TitansMALModel, TitansMALConfig

config = TitansMALConfig(num_hidden_layers=12, neural_memory_layers=[1, 4, 8, 12])
model = TitansMALModel(config)
```

## ⚖️ Which one to use?

| Variant | Best For | Complexity |
| :--- | :--- | :--- |
| **MAC** | Explicit long-term context recall | Medium |
| **MAG** | Dynamic filtering of attention features | High |
| **MAL** | General purpose memory-augmented modeling | Low |
