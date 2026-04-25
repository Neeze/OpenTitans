# OpenTitans

**OpenTitans** is a modular, high-performance framework designed to implement and explore the next generation of sequence models. While Transformers revolutionized AI, their quadratic context limitations have met their match in memory-augmented architectures.

Inspired by groundbreaking research from Google and other top labs, OpenTitans focuses on **Memory-Augmented Models** that learn to memorize, optimize, and cache their internal states at test time.

## Core Philosophy

The project is built on three pillars:
1.  **Test-Time Memorization**: Moving beyond static weights to models that can update their internal state during inference.
2.  **Modular Neural Memory**: Plug-and-play memory modules that can be integrated into various transformer-like architectures.
3.  **Infinite Context Modeling**: Efficiently handling extremely long sequences by leveraging persistent and long-term memory.

## Key Features

-   **Multiple Variants**: Implementations of MAC (Memory as Context), MAG (Memory as a Gate), and MAL (Memory as a Layer).
-   **Neural Memory Module**: A robust implementation of neural memory that supports fast weight updates.
-   **High Performance**: Optimized for both CPU and CUDA, with support for advanced techniques like rotary embeddings and axial positional embeddings.
-   **Research Friendly**: Clean, modular code designed for researchers to experiment with new memory-augmented mechanisms.

## Project Structure

-   `open_titans/models/`: Core implementations of different Titans variants.
-   `open_titans/memory/`: Neural memory modules and logic.
-   `open_titans/optim/`: Optimizers specifically designed for test-time training.
-   `tests/`: Comprehensive test suite for ensuring correctness and benchmarking performance.
