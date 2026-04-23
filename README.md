<div align="center">

# 🌌 OpenTitans
**The Open-Source Framework for Memory-Augmented Sequence Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Neeze/OpenTitans/graphs/commit-activity)

---

**Democratizing Test-Time Memorization and Neural Memory Architectures.**

[Introduction](#-introduction) • [Documentation](./docs/README.md) • [Features](#-key-features) • [Quick Start](#-quick-start) • [Citations](#-citations--acknowledgements)

</div>

## 🌟 Introduction

**OpenTitans** is a modular, high-performance framework designed to implement and explore the next generation of sequence models. While Transformers revolutionized AI, their quadratic context limitations have met their match. 

Inspired by groundbreaking research from Google and other top labs, OpenTitans focuses on **Memory-Augmented Models** that learn to memorize, optimize, and cache their internal states at test time. Our goal is to provide a "HuggingFace-like" experience for researchers and engineers building the future of infinite-context modeling.

---


## 📖 Documentation

For detailed information on how to use OpenTitans, please refer to our **[Documentation Index](./docs/README.md)**.

*   **[Installation Guide](./docs/Installation.md)**: Setup and dependency management.
*   **[Quickstart](./docs/Quickstart.md)**: Your first model in 5 minutes.
*   **[Titans Variants](./docs/Titans_Variants.md)**: Understanding MAC, MAG, and MAL.
*   **[Neural Memory](./docs/Neural_Memory.md)**: Customizing memory models and fast-weight updates.

---

## 📦 Quick Start

### Installation

> [!TIP]
> We recommend using a virtual environment (venv or conda) for the best experience.

```bash
pip install open-titans
```

#### Development Installation

```bash
# Clone the repository
git clone https://github.com/Neeze/OpenTitans.git
cd OpenTitans

# Install in editable mode with dependencies
pip install -e .
```

---

## 🤝 Contributing

We are looking for "Titans" to help us build! 🚀

Whether you want to implement a new paper, optimize a CUDA kernel, or just fix a typo, your contributions are welcome. Check out our [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📚 Citations & Acknowledgements

OpenTitans stands on the shoulders of giants. We acknowledge the authors of the following papers for their foundational work:

```bibtex
@misc{behrouz2024titanslearningmemorizetest,
      title={Titans: Learning to Memorize at Test Time}, 
      author={Ali Behrouz and Peilin Zhong and Vahab Mirrokni},
      year={2024},
      url={https://arxiv.org/abs/2501.00663}
}

@misc{behrouz2025itsconnectedjourneytesttime,
      title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization}, 
      author={Ali Behrouz and Meisam Razaviyayn and Peilin Zhong and Vahab Mirrokni},
      year={2025},
      url={https://arxiv.org/abs/2504.13173}
}

@misc{behrouz2025atlaslearningoptimallymemorize,
      title={ATLAS: Learning to Optimally Memorize the Context at Test Time}, 
      author={Ali Behrouz and Zeman Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab Mirrokni},
      year={2025},
      url={https://arxiv.org/abs/2505.23735}
}
```

---

## 📄 License

OpenTitans is released under the **MIT License**. See [LICENSE](LICENSE) for more details.