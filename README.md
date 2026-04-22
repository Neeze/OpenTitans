# OpenTitans: The Open-Source Framework for Memory-Augmented Sequence Models

## Introduction

OpenTitans is a modular, extensible, and user-friendly framework designed to democratize access to the latest breakthroughs in Memory-Augmented Sequence Models. While Transformers have dominated sequence modeling, their quadratic complexity limits their ability to handle infinite context. Recent research from Google has introduced a new paradigm: models that learn to memorize and optimize their internal state at test time.

OpenTitans provides a unified, HuggingFace-like API to train, evaluate, and deploy these cutting-edge architectures, allowing the community to build their own "Titans".

## Features

- **🚀 Efficient Implementations**: Optimized modules for Neural Memory and Test-Time Memorization.
- **🔗 Associative Scan**: Support for fast sequence processing via associative scans.
- **🧩 Modular Design**: Easy-to-extend components for research and development.
- **📊 Benchmarking**: Built-in tools for performance evaluation.

## Quick Start

### Installation

> [!NOTE]
> OpenTitans is currently under active development. You can install it directly from the source.

```bash
git clone https://github.com/Neeze/OpenTitans.git
cd OpenTitans
pip install -e .
```

## Citations & Acknowledgements

This project is an unofficial implementation. All credit for the theoretical foundations and original architectural designs belongs to the authors of the following papers:

```bibtex
@misc{behrouz2024titanslearningmemorizetest,
      title={Titans: Learning to Memorize at Test Time}, 
      author={Ali Behrouz and Peilin Zhong and Vahab Mirrokni},
      year={2024},
      eprint={2501.00663},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.00663}, 
}

@misc{behrouz2025atlaslearningoptimallymemorize,
      title={ATLAS: Learning to Optimally Memorize the Context at Test Time}, 
      author={Ali Behrouz and Zeman Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab Mirrokni},
      year={2025},
      eprint={2505.23735},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23735}, 
}

@misc{behrouz2025nestedlearningillusiondeep,
      title={Nested Learning: The Illusion of Deep Learning Architectures}, 
      author={Ali Behrouz and Meisam Razaviyayn and Peilin Zhong and Vahab Mirrokni},
      year={2025},
      eprint={2512.24695},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.24695}, 
}

@misc{behrouz2026memorycachingrnnsgrowing,
      title={Memory Caching: RNNs with Growing Memory}, 
      author={Ali Behrouz and Zeman Li and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab Mirrokni},
      year={2026},
      eprint={2602.24281},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.24281}, 
}

@misc{behrouz2025itsconnectedjourneytesttime,
      title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization}, 
      author={Ali Behrouz and Meisam Razaviyayn and Peilin Zhong and Vahab Mirrokni},
      year={2025},
      eprint={2504.13173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.13173}, 
}
```