# ⚙️ Installation

To get started with OpenTitans, ensure you have a modern Python environment and the necessary dependencies.

## 📋 Requirements

-   **Python**: 3.10 or higher (3.12 recommended)
-   **PyTorch**: 2.0 or higher
-   **CUDA**: Required for GPU acceleration (NVIDIA GPU with appropriate drivers)

## 📦 Installation via Pip

You can install OpenTitans directly from PyPI:

```bash
pip install open-titans
```

---

## 🛠️ Installing from Source

We recommend installing OpenTitans in a virtual environment.

### 1. Clone the repository
```bash
git clone https://github.com/Neeze/OpenTitans.git
cd OpenTitans
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install the package in editable mode
```bash
pip install -e .
```

This will install all core dependencies, including:
-   `torch`
-   `einops`
-   `transformers`
-   `ninja` (for fast kernel compilation)
-   `rotary-embedding-torch`
-   `assoc-scan`

## 🧪 Verifying the Installation

You can verify that everything is set up correctly by running the test suite:

```bash
# Run all tests
pytest tests/

# Or run a specific test for a model variant
python tests/test_titans_mac.py
```

If you have a GPU, the tests will automatically attempt to run on CUDA and report performance metrics.
