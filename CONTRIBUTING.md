# Contributing to OpenTitans

First off, thank you for considering contributing to OpenTitans! It's people like you who make the open-source community such an amazing place to learn, inspire, and create.

OpenTitans is a research-oriented project, and we welcome contributions ranging from bug fixes to implementing new architectural variants from recent literature.

---

## How Can I Contribute?

### Implementing Research Papers
The primary goal of this repository is to provide clean, efficient implementations of Memory-Augmented Sequence Models. If you see a paper (like those mentioned in the [README](README.md)) that hasn't been fully implemented or could be improved, we'd love your help!

### Reporting Bugs
If you find a bug, please open an issue with a clear description, reproduction steps, and the expected vs. actual behavior.

### Feature Requests
Have an idea for a new feature? Open an issue and let's discuss it!

---

## Development Setup

To get started with development:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/OpenTitans.git
    cd OpenTitans
    ```
3.  **Install development dependencies**:
    ```bash
    pip install -e .
    pip install pytest ruff
    ```

---

## Development Workflow

### Branching Strategy
- Always create a new branch for your work: `git checkout -b feat/your-feature-name` or `fix/issue-description`.
- Keep branches focused on a single change.

### Coding Standards
- **Style**: We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. You can run `make lint` to check your code.
- **Type Hints**: Please use Python type hints for all new code to improve maintainability.
- **Documentation**: Add docstrings to all new classes and functions.

### Testing
Before submitting a pull request, ensure all tests pass:
```bash
make test
```
If you're adding a new feature, please include corresponding unit tests in the `tests/` directory.

---

## Pull Request Process

1.  Ensure your code adheres to the coding standards and passes all tests.
2.  Update the `README.md` if your change introduces new functionality or changes existing APIs.
3.  Submit a Pull Request (PR) against the `main` branch.
4.  Provide a clear description of the changes in the PR body.
5.  Once submitted, a maintainer will review your PR and provide feedback.

---

## Code of Conduct
We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

---

## Makefile Commands
For convenience, we provide a `Makefile` with common development tasks:
- `make install`: Install the package in editable mode.
- `make test`: Run the test suite.
- `make lint`: Run the linter.
- `make clean`: Remove build artifacts and caches.

---

Thank you for being a part of OpenTitans!
