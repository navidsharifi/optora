---
icon: lucide/rocket
---

# Optora

Optora is a pre-alpha optimization library focused on a small GPU-first
PyTorch deterministic core that can grow toward stochastic methods,
differentiable backends, optimal transport, and reinforcement learning.

<!-- optora-version-start -->
Latest release: `v0.1.0`
<!-- optora-version-end -->

## Install

```bash
pip install optora
```

For local development, install the project in editable mode with the developer
and documentation extras:

```bash
pip install -e ".[dev,docs]"
```

## Build documentation

```bash
zensical build --clean
```

The API reference is generated from source code with Zensical's `mkdocstrings`
plugin integration.
