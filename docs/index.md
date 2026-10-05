---
icon: lucide/rocket
---

# Optora

Optora is an alpha optimization library focused on a small GPU-first
PyTorch deterministic core that can grow toward stochastic methods,
differentiable backends, optimal transport, and reinforcement learning.

<!-- optora-version-start -->
Latest release: `v0.1.0`
<!-- optora-version-end -->

[Get started :octicons-arrow-right-24:](#install){ .md-button .md-button--primary }
[Training guide](training.md){ .md-button }
[Examples](examples/index.md){ .md-button }
[API reference](api/index.md){ .md-button }

## Why Optora

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } __Robust by construction__

    ---

    Optimize against the worst case over an ambiguity set instead of trusting a
    single empirical distribution.

    [:octicons-arrow-right-24: Formulations](formulations.md)

-   :material-vector-triangle:{ .lg .middle } __Five ambiguity sets__

    ---

    Kullback-Leibler, $\phi$-divergence, $\chi^2$, total variation, and
    Wasserstein, all behind one `AmbiguitySet` contract.

    [:octicons-arrow-right-24: API reference](api/index.md)

-   :material-speedometer:{ .lg .middle } __GPU-first__

    ---

    Vectorized PyTorch throughout, with tensor state that moves to an
    accelerator through a single `.to(device)` call.

-   :material-school-outline:{ .lg .middle } __Trains like any PyTorch loss__

    ---

    The worst-case expectation is an ordinary differentiable objective, so
    an `nn.Module` trains against it with plain `torch.optim`.

    [:octicons-arrow-right-24: Training guide](training.md)

-   :material-flask-outline:{ .lg .middle } __Verified numerics__

    ---

    Each formulation is checked against closed forms, independent grid
    searches, and convergence limits under mypy's strict mode.

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>

## Install

```bash
pip install optora
```

For local development, install the project in editable mode with the developer
and documentation extras:

```bash
pip install -e ".[dev,docs]"
```

## Imports

Every main class is re-exported from the package root, so the two forms below
name the same object:

```python
from optora import KLAmbiguitySet, MinimaxSolver
from optora.dro import KLAmbiguitySet, MinimaxSolver
```

The root exports classes only: the `Solver`, `Divergence`, and `AmbiguitySet`
contracts, their reference implementations, and the problem and result types.
Convergence diagnostics and helper functions stay in `optora.core`. There are
no root-level workflow functions; run an algorithm with `Solver.solve(problem)`.

## Build documentation

=== "Preview"

    ```bash
    python -m tools.docs serve
    ```

=== "Production build"

    ```bash
    python -m tools.docs build
    ```

!!! tip "The API reference is generated"

    `tools.docs` discovers every public module under `optora/`, writes one page
    per module, and rewrites the API navigation. Never edit the generated pages
    by hand.

## Contributing

Development workflow, coding style, and the pull request checklist live in
[`CONTRIBUTING.md`](https://github.com/navidsharifi/optora/blob/main/CONTRIBUTING.md).
