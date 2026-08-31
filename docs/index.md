---
icon: lucide/rocket
---

# Optora

Optora is a pre-alpha optimization library focused on a small GPU-first
PyTorch deterministic core that can grow toward stochastic methods,
differentiable backends, optimal transport, and reinforcement learning.

<!-- optora-version-start -->
Latest release: `v0.0.5`
<!-- optora-version-end -->

[Get started :octicons-arrow-right-24:](#install){ .md-button .md-button--primary }
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
