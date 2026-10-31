---
icon: lucide/house
---

# Optora { .hero__title }

<div class="hero" markdown>

![Optora](assets/images/optora-logo.jpg){ .hero__logo }

<p class="hero__tagline" markdown>
A small, composable, PyTorch-native library for <b>distributionally robust
optimization</b>: ambiguity sets, the divergences underneath them, and the
minimax solve that connects them, all differentiable and GPU-ready.
</p>

<div class="hero__actions" markdown>

[Get started :octicons-arrow-right-24:](#install){ .md-button .md-button--primary }
[:fontawesome-brands-github: Star on GitHub ![GitHub stars](https://img.shields.io/github/stars/navidsharifi/optora?style=social&label=){ .hero__star-count }](https://github.com/navidsharifi/optora){ .md-button }

</div>

</div>

<div class="badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/optora?style=flat-square&color=4051b5)](https://pypi.org/project/optora/)
[![Python](https://img.shields.io/pypi/pyversions/optora?style=flat-square)](https://pypi.org/project/optora/)
[![License](https://img.shields.io/github/license/navidsharifi/optora?style=flat-square)](https://github.com/navidsharifi/optora/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/navidsharifi/optora/ci.yml?branch=main&style=flat-square&label=ci)](https://github.com/navidsharifi/optora/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/navidsharifi/optora/docs.yml?branch=main&style=flat-square&label=docs)](https://github.com/navidsharifi/optora/actions/workflows/docs.yml)
</div>

<!-- optora-version-start -->
Latest release: `v0.1.0`
<!-- optora-version-end -->

## Overview

Empirical risk minimization trusts the training distribution. Optora
optimizes against the *worst* distribution within a chosen neighbourhood of
it instead,

$$
\min_x \; \sup_{q \,:\, D(q \,\|\, \mathrm{nominal}) \le \mathrm{radius}} \mathbb{E}_q[\ell(x, \xi)],
$$

and ships that formula as three composable pieces: a `Divergence` is $D$, an
`AmbiguitySet` bundles a divergence with a nominal distribution and a radius
and solves the inner supremum exactly, and a `Solver` minimizes over the
decision. The worst-case expectation is an ordinary differentiable PyTorch
objective, so it drops into any `nn.Module` and `torch.optim` loop.
The derivations live in [Mathematical formulations](formulations.md).

## In action

Seven ambiguity sets, one nominal distribution (grey) and one loss. Each
adversary spends the same fraction of its own budget, and each geometry
moves the probability mass somewhere different:

<figure class="figure" markdown="span">
  ![Where each ambiguity set's adversary moves the probability mass as its budget grows](assets/figures/mass-migration-light.gif#only-light){ loading=lazy }
  ![Where each ambiguity set's adversary moves the probability mass as its budget grows](assets/figures/mass-migration-dark.gif#only-dark){ loading=lazy }
  <figcaption>The worst-case distribution of every ambiguity set on the same
  twelve-scenario problem, read off autograd as the gradient of the
  worst-case expectation, as each ball grows to the radius at which it
  contains the point mass on the highest loss.</figcaption>
</figure>

<figure class="figure" markdown="span">
  ![The seven ambiguity sets drawn on the probability simplex as their radius grows](assets/figures/simplex-geometry-light.gif#only-light){ loading=lazy }
  ![The seven ambiguity sets drawn on the probability simplex as their radius grows](assets/figures/simplex-geometry-dark.gif#only-dark){ loading=lazy }
  <figcaption>The same sets on a three-scenario simplex: the shaded region is
  the ball around the nominal (grey dot), the trail is the path of the
  worst-case distribution, and the last panel is the worst-case value.</figcaption>
</figure>

Every figure on this site is rendered from the library at build time, in
both colour schemes, by `python -m tools.figures`; nothing is hand drawn
and no image is committed to the repository.

## Core components

<div class="grid cards" markdown>

-   :material-shape-outline:{ .lg .middle } __Core contracts__

    ---

    `Solver`, `Divergence` and `AmbiguitySet` are the three abstract base
    classes a researcher subclasses, plus device-resident convergence
    tracking and the working-dtype policy.

    [:octicons-arrow-right-24: `optora.core`](api/core/index.md)

-   :material-vector-triangle:{ .lg .middle } __Divergences__

    ---

    Kullback-Leibler, general $\varphi$, $\chi^2$, total variation, the
    max divergence, the Sinkhorn and entropic transport divergences, and
    maximum mean discrepancy.

    [:octicons-arrow-right-24: `optora.divergences`](api/divergences/index.md)

-   :material-shield-half-full:{ .lg .middle } __Ambiguity sets__

    ---

    KL, $\varphi$, $\chi^2$, total variation, CVaR, Wasserstein, Sinkhorn and
    MMD balls, each solving its worst case exactly: a bracketed tilt, a
    closed form, a bisection, or an interior-point cone program.

    [:octicons-arrow-right-24: `optora.dro`](api/dro/index.md)

-   :material-swap-vertical-bold:{ .lg .middle } __Minimax solve__

    ---

    `MinimaxSolver` composes any ambiguity set with an outer solver over the
    decision variable; the inner supremum is already differentiable by the
    envelope theorem.

    [:octicons-arrow-right-24: `MinimaxSolver`](api/dro/minimax_solver.md)

-   :material-chart-timeline-variant:{ .lg .middle } __Solvers__

    ---

    Fixed-step gradient descent and a primal-dual saddle-point solver, both
    with device-side stopping rules and `torch.func` support.

    [:octicons-arrow-right-24: `optora.solvers`](api/solvers/index.md)

-   :material-tune:{ .lg .middle } __Radius calibration__

    ---

    Turn a confidence level and a sample size into a KL, $\chi^2$, general
    $\varphi$ or Wasserstein radius with finite-sample guarantees.

    [:octicons-arrow-right-24: `optora.dro.calibration`](api/dro/calibration.md)

-   :material-layers-triple-outline:{ .lg .middle } __Minibatch estimators__

    ---

    A multilevel Monte Carlo estimator and a stochastic $\varphi$-dual make
    the worst case unbiased on minibatches drawn from a population.

    [:octicons-arrow-right-24: `optora.dro.minibatch`](api/dro/minibatch.md)

-   :material-school-outline:{ .lg .middle } __Training__

    ---

    The worst case trains like any PyTorch loss: exact first and second
    derivatives in the loss, the nominal and the radius.

    [:octicons-arrow-right-24: Training guide](training.md)

</div>

## Why Optora

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } __Robust by construction__

    ---

    Optimize against the worst case over an ambiguity set instead of trusting a
    single empirical distribution.

    [:octicons-arrow-right-24: Formulations](formulations.md)

-   :material-check-decagram:{ .lg .middle } __Exact inner solves__

    ---

    No inner tolerance to tune: every shipped ambiguity set reaches the
    precision of its dtype, cross-checked against independent LP, cone and
    closed-form references.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-speedometer:{ .lg .middle } __GPU-first__

    ---

    Vectorized PyTorch throughout, batched over leading dimensions, zero
    host synchronizations per exact solve, and tensor state that moves with a
    single `.to(device)` call.

-   :material-flask-outline:{ .lg .middle } __Verified numerics__

    ---

    Property-based invariants, `gradcheck` at smooth points and at kinks, a
    documented dtype table, and a benchmark suite that fails CI on a
    regression.

    [:octicons-arrow-right-24: Numerical behaviour](numerics.md)

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

## Quick start

Find the decision $x$ that minimizes the worst-case squared error against a
KL ball of plausible distributions over four observed outcomes:

```python
import torch
from optora import GradientDescent, KLAmbiguitySet, MinimaxProblem, MinimaxSolver

nominal = torch.tensor([0.25, 0.25, 0.25, 0.25])
outcomes = torch.tensor([1.0, 2.0, 3.0, 10.0])

problem = MinimaxProblem(
    ambiguity_set=KLAmbiguitySet(nominal=nominal, radius=0.1),
    loss_fn=lambda x: (outcomes - x) ** 2,
    initial_point=torch.tensor(0.0),
)
outer_solver = GradientDescent(step_size=0.01, max_iter=200, tol=1e-6)
robust_decision = MinimaxSolver(solver=outer_solver).solve(problem).point
```

To train an `nn.Module` instead, call `worst_case_expectation` on the
per-sample loss and step any `torch.optim` optimizer; see the
[training guide](training.md).

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
    python -m tools.figures   # render the figures once (optional)
    python -m tools.docs serve
    ```

=== "Production build"

    ```bash
    python -m tools.figures
    python -m tools.docs build
    ```

!!! tip "The API reference and the figures are generated"

    `tools.docs` discovers every public module under `optora/`, writes one page
    per module, and rewrites the API navigation. `tools.figures` renders every
    chart, animation and benchmark on the site from the library itself, in
    both colour schemes. Never edit the generated pages by hand, and never
    commit a rendered figure.

## Versioning

Optora is alpha and follows Semantic Versioning: during `0.x`, a minor
release may change the public API and a patch release does not. See
[API stability](stability.md) for what counts as public.

## Citing

If you use Optora in your research, please cite it with the metadata in
[`CITATION.cff`](https://github.com/navidsharifi/optora/blob/main/CITATION.cff),
which GitHub's "Cite this repository" button turns into APA or BibTeX.

## Contributing

Development workflow, coding style, and the pull request checklist live in
[`CONTRIBUTING.md`](https://github.com/navidsharifi/optora/blob/main/CONTRIBUTING.md).
Enjoying Optora? A star on
[GitHub :fontawesome-brands-github:](https://github.com/navidsharifi/optora)
helps other researchers find it.
