# Optora

**A small, composable, PyTorch-native library for distributionally robust optimization (DRO).**

Status: pre-alpha (`v0.0.3`) &middot; Python 3.10+ &middot; PyTorch &ge;2.2 &middot; MIT License

I started building Optora because most of the DRO code I ran into while
reading papers lived inside one-off experiment scripts, hard-wired to
whatever setup that particular paper used. Optora is a small, composable
set of pieces instead — a divergence, an ambiguity set, a solver — that
you wire together yourself rather than calling a `fit()` that picks an
algorithm on your behalf. Everything runs on PyTorch tensors, so gradients
flow through the whole pipeline and nothing here blocks running on a GPU.

## Quick start

Optora is pre-alpha and not yet published on PyPI; install it from a local
clone in editable mode:

```powershell
pip install -e ".[dev,docs]"
```

Find the decision `x` that minimizes the worst-case squared error against
a KL-ball of plausible distributions over four observed outcomes:

```python
import torch
from optora.dro import KLAmbiguitySet, MinimaxProblem, MinimaxSolver

nominal = torch.tensor([0.25, 0.25, 0.25, 0.25])
outcomes = torch.tensor([1.0, 2.0, 3.0, 10.0])

problem = MinimaxProblem(
    ambiguity_set=KLAmbiguitySet(nominal=nominal, radius=0.1),
    loss_fn=lambda x: (outcomes - x) ** 2,
    initial_point=torch.tensor(0.0),
)
robust_decision = MinimaxSolver().solve(problem).point
```

## What is implemented today

Ambiguity sets (`optora.dro`), the divergences underneath them
(`optora.divergences`), and the solvers underneath those (`optora.solvers`)
are implemented end to end: `KLAmbiguitySet`, `PhiAmbiguitySet` (plus
`ChiSquareAmbiguitySet` and `TotalVariationAmbiguitySet`), and
`WassersteinAmbiguitySet`, wired together by `MinimaxSolver`. Every piece
is covered by pytest tests checked against known closed-form results,
independent grid-search cross-checks, convergence limits, and
monotonicity properties, and the whole package is type-checked under
mypy's strict mode.

## Learn more

- **Full mathematical formulations and design rationale:**
  [`docs/formulations.md`](docs/formulations.md).
- **Runnable, visualized DRO examples** (GitHub-only, not shipped with the
  package): [`examples/`](examples/README.md) — install with the
  `examples` extra (`pip install -e ".[examples]"`).
- **API reference:** built with Zensical from source docstrings, see
  [`docs/`](docs/index.md).

## Development

Optora targets Python 3.10+ and PyTorch 2.2+. From an activated virtual
environment in the repository root:

```powershell
pip install -e ".[dev,docs]"
pytest
ruff check .
ruff format --check .
mypy
```

## License

Optora is distributed under the MIT License; see [`LICENSE`](../LICENSE)
for the full text.
