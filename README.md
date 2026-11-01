<p align="center">
  <a href="https://navidsharifi.github.io/optora/">
    <img src="https://raw.githubusercontent.com/navidsharifi/optora/main/docs/assets/images/optora-logo.jpg" alt="Optora" width="560">
  </a>
</p>

# Optora

**A small, composable, PyTorch-native library for distributionally robust optimization (DRO).**

Status: alpha &middot; Python 3.10+ &middot; PyTorch &ge;2.2 &middot; MIT License

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23850922.svg)](https://doi.org/10.5281/zenodo.23850922)

I started building Optora because most of the DRO code I ran into while
reading papers lived inside one-off experiment scripts, hard-wired to
whatever setup that particular paper used. Optora is a small, composable
set of pieces instead — a divergence, an ambiguity set, a solver — that
you wire together yourself rather than calling a `fit()` that picks an
algorithm on your behalf. Everything runs on PyTorch tensors, so gradients
flow through the whole pipeline and nothing here blocks running on a GPU.

## Quick start

Install the latest release from PyPI:

```powershell
pip install optora
```

To develop locally, install from a clone in editable mode:

```powershell
pip install -e ".[dev,docs]"
```

Find the decision `x` that minimizes the worst-case squared error against
a KL-ball of plausible distributions over four observed outcomes:

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

## What is implemented today

Ambiguity sets (`optora.dro`), the divergences underneath them
(`optora.divergences`), and the solvers underneath those (`optora.solvers`)
are implemented end to end: `KLAmbiguitySet`, `PhiAmbiguitySet` (plus
`ChiSquareAmbiguitySet`, `TotalVariationAmbiguitySet`, and
`CVaRAmbiguitySet`), `SinkhornAmbiguitySet`, `MMDAmbiguitySet`, and
`WassersteinAmbiguitySet`, wired together by `MinimaxSolver`. Minibatch
training is covered by `MultilevelMonteCarloEstimator`,
`StochasticPhiDual`, and `OnlineGroupDRO` (group DRO, with
`optora.dro.group_losses`). Every piece
is covered by pytest tests checked against known closed-form results,
independent grid-search cross-checks, convergence limits, and
monotonicity properties, and the whole package is type-checked under
mypy's strict mode.

Every main class is importable from the package root
(`from optora import KLAmbiguitySet`) or from its subpackage
(`from optora.dro import KLAmbiguitySet`); both name the same object.
Convergence diagnostics and helper functions live in `optora.core`.

## Learn more

- **Full mathematical formulations and design rationale:**
  [`docs/formulations.md`](docs/formulations.md).
- **Training an `nn.Module` against a DRO objective with `torch.optim`:**
  [`docs/training.md`](docs/training.md).
- **Runnable, visualized DRO examples** (GitHub-only, not shipped with the
  package): [`examples/`](examples/README.md) — install with the
  `examples` extra (`pip install -e ".[examples]"`).
- **API reference:** built with Zensical from source docstrings, see
  [`docs/`](docs/index.md).
- **Benchmarks:** latency, scaling, batched throughput, host
  synchronizations and agreement with independent reference solvers,
  measured on every documentation build:
  [navidsharifi.github.io/optora/benchmarks](https://navidsharifi.github.io/optora/benchmarks/).
- **Positioning:** how Optora compares with `skwdro` and `dro`, and when
  another library fits better:
  [`docs/positioning.md`](docs/positioning.md).
- **API stability:** what counts as public API and what may change during
  `0.x`: [`docs/stability.md`](docs/stability.md).
- **Contributing:** development workflow, coding style, and the PR
  checklist live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing Optora

If you use Optora in your research, please cite it. Every release is archived
on Zenodo under the concept DOI
[10.5281/zenodo.23850922](https://doi.org/10.5281/zenodo.23850922), which
always resolves to the latest version. The metadata lives in
[`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button turns
it into APA or BibTeX.

## Development

Optora targets Python 3.10+ and PyTorch 2.2+. From an activated virtual
environment in the repository root:

```powershell
pip install -e ".[dev,docs]"
pytest --cov
ruff check .
ruff format --check .
mypy
```

CI runs the tests on Linux, macOS, and Windows against both the minimum and
the latest PyTorch, enforces the coverage gate set in `pyproject.toml`, and
installs the built wheel into a clean virtual environment to run every
script under `examples/`. Reproduce that smoke test locally with:

```powershell
python -m build
python -m tools.smoke
```

CI also times the benchmarks in `tests/benchmarks/` against the Pull
Request's base commit and fails on a large regression; each benchmark's exact
host-synchronization budget is checked by every `pytest` run (see
[CONTRIBUTING.md](CONTRIBUTING.md#performance-guard)).

Generate the hierarchical API reference, preview it, or build the complete
documentation site without maintaining API pages or navigation by hand:

```powershell
python -m tools.figures        # render the site's figures and benchmarks
python -m tools.docs generate
python -m tools.docs check
python -m tools.docs serve
python -m tools.docs build
```

`tools.figures` renders every chart, animation and benchmark on the site
from the library itself, in both colour schemes, into the git-ignored
`docs/assets/figures/` (`--list` shows what exists, `--only NAME` renders
one, `--quick` is a fast low-fidelity pass). The site builds without them,
so the step is optional locally; the deploy workflow always runs it.

`generate` writes one page per public module (with its classes and functions)
plus a matching index for every subpackage, and rewrites the API navigation in
`zensical.toml`. Because the generated pages are committed, plain
`zensical serve` and `zensical build` also work from a fresh clone; run
`python -m tools.docs serve` whenever `optora/` changes so the pages are
regenerated first. `check` fails when the committed pages drift from the
sources, and runs in CI.

Every push to `main` publishes the site to GitHub Pages through
`.github/workflows/docs.yml`, which enables Pages automatically and deploys the
freshly generated build.

## License

Optora is distributed under the MIT License; see [`LICENSE`](../LICENSE)
for the full text.
