# Optora examples

Runnable, self-contained scripts that put `optora`'s DRO formulations to
work on concrete problems and visualize the results. Everything here is
GitHub-only documentation content: **`examples/` is deliberately excluded
from the PyPI wheel and sdist** (see `MANIFEST.in` and the
`[tool.setuptools]` configuration in `pyproject.toml`), so nothing here
ever adds weight to `pip install optora`.

## Setup

Examples need `matplotlib`, which is *not* a runtime dependency of
`optora` itself. Install it via the dedicated `examples` extra:

```powershell
pip install -e ".[examples]"
```

Then run any script directly from the repository root:

```powershell
python examples/01_kl_dro_radius_sweep.py
```

Each script prints its key numerical checks to stdout and saves any
figure it produces under `examples/outputs/` (git-ignored — see
`.gitignore`). Nothing under `examples/outputs/` is meant to be committed.

Every script below also has a walkthrough page on the documentation site
under [Examples](https://navidsharifi.github.io/optora/examples/), which
embeds the same source and explains what each experiment is checking.

## What is here

| Script | Demonstrates |
| --- | --- |
| [`01_kl_dro_radius_sweep.py`](01_kl_dro_radius_sweep.py) | The core DRO primitive in isolation: how `KLAmbiguitySet.worst_case_expectation` grows with the ambiguity radius, checked against its two closed-form limits (`radius = 0` and `radius -> inf`). |
| [`02_worst_case_distribution_shift.py`](02_worst_case_distribution_shift.py) | Reconstructs the actual worst-case candidate distribution `q*` inside a total-variation ambiguity set (not just its expectation) and visualizes probability mass migrating toward the worst-case scenario as the radius grows — convergence *across probability space*, not just of a scalar. |
| [`03_robust_decision_across_ambiguity_sets.py`](03_robust_decision_across_ambiguity_sets.py) | Solves the same robust decision problem with `MinimaxSolver` under all four ambiguity-set geometries (KL, chi-square, total variation, Wasserstein) and compares how the robust decision diverges from the empirical-risk baseline as radius grows. |
| [`04_convergence_diagnostics.py`](04_convergence_diagnostics.py) | Manually unrolls the outer gradient trajectory of a robust decision problem, then independently verifies the inner KL-DRO solve against a fine grid search over the dual variable. |
| [`05_wasserstein_lipschitz_equivalence.py`](05_wasserstein_lipschitz_equivalence.py) | Measures the Wasserstein-DRO / Lipschitz-regularization equivalence numerically: the exact dual value of `WassersteinAmbiguitySet` against the linear surrogate `E_nominal[loss] + radius * Lip(loss)`, as the radius shrinks and as the sample refines. |
| [`06_calibrated_ambiguity_radius.py`](06_calibrated_ambiguity_radius.py) | Replaces the hand-picked radius with the Duchi-Namkoong finite-sample calibration `radius = chi2_{1, 1 - alpha} / n`, then measures how often the resulting `ChiSquareAmbiguitySet` interval really covers the true mean, across nominal confidence levels and sample sizes. |
| [`07_subpopulation_shift_training.py`](07_subpopulation_shift_training.py) | Trains a linear model with `torch.optim.Adam` on the KL-DRO objective and compares it with ERM, a ridge control and a noise-only negative control under subpopulation shift, using exact group risks, paired seeds and convergence checks against closed-form and L-BFGS references. |

## Media asset policy

Rich visual content (plots, GIFs, animations) referenced from this file or
any other markdown in the project **must be hosted externally** — a CDN,
an image host, or a dedicated orphan branch/release asset outside the main
Git history — and linked with a plain `![alt](https://.../asset.png)` URL.
Do not commit `.png`/`.gif`/`.mp4` files into the repository tree: they
bloat every clone and every `git fetch` for all contributors indefinitely,
long after the figure itself is stale. `.gitignore` already blocks
anything the scripts above generate under `examples/outputs/` from being
tracked by accident.

To add a real visual to this README:

1. Run the script that produces the figure you want (see the table
   above); it is written to `examples/outputs/<name>.png`.
2. Upload that file to an external host of your choice (for example a
   dedicated `media`/`gh-pages`-style orphan branch, a release asset, or
   an image CDN) — never to a branch that `main` merges from.
3. Reference the resulting URL here. Optora hosts its figures on the
   orphan `media` branch (it shares no history with `main`); the file
   `01_kl_dro_radius_sweep.png` on that branch renders as:

   ![KL-DRO worst-case expectation vs. radius](https://raw.githubusercontent.com/navidsharifi/optora/media/01_kl_dro_radius_sweep.png)

   To add another figure, commit it to the `media` branch (for example in a
   `git worktree`) and link
   `https://raw.githubusercontent.com/navidsharifi/optora/media/<name>.png`;
   do not link to a path inside `main`.

## Notes on solver tuning

Several examples nest an outer `MinimaxSolver`/`GradientDescent` around an
ambiguity set's own inner solve. That composition multiplies iteration
counts fast in eager PyTorch, so the scripts here deliberately use small,
tuned-down outer `max_iter`/`step_size` budgets rather than the solver's
own defaults, and check qualitative behavior (monotonicity, closed-form
limits, cross-checked reconstructions) rather than tight-tolerance
convergence. See `progress/decisions.md` for the underlying numerical
background.
