---
icon: lucide/compass
---

# Positioning

Optora is one of several Python libraries for distributionally robust
optimization (DRO). This page says what it does differently, where another
library is the better choice, and how the claims are checked. The facts
about other libraries were taken from their own repositories, documentation
and papers on 2026-11-01; every one carries a footnote with its source.
Corrections are welcome as
[issues](https://github.com/navidsharifi/optora/issues).

## In one sentence

Optora is a small set of composable PyTorch building blocks, a `Divergence`,
an `AmbiguitySet` that solves its worst case exactly on a finite support, and
a `Solver`. Its worst-case expectation is an ordinary differentiable loss for
any `nn.Module` and `torch.optim` loop, and every solve is checked against an
independent reference implementation.

## Comparison

| | Optora | `skwdro` | `dro` (python-dro) |
| --- | --- | --- | --- |
| Latest release | [![optora on PyPI](https://img.shields.io/pypi/v/optora?style=flat-square&label=)](https://pypi.org/project/optora/) | [![skwdro on PyPI](https://img.shields.io/pypi/v/skwdro?style=flat-square&label=)](https://pypi.org/project/skwdro/) | [![dro on PyPI](https://img.shields.io/pypi/v/dro?style=flat-square&label=)](https://pypi.org/project/dro/) |
| License | MIT | BSD-3-Clause[^skwdro-repo] | MIT[^dro-pypi] |
| Runtime dependencies | `torch` only | torch, scikit-learn, cvxpy, cvxopt, dask and others[^skwdro-pypi] | torch, scikit-learn, cvxpy, xgboost, lightgbm and others[^dro-pypi] |
| Ambiguity sets | KL, general $\varphi$, $\chi^2$, total variation, CVaR, Wasserstein, Sinkhorn, MMD, group DRO | Wasserstein, entropically smoothed[^skwdro-why] | $\chi^2$, KL, CVaR, TV, Wasserstein, Sinkhorn, MMD, group DRO, holistic robust, outlier-robust Wasserstein and more, depending on the model family[^dro-repo] |
| Adversary support | finite: the nominal's scenarios, with a user-given cost or kernel matrix | continuous, sampled by Monte Carlo around each data point[^skwdro-why] | per method: finite-sample convex programs for linear models, adversarial perturbations for neural networks[^dro-wnn] |
| Interface | a robust-risk operator on a loss tensor, for any model | scikit-learn estimators, and `robustify` to wrap a PyTorch loss[^skwdro-torch] | scikit-learn-style estimators for fixed model families (linear, kernel, tree ensembles, neural networks)[^dro-linear] |
| Inner solve | exact to the working dtype: bracketed tilt, closed form, bisection, or an interior-point cone program | stochastic gradient on the smoothed dual[^skwdro-paper] | cvxpy with CLARABEL by default for linear models; gradient methods for neural networks[^dro-repo] |
| Worst-case distribution | every set, as the gradient of the worst case with respect to the loss | not documented | $\chi^2$, KL and Wasserstein estimators[^dro-worst] |
| Accelerators | device-agnostic tensors, zero host synchronizations per exact solve | not documented | `device="cuda"` for neural and Sinkhorn models[^dro-nn] |
| Verification | independent NumPy and SciPy references, property-based tests, `gradcheck` and `gradgradcheck` | unit tests, linting, coverage[^skwdro-repo] | unit tests per method, coverage[^dro-tests] |

## What Optora does differently

<div class="grid cards" markdown>

-   :material-puzzle-outline:{ .lg .middle } __Composable contracts__

    ---

    The three abstract base classes are the API. A new divergence or
    ambiguity set is a subclass that works with every solver, estimator and
    training loop already in the library, rather than one more estimator
    class per model.

    [:octicons-arrow-right-24: Core abstractions](formulations.md#core-abstractions)

-   :material-check-decagram:{ .lg .middle } __Exact worst cases__

    ---

    On a finite support, each set solves its inner supremum to the precision
    of its dtype, with no inner tolerance or step size to tune. The robust
    loss is the worst case itself, not a smoothed or sampled surrogate.

    [:octicons-arrow-right-24: Formulations](formulations.md#formulations-implemented-today)

-   :material-function-variant:{ .lg .middle } __Correct gradients__

    ---

    First and second derivatives in the loss, the nominal and the radius
    follow from the envelope theorem. They are checked with `gradcheck` and
    `gradgradcheck` at smooth points, and with finite differences at kinks.

    [:octicons-arrow-right-24: Training guide](training.md)

-   :material-scale-balance:{ .lg .middle } __Independent oracles__

    ---

    Every ambiguity set is compared with a NumPy or SciPy linear, cone or
    closed-form solver that shares no code with Optora. The agreement below
    is measured on every documentation build.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md#against-independent-reference-solvers)

</div>

Measured by this build of the documentation (CPU runner, float32, see
[Benchmarks](benchmarks.md) for the environment and the full tables):

--8<-- "docs/assets/figures/benchmarks-highlights.md"

## When another library fits better

!!! tip "Choose by problem, not by library"

    - **Continuous adversary.** If the worst-case distribution may move data
      points anywhere in feature space, as in Wasserstein DRO with a
      continuous transport cost, use `skwdro`[^skwdro-why] or the adversarial
      Wasserstein estimators of `dro`[^dro-wnn]. Optora's sets are restricted
      to a finite support that you choose.
    - **A ready-made estimator.** For a robust logistic regression, support
      vector machine or gradient-boosted tree behind `fit` and `predict`,
      `dro` covers many formulations for each model family[^dro-repo].
    - **General robust optimization models.** For robust and
      distributionally robust linear or conic programs with modelling
      syntax and external solvers, RSOME is built for that[^rsome].

Optora fits when you train your own PyTorch model, want the exact worst case
of a standard ambiguity set as a differentiable loss, need to compare
several geometries under one interface, or are building a new set on top of
tested contracts.

[^skwdro-pypi]: PyPI metadata, <https://pypi.org/pypi/skwdro/json>.
[^skwdro-repo]: Repository README, <https://github.com/iutzeler/skwdro>.
[^skwdro-why]: "Why skwdro", <https://skwdro.readthedocs.io/latest/why_skwdro.html>.
[^skwdro-torch]: PyTorch interface, <https://skwdro.readthedocs.io/latest/pytorch.html>.
[^skwdro-paper]: Vincent, Azizian, Iutzeler and Malick, "skwdro: a library for Wasserstein distributionally robust machine learning", JMLR 27(8), 2026, <https://jmlr.org/papers/v27/24-1840.html>.
[^dro-pypi]: PyPI metadata, <https://pypi.org/pypi/dro/json>.
[^dro-repo]: Repository README, <https://github.com/namkoong-lab/dro>; Liu, Wang, Lam, Namkoong and Blanchet, arXiv:2505.23565, <https://arxiv.org/abs/2505.23565>.
[^dro-linear]: Linear model API, <https://python-dro.org/api/apis/base_linear.html>.
[^dro-nn]: Neural model API, <https://python-dro.org/api/apis/base_nn.html>.
[^dro-wnn]: Wasserstein neural DRO, <https://python-dro.org/api/apis/wdro_nn.html>.
[^dro-worst]: <https://python-dro.org/api/apis/chi2dro_linear.html>, <https://python-dro.org/api/apis/kldro_linear.html>, <https://python-dro.org/api/apis/wdro_linear.html>.
[^dro-tests]: Test directory, <https://github.com/namkoong-lab/dro/tree/HEAD/tests>.
[^rsome]: RSOME documentation, <https://xiongpengnus.github.io/rsome/>.
