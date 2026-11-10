---
icon: lucide/compass
---

# Positioning

Optora is one of several Python libraries for distributionally robust
optimization (DRO). This page describes how it differs from the others,
when another library is the better choice, and how the claims are checked.
The statements about other libraries were taken from their own
repositories, documentation and papers on 2026-11-01, and each one has a
footnote with its source. Corrections are welcome as
[issues](https://github.com/navidsharifi/optora/issues).

## Overview

Optora is a small set of composable PyTorch classes, namely a `Divergence`,
an `AmbiguitySet` that solves its worst case on a finite support, and a
`Solver`. Its worst-case expectation is an ordinary differentiable loss for
any `nn.Module` and `torch.optim` loop, and the test suite checks every
ambiguity set against an independent reference implementation.

## Comparison

| | Optora | `skwdro` | `dro` (python-dro) |
| --- | --- | --- | --- |
| Latest release | [![optora on PyPI](https://img.shields.io/pypi/v/optora?style=flat-square&label=)](https://pypi.org/project/optora/) | [![skwdro on PyPI](https://img.shields.io/pypi/v/skwdro?style=flat-square&label=)](https://pypi.org/project/skwdro/) | [![dro on PyPI](https://img.shields.io/pypi/v/dro?style=flat-square&label=)](https://pypi.org/project/dro/) |
| License | MIT | BSD-3-Clause[^skwdro-repo] | MIT[^dro-pypi] |
| Runtime dependencies | `torch` only | torch, scikit-learn, cvxpy, cvxopt, dask and others[^skwdro-pypi] | torch, scikit-learn, cvxpy, xgboost, lightgbm and others[^dro-pypi] |
| Ambiguity sets | KL, general $\varphi$, $\chi^2$, total variation, CVaR, Wasserstein, Sinkhorn, MMD, group DRO | Wasserstein, entropically smoothed[^skwdro-why] | $\chi^2$, KL, CVaR, TV, Wasserstein, Sinkhorn, MMD, group DRO, holistic robust, outlier-robust Wasserstein and more, depending on the model family[^dro-repo] |
| Adversary support | finite, on the nominal's scenarios, with a user-given cost or kernel matrix | continuous, sampled by Monte Carlo around each data point[^skwdro-why] | depends on the method, with finite-sample convex programs for linear models and adversarial perturbations for neural networks[^dro-wnn] |
| Interface | a robust-risk operator on a loss tensor, for any model | scikit-learn estimators, and `robustify` to wrap a PyTorch loss[^skwdro-torch] | scikit-learn-style estimators for fixed model families (linear, kernel, tree ensembles, neural networks)[^dro-linear] |
| Inner solve | bracketed tilt, closed form, bisection or interior-point cone program, accurate to the working dtype | stochastic gradient on the smoothed dual[^skwdro-paper] | cvxpy with CLARABEL by default for linear models, and gradient methods for neural networks[^dro-repo] |
| Worst-case distribution | every set, as the gradient of the worst case with respect to the loss | not documented | $\chi^2$, KL and Wasserstein estimators[^dro-worst] |
| Accelerators | device-agnostic tensors, with no host synchronization in the KL, $\chi^2$, TV, CVaR, Wasserstein and Sinkhorn sets | not documented | `device="cuda"` for neural and Sinkhorn models[^dro-nn] |
| Verification | independent NumPy, SciPy and Clarabel references, property-based tests, `gradcheck` and `gradgradcheck` | unit tests, linting, coverage[^skwdro-repo] | unit tests per method, coverage[^dro-tests] |

## What Optora does differently

<div class="grid cards" markdown>

-   :material-puzzle-outline:{ .lg .middle } __Composable contracts__

    ---

    The three abstract base classes are the API. A new divergence or
    ambiguity set is a subclass, and it works with every solver, estimator
    and training loop already in the library. There is no separate
    estimator class per model.

    [:octicons-arrow-right-24: Core abstractions](formulations.md#core-abstractions)

-   :material-check-decagram:{ .lg .middle } __Exact worst cases__

    ---

    On a finite support, each set solves its inner supremum to the precision
    of its working dtype, up to the exceptions listed on
    [Numerical behaviour](numerics.md). The default solves have no inner
    tolerance or step size to tune. The robust loss is the worst case
    itself, not a smoothed or sampled surrogate.

    [:octicons-arrow-right-24: Formulations](formulations.md#formulations-implemented-today)

-   :material-function-variant:{ .lg .middle } __Correct gradients__

    ---

    First and second derivatives with respect to the loss, the nominal and
    the radius follow from the envelope theorem. The tests check them with
    `gradcheck` and `gradgradcheck` at smooth points, and with one-sided
    finite differences at kinks.

    [:octicons-arrow-right-24: Training guide](training.md)

-   :material-scale-balance:{ .lg .middle } __Independent oracles__

    ---

    The test suite compares every ambiguity set with a NumPy, SciPy or
    Clarabel reference that shares no code with Optora. Each reference is a
    linear or cone program, a closed form, or a separately minimized dual.
    The agreement shown below is measured on every documentation build.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md#against-independent-reference-solvers)

</div>

The statistics below are measured by this build of the documentation on a
CPU runner. Timings are in float32 and the agreement is in float64.
[Benchmarks](benchmarks.md) lists the environment and the full tables.

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
    - **General robust optimization models.** RSOME is designed for robust
      and distributionally robust linear or conic programs written in a
      modelling syntax and passed to external solvers[^rsome].

Optora is a good fit when you train your own PyTorch model and want the
exact worst case of a standard ambiguity set as a differentiable loss. It
also suits comparing several geometries under one interface, or building a
new set on top of tested contracts.

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
