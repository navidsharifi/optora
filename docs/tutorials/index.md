---
icon: lucide/graduation-cap
---

# Tutorials

The scripts under `tutorials/` are small numerical experiments. Each one
tests a property that a formulation predicts, for example that the
worst-case expectation saturates at $\max_i \mathrm{loss}_i$, or that the
inner solve lands where a brute-force grid search over the dual variable
places the optimum. Each script runs end to end and prints the quantities
it checks, and it also saves a figure. Each page additionally shows an
animation, rendered from the library when the site is built, which
replays the sweep or the training run frame by frame.

A tutorial combines several parts of the library to answer one question
and explains the result in prose. For a short script that shows a single
component and nothing else, see the [Examples](../examples/index.md).

## Running them

The scripts need `matplotlib` for plotting. It is not a runtime dependency
of `optora`, so it is installed through a separate extra.

```bash
pip install -e ".[tutorials]"
```

Each script then runs from the repository root.

```bash
python tutorials/01_kl_dro_radius_sweep.py
```

!!! note "Where the figures go"

    Every script writes its figure to `tutorials/outputs/`, which git
    ignores. No figure or animation is committed to this repository. The
    figures on the pages below are produced when the site is built.
    `python -m tools.figures` runs every script under the Matplotlib style
    of the site, once per colour scheme, and copies the result next to the
    page. The reasons for this policy are given in
    [`tutorials/README.md`](https://github.com/navidsharifi/optora/blob/main/tutorials/README.md).

## The tutorials

The tutorials fall into four groups, in the order of the sidebar. The
script numbers keep the order in which the tutorials were written, and the
pages refer to each other by those numbers.

### Ambiguity sets and worst cases

These tutorials study the inner problem on its own. There is a fixed loss
vector and no model, and the question is what the worst case over a given
ball is and where it puts its mass.

| Tutorial | Question it answers |
| --- | --- |
| [01 KL-DRO radius sweep](kl_dro_radius_sweep.md) | How does the worst-case expectation grow with the ambiguity radius, and does it respect both closed-form limits? |
| [02 Worst-case distribution shift](worst_case_distribution_shift.md) | Where does the adversary put the probability mass? |
| [10 Recovering CVaR from a phi-divergence ball](cvar_from_phi_divergence.md) | Which phi-divergence ball has CVaR as its worst case, and why does a total-variation ball of radius $(1-\alpha)/\alpha$ not? |
| [03 Robust decisions across geometries](robust_decision_across_ambiguity_sets.md) | How much does the choice of ambiguity set change the resulting decision? |

### Wasserstein and optimal transport

These tutorials use a ground cost on the support, so the adversary can
move mass between nearby scenarios instead of only reweighting them.

| Tutorial | Question it answers |
| --- | --- |
| [05 Wasserstein-DRO and Lipschitz regularization](wasserstein_lipschitz_equivalence.md) | In what precise sense is Wasserstein-DRO equivalent to a Lipschitz penalty? |
| [12 Sinkhorn relaxation of Wasserstein DRO](sinkhorn_relaxation_of_wasserstein_dro.md) | How close does the entropic (Sinkhorn) ball get to the exact Wasserstein worst case as $\epsilon \to 0$, and how does its smoothness help a gradient-based outer solve? |
| [11 Robust regret against robust loss](distributionally_robust_regret.md) | If the adversary is charged only for avoidable loss, how much less conservative is the robust decision, and does it agree with ERM to first order? |

### Radius and convergence

Two questions come up in every application, namely which radius to use
and whether both the outer and the inner solve have converged.

| Tutorial | Question it answers |
| --- | --- |
| [06 Calibrating the ambiguity radius](calibrated_ambiguity_radius.md) | Which radius should we use, and does the finite-sample theory behind that choice hold at the available sample size? |
| [04 Convergence diagnostics](convergence_diagnostics.md) | Have the outer and the inner solve both converged, and how can each be verified independently? |

### Training models

These tutorials train a model with `torch.optim` and compare the DRO
objective with an ERM baseline.

| Tutorial | Question it answers |
| --- | --- |
| [07 Training under subpopulation shift](subpopulation_shift_training.md) | Does a DRO-trained model have lower risk than an ERM model when the test mixture of subpopulations shifts, and where is it worse? |
| [08 Minibatch DRO against the full batch](minibatch_dro_training.md) | How biased is the worst case of a minibatch, and do the multilevel and stochastic-dual estimators train to the full-batch optimum instead? |
| [09 Group DRO under a spurious correlation](group_dro_training.md) | Does training on the worst group correct a model that relies on a spurious feature, and why does it fail without regularization? |

!!! warning "Run times and reduced iteration budgets"

    Tutorials 3 and 4 nest an outer gradient descent around the inner
    solve of an ambiguity set, and in eager PyTorch this composition
    multiplies the iteration counts. The outer budgets in those scripts
    are therefore much smaller than the defaults of the solver. They are
    chosen to show qualitative behaviour (monotonicity, limits and the
    ordering of the decisions) rather than to reach tight tolerances.
    Tutorial 6 is slow for a different reason. Estimating a coverage
    probability needs thousands of independent inner solves rather than
    deeper ones. Tutorial 7 takes about ten seconds on a CPU because it
    trains a model, batching all seeds and radii into one joint solve per
    step, and then verifies the trained models against closed-form and
    L-BFGS references. Tutorial 8 takes 20 to 30 seconds, because its
    training loops run thousands of small inner solves whose cost is
    dominated by kernel-launch overhead. Tutorial 9 takes about ten
    seconds because it trains twenty over-parameterized models jointly.
    Tutorial 10 also takes about ten seconds, almost all of it in the
    staged subgradient solve of a nonsmooth dual. Tutorial 11 takes about
    twenty-five seconds. Every outer bisection step of the regret policy
    requires several inner Wasserstein solves, and a single such solve has
    a fixed overhead of a few milliseconds regardless of its batch size.
    Tutorial 12 takes about twenty seconds. Most of this time is spent in
    the bisections that locate thirteen Sinkhorn minimizers and in the
    four hundred outer steps given to the Wasserstein objective, which
    does not converge within them.

## The shared plotting helper

One Python module in the folder is not a tutorial. It saves every figure
to a common location, so that no script has to handle output paths
itself.

```py title="tutorials/_plotting.py"
--8<-- "tutorials/_plotting.py"
```
