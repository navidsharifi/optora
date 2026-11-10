---
icon: lucide/play
---

# Examples

The scripts under `examples/` are small numerical experiments. Each one
tests a property that a formulation predicts, for example that the
worst-case expectation saturates at $\max_i \mathrm{loss}_i$, or that the
inner solve lands where a brute-force grid search over the dual variable
places the optimum. Each script runs end to end and prints the quantities
it checks, and it also saves a figure. Each page additionally shows an
animation, rendered from the library when the site is built, which
replays the sweep or the training run frame by frame.

## Running them

The scripts need `matplotlib` for plotting. It is not a runtime dependency
of `optora`, so it is installed through a separate extra.

```bash
pip install -e ".[examples]"
```

Each script then runs from the repository root.

```bash
python examples/01_kl_dro_radius_sweep.py
```

!!! note "Where the figures go"

    Every script writes its figure to `examples/outputs/`, which git
    ignores. No figure or animation is committed to this repository. The
    figures on the pages below are produced when the site is built.
    `python -m tools.figures` runs every script under the Matplotlib style
    of the site, once per colour scheme, and copies the result next to the
    page. The reasons for this policy are given in
    [`examples/README.md`](https://github.com/navidsharifi/optora/blob/main/examples/README.md).

## The experiments

| Example | Question it answers |
| --- | --- |
| [KL-DRO radius sweep](kl_dro_radius_sweep.md) | How does the worst-case expectation grow with the ambiguity radius, and does it respect both closed-form limits? |
| [Worst-case distribution shift](worst_case_distribution_shift.md) | Where does the adversary put the probability mass? |
| [Robust decisions across geometries](robust_decision_across_ambiguity_sets.md) | How much does the choice of ambiguity set change the resulting decision? |
| [Convergence diagnostics](convergence_diagnostics.md) | Have the outer and the inner solve both converged, and how can each be verified independently? |
| [Wasserstein-DRO and Lipschitz regularization](wasserstein_lipschitz_equivalence.md) | In what precise sense is Wasserstein-DRO equivalent to a Lipschitz penalty? |
| [Calibrating the ambiguity radius](calibrated_ambiguity_radius.md) | Which radius should we use, and does the finite-sample theory behind that choice hold at the available sample size? |
| [Training under subpopulation shift](subpopulation_shift_training.md) | Does a DRO-trained model have lower risk than an ERM model when the test mixture of subpopulations shifts, and where is it worse? |
| [Minibatch DRO against the full batch](minibatch_dro_training.md) | How biased is the worst case of a minibatch, and do the multilevel and stochastic-dual estimators train to the full-batch optimum instead? |
| [Group DRO under a spurious correlation](group_dro_training.md) | Does training on the worst group correct a model that relies on a spurious feature, and why does it fail without regularization? |
| [Recovering CVaR from a phi-divergence ball](cvar_from_phi_divergence.md) | Which phi-divergence ball has CVaR as its worst case, and why does a total-variation ball of radius $(1-\alpha)/\alpha$ not? |
| [Robust regret against robust loss](distributionally_robust_regret.md) | If the adversary is charged only for avoidable loss, how much less conservative is the robust decision, and does it agree with ERM to first order? |
| [Sinkhorn relaxation of Wasserstein DRO](sinkhorn_relaxation_of_wasserstein_dro.md) | How close does the entropic (Sinkhorn) ball get to the exact Wasserstein worst case as $\epsilon \to 0$, and how does its smoothness help a gradient-based outer solve? |

The examples are ordered roughly by how much of the library they use. The
first one uses a single [`AmbiguitySet`](../api/core/dro_base.md). The
fourth unrolls a solver to inspect its trajectory, and the fifth tests the
dual formula itself. The sixth calibrates the radius instead of treating
it as a free parameter. The seventh trains a model with `torch.optim` and
compares the result with an ERM baseline and two controls. The eighth
trains on minibatches and compares every estimator with the full-batch
objective it estimates, and the ninth trains on the worst of several
groups instead of their average. The tenth connects DRO to a classical
risk measure by rebuilding CVaR from the general phi-divergence dual and
checking it against the definition. The eleventh changes the payoff of the
adversary from loss to regret and measures how far this moves the robust
decision. The twelfth replaces the Wasserstein ball with its entropic
relaxation and measures both the accuracy lost to the smoothing and the
effect of the smoothing on an outer gradient method.

!!! warning "Run times and reduced iteration budgets"

    Examples 3 and 4 nest an outer gradient descent around the inner solve
    of an ambiguity set, and in eager PyTorch this composition multiplies
    the iteration counts. The outer budgets in those scripts are therefore
    much smaller than the defaults of the solver. They are chosen to show
    qualitative behaviour (monotonicity, limits and the ordering of the
    decisions) rather than to reach tight tolerances. Example 6 is slow for
    a different reason. Estimating a coverage probability needs thousands
    of independent inner solves rather than deeper ones. Example 7 takes
    about ten seconds on a CPU because it trains a model, batching all
    seeds and radii into one joint solve per step, and then verifies the
    trained models against closed-form and L-BFGS references. Example 8
    takes 20 to 30 seconds, because its training loops run thousands of
    small inner solves whose cost is dominated by kernel-launch overhead.
    Example 9 takes about ten seconds because it trains twenty
    over-parameterized models jointly. Example 10 also takes about ten
    seconds, almost all of it in the staged subgradient solve of a
    nonsmooth dual. Example 11 takes about twenty-five seconds. Every outer
    bisection step of the regret policy requires several inner Wasserstein
    solves, and a single such solve has a fixed overhead of a few
    milliseconds regardless of its batch size. Example 12 takes about
    twenty seconds. Most of this time is spent in the bisections that
    locate thirteen Sinkhorn minimizers and in the four hundred outer steps
    given to the Wasserstein objective, which does not converge within
    them.

## The shared plotting helper

One Python module in the folder is not an example. It saves every figure
to a common location, so that no script has to handle output paths
itself.

```py title="examples/_plotting.py"
--8<-- "examples/_plotting.py"
```
