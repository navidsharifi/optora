"""Minibatch KL-DRO against the full-batch objective on the same problem.

Every earlier example hands the ambiguity set the whole dataset. Training at
scale cannot: each step sees a minibatch of `m` samples, and running the
ambiguity set on that minibatch alone, the *plug-in* estimate, is biased. The
worst case is a nonlinear functional of the sample, so its expectation over
minibatches is not the full-batch worst case, and its gradient is not an
unbiased stochastic gradient of the full-batch objective (Levy et al. 2020).
For KL-DRO the plug-in is the log-sum-exp of a subsample, and it is biased
*low*: a small batch rarely contains the rare, high-loss points the
adversary would upweight.

The problem. A dataset of `N = 4096` points from the two-group regression of
example 07 (90% with slope `(1, 0)`, 10% with slope `(1, 1.5)`, noise 0.3),
a linear model with a bias, squared loss, and a KL ball. Minibatches are drawn
with replacement from the dataset, so they are i.i.d. draws from its
empirical distribution and the full-batch objective

    F(theta) = sup_{KL(q || uniform_N) <= radius} E_q[loss(theta)]

is *exactly* the population objective the minibatch estimators target. No
asymptotics are involved in the comparison: every number below is measured
against `F`, computed exactly by `KLAmbiguitySet` on all `N` points.

Part 1 fixes the model at its least-squares fit and measures the bias of each
estimator of `F`, as a fraction of the robust excess `F - mean(loss)`, over a
range of radii that includes the 95% calibrated radius `kl_radius(N, 0.95)`.
The estimators:

* plug-in on `m = 16, 64, 256` samples;
* `MultilevelMonteCarloEstimator` with base batch 16 and 4 levels: the bias
  of the 256-sample plug-in at an expected batch of about 68 samples
  (Blanchet and Glynn 2015; Levy et al. 2020);
* `StochasticPhiDual` on 64 samples, at the dual variables that are optimal
  for the full batch: there its minibatch mean is exactly `F`.

Each estimate is reported minus the mean loss of the same minibatch (for the
multilevel estimator, of its base level). That control variate has the
full-batch mean loss as its expectation, so it changes no bias and removes
most of the sampling noise of the mean itself.

Part 2 trains the model from a warm start (least squares on a pilot batch)
with `torch.optim.Adam` on each minibatch objective, over several seeds, and
tracks the suboptimality `F(theta_bar) - min F` of the averaged iterate
against a full-batch L-BFGS reference. The plug-in descends quickly and then
stalls at a floor set by its bias, not by its noise: more steps do not move
it. The multilevel estimator goes below that floor on a comparable number of
samples, at three exact solves per step. The stochastic dual has no inner
solve and no bias, but KL's exponential conjugate makes its gradients heavy
tailed, so it needs many more, much cheaper, steps, and a start for its dual
variables on the scale of the loss: here the exact dual optimum of the pilot
batch, which uses those samples alone.

References:
    Daniel Levy, Yair Carmon, John C. Duchi, and Aaron Sidford, "Large-Scale
    Methods for Distributionally Robust Optimization", NeurIPS 2020.
    Jose H. Blanchet and Peter W. Glynn, "Unbiased Monte Carlo for
    Optimization and Functions of Expectations via Multi-Level
    Randomization", Winter Simulation Conference 2015.
    Jikai Jin, Bohang Zhang, Haiyang Wang, and Liwei Wang, "Non-convex
    Distributionally Robust Optimization: Non-asymptotic Analysis",
    NeurIPS 2021.

Run:
    python examples/08_minibatch_dro_training.py
"""

import functools
import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import (
    KLAmbiguitySet,
    MultilevelMonteCarloEstimator,
    StochasticPhiDual,
    kl_conjugate,
    kl_radius,
)

SEED = 20261026
NUM_POINTS = 4096
MINORITY_SHARE = 0.1
NOISE = 0.3
MAJORITY_SLOPE = (1.0, 0.0)
MINORITY_SLOPE = (1.0, 1.5)

BIAS_RADII = (kl_radius(NUM_POINTS, 0.95), 0.01, 0.05, 0.2, 1.0)
PLUG_IN_SIZES = (16, 64, 256)
BASE_BATCH, MAX_LEVEL = 16, 4
DUAL_BATCH = 64
REPLICATIONS = 4000

TRAIN_RADIUS = 0.2
NUM_SEEDS = 8
LEARNING_RATE = 0.03
STEPS = {"plug-in (m = 64)": 800, "multilevel": 500, "stochastic dual": 3000}

Estimator = Callable[[torch.Tensor], torch.Tensor]


def make_dataset(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the two-group regression dataset.

    Args:
        generator: Seeded generator.

    Returns:
        The `(N, 3)` design matrix (two features and a bias column) and the
        `(N,)` targets, in float64.
    """
    minority = torch.rand(NUM_POINTS, generator=generator) < MINORITY_SHARE
    features = torch.randn(NUM_POINTS, 2, generator=generator, dtype=torch.float64)
    slope = torch.where(
        minority.unsqueeze(-1),
        torch.tensor(MINORITY_SLOPE, dtype=torch.float64),
        torch.tensor(MAJORITY_SLOPE, dtype=torch.float64),
    )
    noise = torch.randn(NUM_POINTS, generator=generator, dtype=torch.float64)
    targets = torch.sum(slope * features, dim=-1) + NOISE * noise
    design = torch.cat([features, torch.ones(NUM_POINTS, 1, dtype=torch.float64)], -1)
    return design, targets


def squared_loss(
    theta: torch.Tensor, design: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-sample squared loss of a batch of linear models.

    Args:
        theta: Parameters of shape `(..., 3)`.
        design: Design rows of shape `(..., m, 3)` or `(m, 3)`.
        targets: Targets of shape `(..., m)` or `(m,)`.

    Returns:
        Losses of shape `(..., m)`.
    """
    prediction = torch.sum(design * theta.unsqueeze(-2), dim=-1)
    return (prediction - targets) ** 2


def minimize(
    parameters: list[torch.Tensor], objective: Callable[[], torch.Tensor]
) -> None:
    """Minimize `objective` over `parameters` to float64 precision with L-BFGS."""
    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=500,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = objective()
        value.backward()
        return value

    optimizer.step(closure)


def relative_excess_bias(
    estimates: torch.Tensor, batch_means: torch.Tensor, excess: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the bias of an estimator as a fraction of the robust excess.

    Args:
        estimates: Estimates of shape `(radii, replications)`.
        batch_means: The control variate, of shape `(replications,)`, whose
            expectation is the full-batch mean loss.
        excess: True robust excess `F - mean(loss)` per radius.

    Returns:
        The relative bias and its Monte Carlo standard error, per radius.
    """
    excess_estimates = estimates - batch_means
    replications = excess_estimates.shape[-1]
    bias = (excess_estimates.mean(-1) - excess) / excess
    standard_error = excess_estimates.std(-1) / math.sqrt(replications) / excess
    return bias, standard_error


def estimator_bias(
    design: torch.Tensor, targets: torch.Tensor, generator: torch.Generator
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Measure every estimator's bias against the full-batch worst case.

    Args:
        design: Full design matrix.
        targets: Full targets.
        generator: Seeded generator for the minibatch draws.

    Returns:
        Relative bias and standard error per radius, keyed by estimator.
    """
    theta = torch.linalg.lstsq(design, targets.unsqueeze(-1)).solution.squeeze(-1)
    loss = squared_loss(theta, design, targets)
    radii = torch.tensor(BIAS_RADII, dtype=torch.float64).unsqueeze(-1)
    uniform = torch.full((NUM_POINTS,), 1.0 / NUM_POINTS, dtype=torch.float64)
    full_batch = KLAmbiguitySet(uniform, radius=radii.squeeze(-1))
    excess = full_batch.worst_case_expectation(loss) - loss.mean()

    def draw(size: int, count: int = REPLICATIONS) -> torch.Tensor:
        return loss[torch.randint(NUM_POINTS, (count, size), generator=generator)]

    results = {}
    for size in PLUG_IN_SIZES:
        batch = draw(size)
        plug_in = KLAmbiguitySet(
            torch.full((size,), 1.0 / size, dtype=torch.float64), radius=radii
        )
        results[f"plug-in (m = {size})"] = relative_excess_bias(
            plug_in.worst_case_expectation(batch), batch.mean(-1), excess
        )

    multilevel = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=radii),
        BASE_BATCH,
        MAX_LEVEL,
        dtype=torch.float64,
    )
    sizes = torch.tensor(
        [multilevel.sample_batch_size(generator) for _ in range(REPLICATIONS)]
    )
    estimates, base_means = [], []
    for size in sizes.unique().tolist():
        batch = draw(size, int((sizes == size).sum()))
        estimates.append(multilevel(batch))
        base_means.append(batch[..., :BASE_BATCH].mean(-1))
    results["multilevel"] = relative_excess_bias(
        torch.cat(estimates, dim=-1), torch.cat(base_means), excess
    )

    dual = StochasticPhiDual(
        radii.squeeze(-1), kl_conjugate, initial_lam=float(loss.mean())
    )
    minimize(list(dual.parameters()), lambda: dual(loss).sum())
    batch = draw(DUAL_BATCH)
    with torch.no_grad():
        dual_estimates = dual(batch.unsqueeze(-2)).transpose(0, 1)
    results["stochastic dual"] = relative_excess_bias(
        dual_estimates, batch.mean(-1), excess
    )
    return results


def pilot_dual_point(pilot_loss: torch.Tensor) -> tuple[float, float]:
    """Solve the KL dual exactly on the pilot batch.

    The stochastic dual's variables carry the loss's scale, and a start far
    from it makes KL's exponential conjugate produce rare, enormous
    gradients that stall Adam. The minibatch dual optimum of the pilot batch
    is computed from those samples alone, so it is a legitimate start.

    Args:
        pilot_loss: Losses of the pilot batch at the warm-start model.

    Returns:
        The pilot batch's optimal `(log_eta, lam)`.
    """
    dual = StochasticPhiDual(
        torch.tensor(TRAIN_RADIUS, dtype=torch.float64),
        kl_conjugate,
        initial_lam=float(pilot_loss.mean()),
    )
    minimize(list(dual.parameters()), lambda: dual(pilot_loss))
    return float(dual.log_eta.detach()), float(dual.lam.detach())


def train(
    method: str,
    design: torch.Tensor,
    targets: torch.Tensor,
    theta_start: torch.Tensor,
    dual_start: tuple[float, float],
    objective: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[list[int], torch.Tensor]:
    """Train `NUM_SEEDS` models with Adam on one minibatch objective.

    Training runs in float32, as a training loop would; the full-batch
    objective is evaluated in float64.

    Args:
        method: Key of `STEPS` naming the estimator.
        design: Full design matrix.
        targets: Full targets.
        theta_start: Warm start shared by every seed.
        dual_start: Starting `(log_eta, lam)` of the stochastic dual.
        objective: The full-batch objective `F`, on `(..., 3)` parameters.

    Returns:
        The number of samples drawn by each checkpoint, and `F` at the
        average of the second half of the iterates up to each checkpoint,
        of shape `(checkpoints, NUM_SEEDS)`.
    """
    generator = torch.Generator().manual_seed(SEED + len(method))
    design32, targets32 = design.float(), targets.float()
    theta = theta_start.float().expand(NUM_SEEDS, 3).clone().requires_grad_()
    parameters = [theta]
    if method == "stochastic dual":
        dual = StochasticPhiDual(
            torch.full((NUM_SEEDS,), TRAIN_RADIUS),
            kl_conjugate,
            initial_log_eta=dual_start[0],
            initial_lam=dual_start[1],
        )
        parameters += list(dual.parameters())
        estimate: Estimator = dual
    elif method == "multilevel":
        estimate = MultilevelMonteCarloEstimator(
            functools.partial(KLAmbiguitySet, radius=TRAIN_RADIUS),
            BASE_BATCH,
            MAX_LEVEL,
        )
    else:
        estimate = KLAmbiguitySet(
            torch.full((DUAL_BATCH,), 1.0 / DUAL_BATCH), radius=TRAIN_RADIUS
        ).worst_case_expectation

    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 / math.sqrt(1.0 + step / 100.0)
    )
    steps = STEPS[method]
    checkpoints = sorted({int(x) for x in torch.logspace(1, math.log10(steps), 25)})
    history = torch.empty(steps, NUM_SEEDS, 3, dtype=torch.float64)
    samples, drawn, gaps = [], 0, []
    for step in range(steps):
        if isinstance(estimate, MultilevelMonteCarloEstimator):
            size = estimate.sample_batch_size(generator)
        else:
            size = DUAL_BATCH
        index = torch.randint(NUM_POINTS, (NUM_SEEDS, size), generator=generator)
        loss = squared_loss(theta, design32[index], targets32[index])
        optimizer.zero_grad()
        estimate(loss).sum().backward()
        optimizer.step()
        schedule.step()
        history[step] = theta.detach().double()
        drawn += size
        if step + 1 in checkpoints:
            averaged = history[(step + 1) // 2 : step + 1].mean(0)
            samples.append(drawn)
            gaps.append(objective(averaged))
    return samples, torch.stack(gaps)


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    design, targets = make_dataset(generator)

    bias = estimator_bias(design, targets, generator)
    print(f"bias of the robust excess, as a fraction of it ({REPLICATIONS} draws)")
    print("  radius  " + "".join(f"{name:>22}" for name in bias))
    for index, radius in enumerate(BIAS_RADII):
        cells = "".join(
            f"{float(b[index]):>+13.3f} +- {float(s[index]):.3f}"
            for b, s in bias.values()
        )
        print(f"  {radius:<8.2g}{cells}")

    plug_in = [bias[f"plug-in (m = {size})"][0] for size in PLUG_IN_SIZES]
    largest_error = bias[f"plug-in (m = {PLUG_IN_SIZES[-1]})"][1]
    assert bool(torch.all(plug_in[0] < 0.0))
    assert bool(torch.all(plug_in[0] < plug_in[1])), "bias shrinks with m"
    assert bool(torch.all(plug_in[1] < plug_in[2])), "bias shrinks with m"
    multilevel_bias, multilevel_error = bias["multilevel"]
    combined_error = torch.sqrt(multilevel_error**2 + largest_error**2)
    assert bool(torch.all(torch.abs(multilevel_bias - plug_in[2]) < 4 * combined_error))
    dual_bias, dual_error = bias["stochastic dual"]
    assert bool(torch.all(torch.abs(dual_bias) < 4 * dual_error))

    uniform = torch.full((NUM_POINTS,), 1.0 / NUM_POINTS, dtype=torch.float64)
    full_batch = KLAmbiguitySet(uniform, radius=TRAIN_RADIUS)

    def objective(theta: torch.Tensor) -> torch.Tensor:
        return full_batch.worst_case_expectation(squared_loss(theta, design, targets))

    reference = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    minimize([reference], lambda: objective(reference))
    optimum = objective(reference.detach())
    (gradient,) = torch.autograd.grad(objective(reference), reference)
    assert float(gradient.norm()) < 1e-8

    pilot = torch.randint(
        NUM_POINTS, (DUAL_BATCH,), generator=torch.Generator().manual_seed(SEED)
    )
    theta_start = torch.linalg.lstsq(
        design[pilot], targets[pilot].unsqueeze(-1)
    ).solution.squeeze(-1)
    dual_start = pilot_dual_point(
        squared_loss(theta_start, design[pilot], targets[pilot])
    )
    least_squares = torch.linalg.lstsq(design, targets.unsqueeze(-1)).solution
    erm_gap = float(objective(least_squares.squeeze(-1)) - optimum)
    start_gap = float(objective(theta_start) - optimum)
    print(f"\nfull-batch optimum at radius {TRAIN_RADIUS}: F* = {float(optimum):.5f}")
    print(f"  F(least squares) - F* = {erm_gap:.4f}")
    print(f"  F(warm start) - F*    = {start_gap:.4f}")

    curves = {}
    print("\nsuboptimality F(theta_bar) - F* of the averaged iterate, over seeds")
    print(f"  {'method':<20}{'steps':>7}{'samples':>10}{'mean':>10}{'max':>10}")
    for method in STEPS:
        samples, values = train(
            method, design, targets, theta_start, dual_start, objective
        )
        gaps = values - optimum
        assert bool(torch.all(gaps > -1e-9)), method
        curves[method] = (samples, gaps)
        final = gaps[-1]
        print(
            f"  {method:<20}{STEPS[method]:>7}{samples[-1]:>10}"
            f"{float(final.mean()):>10.4f}{float(final.max()):>10.4f}"
        )

    # Every seed stalls at nearly the same suboptimality: a bias floor, not
    # noise, since independent minibatch noise would spread the seeds out.
    plug_in_final = curves["plug-in (m = 64)"][1][-1]
    floor = float(plug_in_final.mean())
    assert float(plug_in_final.max() - plug_in_final.min()) < 0.5 * floor
    assert float(curves["multilevel"][1][-1].mean()) < 0.5 * floor
    assert float(curves["stochastic dual"][1][-1].mean()) < floor

    colors = {name: f"C{index}" for index, name in enumerate(bias)}
    fig, (ax_bias, ax_train) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (values, errors) in bias.items():
        ax_bias.errorbar(
            BIAS_RADII,
            values,
            yerr=2 * errors,
            marker="o",
            capsize=3,
            color=colors[name],
            label=name,
        )
    ax_bias.axhline(0.0, color="black", linestyle="--")
    ax_bias.axvline(
        BIAS_RADII[0], color="gray", linestyle=":", label="kl_radius(N, 0.95)"
    )
    ax_bias.set_xscale("log")
    ax_bias.set_xlabel("radius")
    ax_bias.set_ylabel("bias / robust excess")
    ax_bias.set_title("minibatch estimates of the full-batch worst case")
    ax_bias.legend(fontsize="small")

    for method, (samples, gaps) in curves.items():
        mean = gaps.mean(-1)
        color = colors[method]
        ax_train.plot(samples, mean, color=color, label=method)
        ax_train.fill_between(
            samples, gaps.amin(-1), gaps.amax(-1), color=color, alpha=0.15
        )
    ax_train.axhline(floor, color="gray", linestyle=":", label="plug-in floor")
    ax_train.set_xscale("log")
    ax_train.set_yscale("log")
    ax_train.set_xlabel("samples drawn")
    ax_train.set_ylabel("F(averaged iterate) - min F")
    ax_train.set_title(f"training at radius {TRAIN_RADIUS} (range over seeds)")
    ax_train.legend(fontsize="small")

    fig.tight_layout()
    save_figure(fig, "08_minibatch_dro_training")


if __name__ == "__main__":
    main()
