r"""Entropic (Sinkhorn) relaxation of Wasserstein DRO.

The worst-case expectation over a Wasserstein ball on a finite support is
a linear program, and its value is a piecewise-linear function of the loss
vector. Along a smooth path of decisions the robust objective is therefore
piecewise smooth in the decision, with kinks wherever the optimal transport
plan changes. The kinks are not a numerical artifact. The minimizer of the
robust objective typically sits on one, because a maximum of smooth
functions is minimized where two of them cross. An outer method that
relies on a gradient-norm stopping test then never stops, and a fixed-step
gradient method oscillates around the kink indefinitely.

Sinkhorn DRO (Wang, Gao and Xie 2025) replaces the Wasserstein distance by
its entropic regularization with strength epsilon. On a finite support the
ball becomes a mixture of KL balls with one shared budget, one around each
row of the Gibbs kernel Q_ij proportional to exp(-c_ij / epsilon). Its
worst case is an exponential tilt, a smooth function of the loss whose
first and second derivatives are computed exactly. `SinkhornAmbiguitySet`
computes it with the same bracketed bisection as `KLAmbiguitySet`. This
script measures what the relaxation costs in accuracy and what it gains in
smoothness.

* As epsilon -> 0 the Sinkhorn worst case converges to the exact
  `WassersteinAmbiguitySet` value at the same radius, and so does the
  Sinkhorn minimizer. The script sweeps epsilon over three decades and
  records the gap at a fixed decision, the uniform gap over a grid of
  decisions, and the distance between the two minimizers.
* The Sinkhorn objective is differentiable everywhere. On a fine grid of
  decisions, the increments of its derivative between neighbouring points
  stay within the grid step times its largest second derivative, as the
  mean value theorem requires of a twice differentiable function. The
  Wasserstein derivative exceeds the bound its own curvature allows by an
  order of magnitude, because it jumps. The Sinkhorn second derivative is
  finite, with peaks that sharpen toward the Wasserstein kinks as epsilon
  falls.
* `MinimaxSolver` with a fixed-step `GradientDescent` converges on the
  Sinkhorn objective and reports it, while on the Wasserstein objective it
  exhausts its iteration budget without meeting the gradient tolerance.

One convention matters for the comparison. `SinkhornAmbiguitySet.radius` is
the transport budget in excess of the smallest Sinkhorn distance any
distribution can have from the nominal, so radius zero is the
kernel-smoothed nominal rather than the nominal itself. Under the counting
reference measure that smallest distance is minus the entropic bias,
epsilon * sum_i p_i log sum_k exp(-c_ik / epsilon), which is epsilon times
the expected log-partition of the kernel rows. The bias vanishes with
epsilon, so the same number serves as the radius of both sets in the
limit. The script prints it next to the gap.

The problem is the one of tutorial 11, namely forty log-normal demands with
equal weight, the ground cost |xi - xi'|, and the quadratic loss
(theta - xi)^2 of a scalar location decision. The loss is smooth in theta,
so every kink in the Wasserstein objective comes from the ambiguity set.

References:
    Jie Wang, Rui Gao and Yao Xie, "Sinkhorn Distributionally Robust
    Optimization", Operations Research, 2025 (arXiv:2109.11926).
    Marco Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
    Transport", NeurIPS, 2013.
    Peyman Mohajerin Esfahani and Daniel Kuhn, "Data-Driven
    Distributionally Robust Optimization Using the Wasserstein Metric",
    Mathematical Programming 171, 2018.

Run:
    python tutorials/12_sinkhorn_relaxation_of_wasserstein_dro.py
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.core import AmbiguitySet
from optora.dro import (
    MinimaxProblem,
    MinimaxSolver,
    SinkhornAmbiguitySet,
    WassersteinAmbiguitySet,
)
from optora.solvers import GradientDescent

SEED = 20261103
NUM_SCENARIOS = 40
RADIUS = 0.1
DECISION = 1.0
EPSILONS = torch.logspace(0.0, -3.0, 13, dtype=torch.float64)
# The value gap and the exact suboptimality of the relaxed decision are
# asserted linear in epsilon from here down, with these constants. Above it
# the kernel is so flat that the smoothed centre of the ball dominates.
LINEAR_FROM, VALUE_RATE, DECISION_RATE = 0.1, 0.15, 0.05
SMOOTH_EPSILONS = (0.1, 0.03, 0.01)
GRID_POINTS = 801
BISECTION_STEPS = 50
OUTER_STEP, OUTER_ITERATIONS, OUTER_TOLERANCE = 0.2, 400, 1e-8
TRAJECTORY_STEPS = 60
INITIAL_DECISION = 2.0


def quadratic(theta: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Squared error of a location decision `theta` of shape `(..., 1)`."""
    return (theta - demand) ** 2


def sample_demand() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uniform nominal over a sorted log-normal demand sample."""
    generator = torch.Generator().manual_seed(SEED)
    demand = torch.exp(
        0.5 * torch.randn(NUM_SCENARIOS, generator=generator, dtype=torch.float64)
    )
    demand = torch.sort(demand).values
    return torch.full_like(demand, 1.0 / NUM_SCENARIOS), demand


def entropic_bias(nominal: torch.Tensor, cost: torch.Tensor, epsilon: float) -> float:
    r"""Return $\epsilon \sum_i p_i \log \sum_k e^{-c_{ik}/\epsilon}$.

    This is minus the smallest Sinkhorn distance from the nominal under the
    counting reference measure, the amount `SinkhornAmbiguitySet.radius`
    leaves out of its budget.
    """
    return epsilon * float(nominal @ torch.logsumexp(-cost / epsilon, dim=-1))


def derivatives(
    ambiguity_set: AmbiguitySet, theta: torch.Tensor, demand: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the robust objective and its first two derivatives at `theta`.

    Both derivatives come from autograd. The first is exact for either set
    by the envelope theorem. For the Sinkhorn set the second is exact as
    well. For the Wasserstein set autograd differentiates the active
    transport plan, so it reports the curvature of the loss and misses the
    jumps of the first derivative, which have no finite second derivative.

    Args:
        ambiguity_set: Wasserstein or Sinkhorn ball over the demands.
        theta: Decisions of shape `(G,)`.
        demand: Scenarios of shape `(n,)`.

    Returns:
        The value, first and second derivative, each of shape `(G,)`.
    """
    theta = theta.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(quadratic(theta[:, None], demand))
    (first,) = torch.autograd.grad(value.sum(), theta, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), theta)
    return value.detach(), first.detach(), second


def robust_slope(
    ambiguity_set: AmbiguitySet, theta: torch.Tensor, demand: torch.Tensor
) -> torch.Tensor:
    """Return the derivative of the robust objective at `theta`, shape `(G,)`."""
    theta = theta.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(quadratic(theta[:, None], demand))
    (first,) = torch.autograd.grad(value.sum(), theta)
    return first


def argmin_by_bisection(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    lower: float,
    upper: float,
) -> float:
    """Minimize a convex function by bisecting the sign of its derivative.

    Fifty halvings of the demand range locate the minimizer to about
    1e-15, also where the derivative jumps, so the result serves as the
    reference minimizer of the nonsmooth objective.
    """
    low = torch.tensor(lower, dtype=torch.float64)
    high = torch.tensor(upper, dtype=torch.float64)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if derivative(middle[None])[0] > 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * float(low + high)


def robust_minimizer(ambiguity_set: AmbiguitySet, demand: torch.Tensor) -> float:
    """Return the minimizer of the robust objective over the demand range."""
    return argmin_by_bisection(
        lambda theta: robust_slope(ambiguity_set, theta, demand),
        float(demand.min()),
        float(demand.max()),
    )


def gradient_trajectory(
    ambiguity_set: AmbiguitySet, demand: torch.Tensor
) -> torch.Tensor:
    """Unroll fixed-step gradient descent and return `|dV/dtheta|` per step."""
    theta = torch.tensor([INITIAL_DECISION], dtype=torch.float64)
    norms = []
    for _ in range(TRAJECTORY_STEPS):
        gradient = robust_slope(ambiguity_set, theta, demand)
        norms.append(gradient.abs())
        theta = theta - OUTER_STEP * gradient
    return torch.cat(norms)


def main() -> None:
    nominal, demand = sample_demand()
    cost = (demand[:, None] - demand[None, :]).abs()
    wasserstein = WassersteinAmbiguitySet(nominal, cost, RADIUS)
    sinkhorn = {
        epsilon: SinkhornAmbiguitySet(nominal, cost, epsilon, RADIUS)
        for epsilon in EPSILONS.tolist()
    }
    smooth = {
        epsilon: SinkhornAmbiguitySet(nominal, cost, epsilon, RADIUS)
        for epsilon in SMOOTH_EPSILONS
    }
    theta = torch.tensor([DECISION], dtype=torch.float64)
    loss = quadratic(theta, demand)

    # 1. The worst-case value at one decision, as epsilon vanishes.
    exact = float(wasserstein.worst_case_expectation(loss))
    values = torch.tensor(
        [float(ball.worst_case_expectation(loss)) for ball in sinkhorn.values()]
    )
    biases = torch.tensor(
        [entropic_bias(nominal, cost, epsilon) for epsilon in sinkhorn]
    )
    gaps = (values - exact).abs()
    print(
        f"radius {RADIUS}, decision theta = {DECISION}: "
        f"exact Wasserstein worst case {exact:.10f}"
    )
    print("  epsilon     Sinkhorn value   |gap|       entropic bias")
    for epsilon, value, gap, bias in zip(
        sinkhorn, values.tolist(), gaps.tolist(), biases.tolist(), strict=True
    ):
        print(f"  {epsilon:9.3e}  {value:14.10f}   {gap:9.3e}   {bias:9.3e}")
    linear = EPSILONS <= LINEAR_FROM
    print(
        f"  gap / epsilon <= {float((gaps / EPSILONS)[linear].max()):.3f} for"
        f" epsilon <= {LINEAR_FROM}; the gap changes sign twice on the way down"
    )
    assert torch.all(gaps[linear] <= VALUE_RATE * EPSILONS[linear])
    assert gaps[-1] < 1e-4 and biases[-1] < 1e-4

    # 2. Smoothness in the decision, on a grid over the demand range.
    grid = torch.linspace(
        float(demand.min()), float(demand.max()), GRID_POINTS, dtype=torch.float64
    )
    step = float(grid[1] - grid[0])
    curves: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
        "wasserstein": derivatives(wasserstein, grid, demand)
    }
    for epsilon in SMOOTH_EPSILONS:
        curves[f"sinkhorn {epsilon}"] = derivatives(smooth[epsilon], grid, demand)
    exact_value, exact_slope, _ = curves["wasserstein"]
    print(
        f"\nderivative dV/dtheta on {GRID_POINTS} grid points (step {step:.2e})"
        " over the demand range"
    )
    print(
        "  objective        largest increment   step * max V''   ratio"
        "   sup |V - V_W|   mean |V' - V_W'|"
    )
    excess = {}
    for name, (value, slope, curvature) in curves.items():
        increment = float((slope[1:] - slope[:-1]).abs().max())
        bound = step * float(curvature.max())
        excess[name] = increment / bound
        print(
            f"  {name:16s} {increment:17.4e}   {bound:14.4e}   {excess[name]:5.2f}"
            f"   {float((value - exact_value).abs().max()):13.3e}"
            f"   {float((slope - exact_slope).abs().mean()):16.3e}"
        )
    # A C^2 function moves its derivative by at most the step times its
    # largest curvature between neighbouring grid points, and attains that
    # bound where the curvature peaks are wide against the step. The
    # Wasserstein derivative exceeds it by an order of magnitude because it
    # jumps.
    assert excess["wasserstein"] > 5.0
    assert all(excess[f"sinkhorn {epsilon}"] <= 1.05 for epsilon in SMOOTH_EPSILONS)
    uniform = [
        float((curves[f"sinkhorn {epsilon}"][0] - exact_value).abs().max())
        for epsilon in SMOOTH_EPSILONS
    ]
    slope_gaps = [
        float((curves[f"sinkhorn {epsilon}"][1] - exact_slope).abs().mean())
        for epsilon in SMOOTH_EPSILONS
    ]
    assert uniform == sorted(uniform, reverse=True)
    assert slope_gaps == sorted(slope_gaps, reverse=True)

    # 3. The minimizers, and the exact robust value of the relaxed decision.
    exact_minimizer = robust_minimizer(wasserstein, demand)
    minimizers = torch.tensor(
        [robust_minimizer(ball, demand) for ball in sinkhorn.values()]
    )
    exact_at = lambda point: float(  # noqa: E731
        wasserstein.worst_case_expectation(
            quadratic(torch.tensor([point], dtype=torch.float64), demand)
        )
    )
    suboptimality = torch.tensor(
        [exact_at(point) - exact_at(exact_minimizer) for point in minimizers.tolist()]
    )
    left, right = (
        float(robust_slope(wasserstein, torch.tensor([point]), demand))
        for point in (exact_minimizer - 1e-7, exact_minimizer + 1e-7)
    )
    print(
        f"\nWasserstein minimizer {exact_minimizer:.8f}; its derivative jumps"
        f" from {left:+.4e} to {right:+.4e} across the minimizer"
    )
    print("  epsilon     Sinkhorn minimizer   distance    exact suboptimality")
    for epsilon, point, gap in zip(
        sinkhorn,
        minimizers.tolist(),
        suboptimality.tolist(),
        strict=True,
    ):
        print(
            f"  {epsilon:9.3e}  {point:18.8f}   {abs(point - exact_minimizer):9.3e}"
            f"   {gap:9.3e}"
        )
    assert left < 0.0 < right
    distances = (minimizers - exact_minimizer).abs()
    print(
        f"  exact suboptimality / epsilon <="
        f" {float((suboptimality / EPSILONS)[linear].max()):.3f} for"
        f" epsilon <= {LINEAR_FROM}; the exact objective rises"
        f" {abs(left) / right:.0f} times faster to the left of its minimizer"
        " than to the right"
    )
    assert torch.all(suboptimality[linear] <= DECISION_RATE * EPSILONS[linear])
    assert distances[-1] < 5e-3 and suboptimality[-1] < 1e-5

    # 4. A gradient-based outer solve through MinimaxSolver.
    solver = GradientDescent(
        step_size=OUTER_STEP,
        max_iter=OUTER_ITERATIONS,
        tol=OUTER_TOLERANCE,
        check_interval=1,
    )
    print(
        f"\nMinimaxSolver with GradientDescent(step_size={OUTER_STEP},"
        f" tol={OUTER_TOLERANCE}) from theta = {INITIAL_DECISION}"
    )
    outcomes = {}
    for name, ball in (
        ("wasserstein", wasserstein),
        *((f"sinkhorn {epsilon}", ball) for epsilon, ball in smooth.items()),
    ):
        result = MinimaxSolver(solver).solve(
            MinimaxProblem(
                ball,
                lambda point: quadratic(point, demand),
                torch.tensor([INITIAL_DECISION], dtype=torch.float64),
            )
        )
        outcomes[name] = result
        print(
            f"  {name:16s} converged {str(result.converged):5s} after"
            f" {result.num_iterations:4d} iterations at"
            f" theta = {float(result.point):.8f}"
        )
    assert not outcomes["wasserstein"].converged
    assert all(outcomes[f"sinkhorn {epsilon}"].converged for epsilon in SMOOTH_EPSILONS)
    trajectories = {
        "wasserstein": gradient_trajectory(wasserstein, demand),
        **{
            f"sinkhorn {epsilon}": gradient_trajectory(ball, demand)
            for epsilon, ball in smooth.items()
        },
    }
    tail = trajectories["wasserstein"][TRAJECTORY_STEPS // 2 :]
    print(
        f"  unrolled {TRAJECTORY_STEPS} steps: Wasserstein |dV/dtheta| stays between"
        f" {float(tail.min()):.1e} and {float(tail.max()):.1e} over the second half,"
        f" Sinkhorn {SMOOTH_EPSILONS[-1]} ends at"
        f" {float(trajectories[f'sinkhorn {SMOOTH_EPSILONS[-1]}'][-1]):.1e}"
    )
    assert float(tail.min()) > 1e-4
    assert all(
        float(trajectories[f"sinkhorn {epsilon}"][-1]) < 1e-8
        for epsilon in SMOOTH_EPSILONS
    )

    fig, (limit, slopes, descent) = plt.subplots(1, 3, figsize=(14.5, 4.4))
    epsilons = EPSILONS.numpy()
    limit.plot(epsilons, gaps.numpy(), "o-", color="C5", label="$|V_\\epsilon - V_W|$")
    limit.plot(
        epsilons,
        distances.numpy(),
        "s-",
        color="C5",
        markerfacecolor="none",
        label="$|\\theta_\\epsilon - \\theta_W|$",
    )
    limit.plot(
        epsilons,
        suboptimality.numpy(),
        "d-",
        color="C4",
        label="$V_W(\\theta_\\epsilon) - V_W(\\theta_W)$",
    )
    limit.plot(
        epsilons, biases.numpy(), ":", color="0.6", linewidth=1.0, label="entropic bias"
    )
    limit.set_xscale("log")
    limit.set_yscale("log")
    limit.invert_xaxis()
    limit.set_xlabel("entropic strength $\\epsilon$")
    limit.set_ylabel("gap to the exact Wasserstein solution")
    limit.set_title("the relaxation converges as $\\epsilon \\to 0$")
    limit.legend(loc="upper right", fontsize=7.5)

    slopes.plot(
        grid.numpy(),
        exact_slope.numpy(),
        color="C4",
        linewidth=1.2,
        label="Wasserstein (jumps)",
    )
    for epsilon, alpha in zip(SMOOTH_EPSILONS, (0.4, 0.7, 1.0), strict=True):
        slopes.plot(
            grid.numpy(),
            curves[f"sinkhorn {epsilon}"][1].numpy(),
            color="C5",
            alpha=alpha,
            label=f"Sinkhorn $\\epsilon$ = {epsilon}",
        )
    slopes.axhline(0.0, color="0.6", linewidth=0.8)
    slopes.axvline(exact_minimizer, color="0.6", linewidth=0.8, linestyle="--")
    window = 0.35
    slopes.set_xlim(exact_minimizer - window, exact_minimizer + window)
    inside = (grid > exact_minimizer - window) & (grid < exact_minimizer + window)
    slopes.set_ylim(
        float(exact_slope[inside].min()) - 0.05, float(exact_slope[inside].max()) + 0.05
    )
    slopes.set_xlabel("decision $\\theta$")
    slopes.set_ylabel("$\\mathrm{d}V / \\mathrm{d}\\theta$")
    slopes.set_title("the Wasserstein derivative jumps, Sinkhorn's bends")
    slopes.legend(loc="upper left", fontsize=7.5)

    steps = range(1, TRAJECTORY_STEPS + 1)
    descent.plot(
        steps,
        trajectories["wasserstein"].numpy(),
        color="C4",
        label="Wasserstein",
    )
    for epsilon, alpha in zip(SMOOTH_EPSILONS, (0.4, 0.7, 1.0), strict=True):
        descent.plot(
            steps,
            trajectories[f"sinkhorn {epsilon}"].numpy(),
            color="C5",
            alpha=alpha,
            label=f"Sinkhorn $\\epsilon$ = {epsilon}",
        )
    descent.axhline(
        OUTER_TOLERANCE, color="0.6", linewidth=0.8, linestyle=":", label="tolerance"
    )
    descent.set_yscale("log")
    descent.set_xlabel("gradient step")
    descent.set_ylabel("$|\\mathrm{d}V / \\mathrm{d}\\theta|$ at the iterate")
    descent.set_title(f"fixed step {OUTER_STEP}: only the relaxation converges")
    descent.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "12_sinkhorn_relaxation_of_wasserstein_dro")


if __name__ == "__main__":
    main()
