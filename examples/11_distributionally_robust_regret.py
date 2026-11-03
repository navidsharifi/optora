r"""Distributionally robust regret against distributionally robust loss.

Wasserstein DRO guards against the worst distribution in a ball around the
sample, and the price is conservatism: the adversary is credited for every
unit of loss it can create, including loss that no decision could have
avoided. Fiechtner and Blanchet (2025) replace the worst-case loss by the
worst-case regret. The regret of a decision theta under a distribution P is
its expected loss in excess of the best decision for that distribution,

    R(theta, P) = E_P[l(theta, xi)] - min_beta E_P[l(beta, xi)],

and distributionally robust regret optimization (DRRO) minimizes
sup_P R(theta, P) over the Wasserstein ball. The benchmark beta is chosen
knowing P but before xi is drawn, which the paper calls ex-ante regret.
Because the inner minimum is itself a supremum of linear functionals,

    sup_P R(theta, P) = sup_beta sup_P E_P[l(theta, xi) - l(beta, xi)],

and the inner supremum is a worst-case expectation of an ordinary
per-scenario loss, the difference l(theta, .) - l(beta, .). That is what
`WassersteinAmbiguitySet.worst_case_expectation` computes, so the only new
ingredient is the outer maximization over the benchmark.

The paper's two main results are what this script measures. With a smooth
loss and a unique empirical-risk minimizer theta_0, the DRRO decision is
theta_0 + o(delta) and the robust regret is kappa delta^2 + o(delta^2),
whereas the Wasserstein DRO decision moves by O(delta) (Theorem 3). For a
strictly convex quadratic loss the empirical-risk decision is DRRO-optimal
at every radius (Theorem 4). Both are checked on a sample of forty
log-normal demands, with three losses:

* a quadratic loss (theta - xi)^2, the Theorem 4 case, where the regret
  under P is (theta - E_P[xi])^2 and the robust regret has a closed form;
* the LINEX loss exp(a(theta - xi)) - a(theta - xi) - 1 (Varian 1975), a
  smooth, asymmetric loss outside the quadratic family, for Theorem 3;
* the newsvendor loss b theta - s min(theta, xi) of the paper's own
  example, piecewise linear, where the smoothness assumption fails and the
  decisions move in steps between scenarios.

The script also tests the simpler per-scenario (ex-post) regret,
l(theta, xi) - min_beta l(beta, xi), because it is a plain wrapper around
the loss and plugs into `MinimaxSolver` unchanged. For the first two
losses every scenario can be fit exactly, the hindsight term is zero, and
ex-post DRRO is Wasserstein DRO under another name. For the newsvendor the
hindsight is -(s - b) xi, and subtracting it turns the profit form of the
loss into the classical cost form b (theta - xi)_+ + (s - b)(xi - theta)_+.

Ground costs. The estimation losses use |xi - xi'|, under which the
reachable means of the ball form the interval [mean - delta, mean + delta]
and the quadratic result is exact. The newsvendor uses (xi - xi')^2, which
makes many short moves cheaper than one long one; with the absolute cost
the adversary's best moves shuffle mass among low-demand scenarios without
crossing the order quantity, and no decision changes until the radius is
large.

Solvers. Every decision is scalar and every objective is convex in it, so
each outer problem is solved by bisection on the sign of its derivative,
which is exact to floating-point precision even at a kink. The derivative
of a worst-case expectation is the expected derivative of the loss under
the worst-case distribution, which autograd returns. The benchmark
maximization starts from the best grid point on either side of theta and
alternates best responses: the adversary answers beta with its worst-case
distribution, beta answers with the empirical-risk minimizer under it.
For the newsvendor the candidates are the scenarios themselves, which is
exact for a piecewise-linear loss. Each inner solution is verified against
a dense benchmark grid, and the Wasserstein DRO and ex-post decisions are
cross-checked against `MinimaxSolver`.

References:
    Lukas-Benedikt Fiechtner and Jose Blanchet, "Wasserstein
    Distributionally Robust Regret Optimization", arXiv:2504.10796, 2025.
    Hal R. Varian, "A Bayesian Approach to Real Estate Assessment", in
    Studies in Bayesian Econometrics and Statistics in Honor of Leonard J.
    Savage, North-Holland, 1975.
    Peyman Mohajerin Esfahani and Daniel Kuhn, "Data-Driven
    Distributionally Robust Optimization Using the Wasserstein Metric",
    Mathematical Programming 171, 2018.

Run:
    python examples/11_distributionally_robust_regret.py
"""

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import MinimaxProblem, MinimaxSolver, WassersteinAmbiguitySet
from optora.solvers import GradientDescent

SEED = 20261103
NUM_SCENARIOS = 40
LINEX_SHAPE = 1.5
ORDER_COST, SALE_PRICE = 1.0, 3.0
RATE_RADII = torch.logspace(-3.0, -1.5, 7, dtype=torch.float64)
# Slopes are read where the asymptotics apply. At the largest radii the pair
# of scenarios the Wasserstein adversary trades between changes, and the
# constant in front of the radius changes with it.
SLOPE_RADII = RATE_RADII <= 1e-2
NEWSVENDOR_RADII = torch.logspace(-4.0, -0.5, 8, dtype=torch.float64)
BENCHMARK_GRID = 9
BEST_RESPONSE_ROUNDS = 2
BISECTION_STEPS = 34
DENSE_GRID = 200
EXACT_TOLERANCE = 1e-9
CROSS_CHECK_RADIUS = 6
CROSS_CHECK_SOLVER = GradientDescent(
    step_size=1e-2, max_iter=150, tol=1e-9, check_interval=1
)

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def quadratic(theta: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Squared error of a location estimate `theta` of shape `(..., 1)`."""
    return (theta - demand) ** 2


def linex(theta: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """LINEX loss, exponential on one side and linear on the other."""
    gap = LINEX_SHAPE * (theta - demand)
    return torch.exp(gap) - gap - 1.0


def newsvendor(theta: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Newsvendor loss in profit form: order cost minus sales revenue."""
    return ORDER_COST * theta - SALE_PRICE * torch.minimum(theta, demand)


@dataclass(frozen=True)
class Problem:
    """One decision problem: a loss, its ground cost and the radii to sweep.

    Attributes:
        name: Label used in the output and the figure.
        loss: Per-scenario loss of a decision of shape `(..., 1)` against
            the demand vector, returning shape `(..., n)`.
        cost_power: Exponent of `|xi - xi'|` in the ground cost.
        radii: Wasserstein radii of the sweep.
        benchmarks: Candidate benchmark decisions the regret maximization
            starts from, or `None` for an even grid over the demand range.
        refine: Whether to improve the best candidates by best-response
            rounds. Off when the candidates already contain the maximizer,
            as the scenarios do for a piecewise-linear loss.
    """

    name: str
    loss: Loss
    cost_power: float
    radii: torch.Tensor
    benchmarks: torch.Tensor | None = None
    refine: bool = True


@dataclass(frozen=True)
class Policies:
    """Decisions of the four policies at every radius, shape `(R,)` each."""

    erm: torch.Tensor
    wdro: torch.Tensor
    ex_post: torch.Tensor
    ex_ante: torch.Tensor


def sample_demand() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uniform nominal over a sorted log-normal demand sample."""
    generator = torch.Generator().manual_seed(SEED)
    demand = torch.exp(
        0.5 * torch.randn(NUM_SCENARIOS, generator=generator, dtype=torch.float64)
    )
    demand = torch.sort(demand).values
    return torch.full_like(demand, 1.0 / NUM_SCENARIOS), demand


def regret_loss_fn(
    loss_fn: Callable[[torch.Tensor], torch.Tensor], hindsight: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a per-scenario loss into its ex-post regret, for `MinimaxProblem`.

    The hindsight term `min_beta l(beta, xi_i)` does not depend on the
    decision, so the wrapper only shifts each scenario; what it changes is
    where the adversary finds it worthwhile to move mass.
    """
    return lambda theta: loss_fn(theta) - hindsight


def argmin_by_bisection(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    steps: int = BISECTION_STEPS,
) -> torch.Tensor:
    """Minimize convex functions by bisecting the sign of their derivative.

    Args:
        derivative: Maps a batch of points to a (sub)derivative of each
            objective at that point, same shape.
        lower: Lower ends of the brackets.
        upper: Upper ends of the brackets, same shape as `lower`.
        steps: Halvings; forty take a unit bracket below $10^{-12}$.

    Returns:
        The bracket midpoints after `steps` halvings.
    """
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        positive = derivative(middle) > 0.0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    return 0.5 * (lower + upper)


def derivative_of(
    objective: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the autograd derivative of a batched scalar objective."""

    def derivative(theta: torch.Tensor) -> torch.Tensor:
        theta = theta.detach().requires_grad_()
        (gradient,) = torch.autograd.grad(objective(theta).sum(), theta)
        return gradient

    return derivative


def hindsight_losses(loss: Loss, demand: torch.Tensor) -> torch.Tensor:
    """Return `min_beta l(beta, xi_i)` for every scenario."""
    margin = demand.max() - demand.min()
    best = argmin_by_bisection(
        derivative_of(lambda beta: torch.diagonal(loss(beta.unsqueeze(-1), demand))),
        torch.full_like(demand, demand.min().item() - margin),
        torch.full_like(demand, demand.max().item() + margin),
        steps=60,
    )
    return torch.diagonal(loss(best.unsqueeze(-1), demand)).detach()


def best_response(
    loss: Loss, demand: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return the empirical-risk minimizer of `loss` under each distribution.

    Args:
        loss: Per-scenario loss.
        demand: Scenarios, shape `(n,)`.
        weights: Distributions over the scenarios, shape `(..., n)`.

    Returns:
        The minimizing decision per distribution, shape `(...)`.
    """
    batch = weights.shape[:-1]
    return argmin_by_bisection(
        derivative_of(
            lambda beta: (weights * loss(beta.unsqueeze(-1), demand)).sum(-1)
        ),
        torch.full(batch, demand.min().item(), dtype=demand.dtype),
        torch.full(batch, demand.max().item(), dtype=demand.dtype),
        steps=32,
    )


def worst_case(
    ambiguity_set: WassersteinAmbiguitySet, loss: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the worst-case expectation of `loss` and the maximizing distribution.

    The maximizer is the gradient of the worst case with respect to the
    loss (Danskin's theorem), so autograd returns it with the value.
    """
    loss = loss.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(loss)
    (adversary,) = torch.autograd.grad(value.sum(), loss)
    return value.detach(), adversary


def robust_regret(
    ambiguity_set: WassersteinAmbiguitySet,
    loss: Loss,
    demand: torch.Tensor,
    theta: torch.Tensor,
    benchmarks: torch.Tensor,
    rounds: int = BEST_RESPONSE_ROUNDS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the ex-ante robust regret of `theta` at every radius.

    Evaluates $\sup_\beta \sup_P \mathbb{E}_P[\ell(\theta) - \ell(\beta)]$:
    the inner supremum is a worst-case expectation of the loss difference,
    the outer one starts from the best candidate in `benchmarks` on either
    side of `theta` and alternates best responses until the pair
    $(\beta, P)$ is stationary. Each round can only raise the value.

    Args:
        ambiguity_set: Wasserstein ball with one radius per entry of
            `theta`, broadcast against a trailing benchmark dimension.
        loss: Per-scenario loss.
        demand: Scenarios, shape `(n,)`.
        theta: Decisions, one per radius, shape `(R,)`.
        benchmarks: Candidate benchmark decisions, shape `(K,)`.
        rounds: Best-response rounds after the candidate scan.

    Returns:
        The robust regret, the maximizing benchmark and the worst-case
        distribution, of shapes `(R,)`, `(R,)` and `(R, n)`.
    """
    base = loss(theta[:, None, None], demand)
    candidates = ambiguity_set.worst_case_expectation(
        base - loss(benchmarks[None, :, None], demand)
    )
    below = torch.where(benchmarks <= theta[:, None], candidates, -torch.inf)
    above = torch.where(benchmarks >= theta[:, None], candidates, -torch.inf)
    beta = torch.stack([benchmarks[below.argmax(-1)], benchmarks[above.argmax(-1)]], -1)
    for _ in range(rounds):
        _, adversary = worst_case(
            ambiguity_set, base - loss(beta.unsqueeze(-1), demand)
        )
        beta = best_response(loss, demand, adversary)
    value, adversary = worst_case(
        ambiguity_set, base - loss(beta.unsqueeze(-1), demand)
    )
    side = value.argmax(-1, keepdim=True)
    return (
        value.gather(-1, side).squeeze(-1),
        beta.gather(-1, side).squeeze(-1),
        adversary.gather(-2, side.unsqueeze(-1).expand(-1, 1, demand.numel())).squeeze(
            -2
        ),
    )


def solve_policies(
    problem: Problem, nominal: torch.Tensor, demand: torch.Tensor
) -> tuple[Policies, WassersteinAmbiguitySet, torch.Tensor]:
    """Solve ERM, Wasserstein DRO, ex-post DRRO and ex-ante DRRO at every radius."""
    cost = (demand[:, None] - demand[None, :]).abs() ** problem.cost_power
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost, radius=problem.radii[:, None]
    )
    benchmarks = (
        problem.benchmarks
        if problem.benchmarks is not None
        else torch.linspace(
            demand.min().item(), demand.max().item(), BENCHMARK_GRID, dtype=demand.dtype
        )
    )
    hindsight = hindsight_losses(problem.loss, demand)
    rounds = BEST_RESPONSE_ROUNDS if problem.refine else 0
    bracket = (
        torch.full_like(problem.radii, demand.min().item()),
        torch.full_like(problem.radii, demand.max().item()),
    )

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        return problem.loss(theta[:, None, None], demand)

    def robust_loss(theta: torch.Tensor) -> torch.Tensor:
        return ambiguity_set.worst_case_expectation(loss_fn(theta)).squeeze(-1)

    def robust_ex_post(theta: torch.Tensor) -> torch.Tensor:
        return ambiguity_set.worst_case_expectation(
            regret_loss_fn(loss_fn, hindsight)(theta)
        ).squeeze(-1)

    def ex_ante_derivative(theta: torch.Tensor) -> torch.Tensor:
        _, _, adversary = robust_regret(
            ambiguity_set, problem.loss, demand, theta, benchmarks, rounds
        )
        return derivative_of(
            lambda point: (adversary * problem.loss(point.unsqueeze(-1), demand)).sum(
                -1
            )
        )(theta)

    policies = Policies(
        erm=argmin_by_bisection(
            derivative_of(lambda theta: (nominal * loss_fn(theta)).sum(-1).squeeze(-1)),
            *bracket,
        ),
        wdro=argmin_by_bisection(derivative_of(robust_loss), *bracket),
        ex_post=argmin_by_bisection(derivative_of(robust_ex_post), *bracket),
        ex_ante=argmin_by_bisection(ex_ante_derivative, *bracket),
    )
    return policies, ambiguity_set, benchmarks


def cross_check_minimax(
    problem: Problem,
    nominal: torch.Tensor,
    demand: torch.Tensor,
    hindsight: torch.Tensor,
    policies: Policies,
) -> float:
    """Re-solve WDRO and ex-post DRRO with `MinimaxSolver` at one radius.

    One outer step costs one inner Wasserstein solve, whose fixed overhead
    dominates at this support size, so the check runs at a single radius
    where the two policies have moved well away from the empirical-risk
    decision. Both objectives are piecewise linear in the decision, so a
    fixed step oscillates around the minimizer by about `step_size` times
    the subgradient; the returned gap is read against that.

    Returns:
        The larger of the two differences from the bisection solutions.
    """
    index = CROSS_CHECK_RADIUS
    cost = (demand[:, None] - demand[None, :]).abs() ** problem.cost_power
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost, radius=problem.radii[index].item()
    )

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        return problem.loss(theta.unsqueeze(-1), demand)

    worst_gap = 0.0
    for wrapped, reference in (
        (loss_fn, policies.wdro[index]),
        (regret_loss_fn(loss_fn, hindsight), policies.ex_post[index]),
    ):
        result = MinimaxSolver(CROSS_CHECK_SOLVER).solve(
            MinimaxProblem(ambiguity_set, wrapped, policies.erm[index].clone())
        )
        worst_gap = max(worst_gap, abs(result.point.item() - reference.item()))
    return worst_gap


def log_slope(radii: torch.Tensor, values: torch.Tensor) -> float:
    """Least-squares slope of `log(values)` against `log(radii)`."""
    x, y = torch.log(radii), torch.log(values)
    return ((x - x.mean()) * (y - y.mean())).sum().item() / (
        (x - x.mean()) ** 2
    ).sum().item()


def main() -> None:
    nominal, demand = sample_demand()
    problems = (
        Problem("quadratic", quadratic, 1.0, RATE_RADII),
        Problem("LINEX", linex, 1.0, RATE_RADII),
        Problem(
            "newsvendor",
            newsvendor,
            2.0,
            NEWSVENDOR_RADII,
            benchmarks=demand,
            refine=False,
        ),
    )
    dense = torch.cat(
        [
            torch.linspace(
                demand.min().item(), demand.max().item(), DENSE_GRID, dtype=demand.dtype
            ),
            demand,
        ]
    )
    results: dict[str, tuple[Policies, torch.Tensor, torch.Tensor]] = {}

    for problem in problems:
        policies, ambiguity_set, benchmarks = solve_policies(problem, nominal, demand)
        regret, _, _ = robust_regret(
            ambiguity_set,
            problem.loss,
            demand,
            policies.ex_ante,
            benchmarks,
            BEST_RESPONSE_ROUNDS if problem.refine else 0,
        )
        dense_regret = ambiguity_set.worst_case_expectation(
            problem.loss(policies.ex_ante[:, None, None], demand)
            - problem.loss(dense[None, :, None], demand)
        ).amax(-1)
        inner_gap = (dense_regret - regret).max().item()
        nominal_value = (nominal * problem.loss(policies.erm[:, None], demand)).sum(-1)
        price = (
            ambiguity_set.worst_case_expectation(
                problem.loss(policies.wdro[:, None, None], demand)
            ).squeeze(-1)
            - nominal_value
        )
        results[problem.name] = (policies, regret, price)

        print(f"\n{problem.name}: empirical-risk decision {policies.erm[0].item():.6f}")
        print(
            "  radius      WDRO - ERM    ex-post - ERM   ex-ante - ERM"
            "   price of robustness   robust regret"
        )
        for index, radius in enumerate(problem.radii.tolist()):
            print(
                f"  {radius:8.2e}  {policies.wdro[index] - policies.erm[index]:+12.3e}"
                f"  {policies.ex_post[index] - policies.erm[index]:+13.3e}"
                f"  {policies.ex_ante[index] - policies.erm[index]:+13.3e}"
                f"  {price[index]:19.3e}   {regret[index]:13.3e}"
            )
        print(f"  benchmark maximization vs dense grid: {inner_gap:.1e}")
        assert inner_gap < EXACT_TOLERANCE

    quadratic_policies, quadratic_regret, _ = results["quadratic"]
    deviation = (quadratic_policies.ex_ante - quadratic_policies.erm).abs().max().item()
    drift = (
        (quadratic_policies.wdro - quadratic_policies.erm - RATE_RADII)
        .abs()
        .max()
        .item()
    )
    curvature = (quadratic_regret - RATE_RADII**2).abs().max().item()
    print(
        f"\nquadratic: max |ex-ante DRRO - ERM| = {deviation:.1e}, "
        f"max |WDRO - ERM - radius| = {drift:.1e}, "
        f"max |regret - radius^2| = {curvature:.1e}"
    )
    assert max(deviation, drift, curvature) < EXACT_TOLERANCE
    for name in ("quadratic", "LINEX"):
        policies = results[name][0]
        assert torch.allclose(policies.ex_post, policies.wdro, atol=1e-10), name

    linex_policies, linex_regret, linex_price = results["LINEX"]
    fitted = RATE_RADII[SLOPE_RADII]
    slopes = {
        "WDRO decision": log_slope(
            fitted, (linex_policies.wdro - linex_policies.erm).abs()[SLOPE_RADII]
        ),
        "ex-ante decision": log_slope(
            fitted, (linex_policies.ex_ante - linex_policies.erm).abs()[SLOPE_RADII]
        ),
        "price of robustness": log_slope(fitted, linex_price[SLOPE_RADII]),
        "robust regret": log_slope(fitted, linex_regret[SLOPE_RADII]),
    }
    print(
        "LINEX log-log slopes against the radius, radius <= 1e-2: "
        + ", ".join(f"{k} {v:.2f}" for k, v in slopes.items())
    )
    assert abs(slopes["WDRO decision"] - 1.0) < 0.1
    assert abs(slopes["price of robustness"] - 1.0) < 0.1
    assert slopes["ex-ante decision"] > 1.5
    assert abs(slopes["robust regret"] - 2.0) < 0.15
    assert (
        (linex_policies.ex_ante - linex_policies.erm).abs()
        < (linex_policies.wdro - linex_policies.erm).abs()
    ).all()

    news_policies, _, _ = results["newsvendor"]
    moves = {
        name: (getattr(news_policies, name) - news_policies.erm)
        for name in ("wdro", "ex_post", "ex_ante")
    }
    large = NEWSVENDOR_RADII >= 1e-2
    direction = (
        (moves["wdro"][large] < 0.0)
        & (moves["ex_post"][large] > 0.0)
        & (moves["ex_ante"][large] > 0.0)
    )
    larger = NEWSVENDOR_RADII >= 3e-2
    least = (moves["ex_ante"].abs() < moves["ex_post"].abs())[larger] & (
        moves["ex_post"].abs() < moves["wdro"].abs()
    )[larger]
    print(
        "newsvendor: DRO orders less and both regret policies order more at every"
        f" radius >= 1e-2: {bool(direction.all())}; ex-ante moves least and DRO"
        f" most at every radius >= 3e-2: {bool(least.all())}"
    )
    assert bool(direction.all()) and bool(least.all())
    gap = cross_check_minimax(
        problems[2],
        nominal,
        demand,
        hindsight_losses(newsvendor, demand),
        news_policies,
    )
    print(
        "newsvendor: MinimaxSolver vs bisection at radius "
        f"{NEWSVENDOR_RADII[CROSS_CHECK_RADIUS].item():.1e}, "
        f"max |difference| = {gap:.1e}"
    )
    assert gap < 2e-2

    fig, (orders, decisions, values) = plt.subplots(1, 3, figsize=(14.5, 4.4))
    radii = NEWSVENDOR_RADII.numpy()
    orders.axhline(news_policies.erm[0].item(), color="0.5", linewidth=1.0, label="ERM")
    for label, series, color, style in (
        ("Wasserstein DRO", news_policies.wdro, "C0", "-"),
        ("ex-post DRRO (cost form)", news_policies.ex_post, "C2", "--"),
        ("ex-ante DRRO", news_policies.ex_ante, "C3", "-"),
    ):
        orders.step(
            radii,
            series.numpy(),
            where="post",
            color=color,
            linestyle=style,
            label=label,
        )
        orders.plot(radii, series.numpy(), "o", color=color, markersize=4)
    orders.set_xscale("log")
    orders.set_xlabel("Wasserstein radius $\\delta$ (squared ground cost)")
    orders.set_ylabel("order quantity")
    orders.set_title("newsvendor: DRO orders less, regret orders more")
    orders.legend(loc="upper left", fontsize=7.5)

    rate = RATE_RADII.numpy()
    decisions.plot(
        rate, rate, color="0.6", linewidth=1.0, linestyle=":", label="slope 1"
    )
    decisions.plot(
        rate, rate**2, color="0.6", linewidth=1.0, linestyle="--", label="slope 2"
    )
    for name, color in (("quadratic", "C1"), ("LINEX", "C3")):
        policies = results[name][0]
        decisions.plot(
            rate,
            (policies.wdro - policies.erm).abs().numpy(),
            "o-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            label=f"{name}: WDRO",
        )
        if name == "LINEX":
            decisions.plot(
                rate,
                (policies.ex_ante - policies.erm).abs().numpy(),
                "s-",
                color=color,
                label=f"{name}: ex-ante DRRO",
            )
    decisions.text(
        0.97,
        0.03,
        "quadratic ex-ante DRRO: exactly ERM\nat every radius (Theorem 4)",
        transform=decisions.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    decisions.set_xscale("log")
    decisions.set_yscale("log")
    decisions.set_xlabel("Wasserstein radius $\\delta$ (absolute ground cost)")
    decisions.set_ylabel("$|\\theta_\\delta - \\theta_{\\mathrm{ERM}}|$")
    decisions.set_title("the regret decision stays with ERM to first order")
    decisions.legend(loc="upper left", fontsize=7.5)

    values.plot(rate, rate, color="0.6", linewidth=1.0, linestyle=":", label="slope 1")
    values.plot(
        rate, rate**2, color="0.6", linewidth=1.0, linestyle="--", label="slope 2"
    )
    for name, color in (("quadratic", "C1"), ("LINEX", "C3")):
        _, regret, price = results[name]
        values.plot(
            rate,
            price.numpy(),
            "o-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            label=f"{name}: price of robustness",
        )
        values.plot(
            rate, regret.numpy(), "s-", color=color, label=f"{name}: robust regret"
        )
    values.set_xscale("log")
    values.set_yscale("log")
    values.set_xlabel("Wasserstein radius $\\delta$ (absolute ground cost)")
    values.set_ylabel("value")
    values.set_title("worst-case loss grows like $\\delta$, regret like $\\delta^2$")
    values.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "11_distributionally_robust_regret")


if __name__ == "__main__":
    main()
