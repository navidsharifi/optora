"""Comparing the robust decision across all four ambiguity-set geometries.

The problem is to choose a single scalar decision `x` that minimizes the
worst-case expected squared error against four observed outcomes,

    min_x sup_{q in ambiguity_set(nominal, radius)} E_q[(outcomes - x) ** 2]

At `radius = 0` every ambiguity set collapses to `{nominal}`, so every
formulation recovers the same empirical-risk decision, the
nominal-weighted mean of `outcomes`. As `radius` grows, the robust
decision `x*` of each formulation moves toward the outcomes onto which
the adversary shifts probability mass, and the divergence that defines
the ambiguity set decides which outcomes those are.

The KL, chi-square and total-variation balls depend only on the
probabilities of the scenarios, so their adversaries put more mass on
the outlier 10 and their decisions rise from 4.0 toward it. The
Wasserstein ball also depends on the distances between outcomes. Under
the squared cost, moving mass onto 10 costs at least 49 per unit, while
moving mass from 2 and 3 down to 1 raises the loss near `x = 4` the
most per unit of transport cost. Its decision therefore falls toward
the low cluster and away from the outlier.

Run:
    python examples/03_robust_decision_across_ambiguity_sets.py
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import (
    ChiSquareAmbiguitySet,
    KLAmbiguitySet,
    MinimaxProblem,
    MinimaxSolver,
    TotalVariationAmbiguitySet,
    WassersteinAmbiguitySet,
)
from optora.solvers import GradientDescent

OUTCOMES = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
NOMINAL = torch.full_like(OUTCOMES, 1.0 / OUTCOMES.numel())
COST = (OUTCOMES.unsqueeze(0) - OUTCOMES.unsqueeze(1)) ** 2

# Every outer step runs a full inner worst-case solve, so nested solves are
# expensive in eager PyTorch. The outer iteration budget below is therefore
# cut well below the solver's default. The example shows qualitative
# behavior and is not a tight-tolerance convergence test.
# The outer solver uses `check_interval=1` because one of its steps costs a
# full inner worst-case solve, which is far more than the synchronization a
# convergence check costs.
OUTER_SOLVER = GradientDescent(step_size=0.01, max_iter=120, tol=1e-7, check_interval=1)


def loss_fn(x: torch.Tensor) -> torch.Tensor:
    return (OUTCOMES - x) ** 2


def robust_decision(ambiguity_set: object, radius: float) -> float:
    problem = MinimaxProblem(
        ambiguity_set=ambiguity_set,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        initial_point=torch.tensor(OUTCOMES.mean().item(), dtype=torch.float64),
    )
    result = MinimaxSolver(solver=OUTER_SOLVER).solve(problem)
    return result.point.item()


def main() -> None:
    erm_decision = OUTCOMES.mean().item()
    print(
        f"empirical-risk decision (every formulation at radius=0): {erm_decision:.4f}"
    )

    families: dict[str, Callable[[float], object]] = {
        "KL": lambda radius: KLAmbiguitySet(nominal=NOMINAL, radius=radius),
        "chi-square": lambda radius: ChiSquareAmbiguitySet(
            nominal=NOMINAL, radius=radius
        ),
        "total variation": lambda radius: TotalVariationAmbiguitySet(
            nominal=NOMINAL, radius=radius
        ),
        "Wasserstein": lambda radius: WassersteinAmbiguitySet(
            nominal=NOMINAL,
            cost=COST,
            radius=radius,
        ),
    }
    radii = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, build_ambiguity_set in families.items():
        decisions = [
            robust_decision(build_ambiguity_set(radius), radius) for radius in radii
        ]
        print(f"{name:>16}: " + " -> ".join(f"{value:.3f}" for value in decisions))
        ax.plot(radii, decisions, marker="o", label=name)

    ax.axhline(
        erm_decision, color="gray", linestyle="--", label="empirical risk (radius=0)"
    )
    ax.set_xlabel("ambiguity radius")
    ax.set_ylabel("robust decision x*")
    ax.set_title("robust decision vs. radius, across ambiguity-set geometries")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "03_robust_decision_across_ambiguity_sets")


if __name__ == "__main__":
    main()
