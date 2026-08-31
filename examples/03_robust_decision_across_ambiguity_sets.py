"""Comparing the robust decision across all four ambiguity-set geometries.

Problem: choose a single scalar decision `x` to minimize the worst-case
expected squared error against four observed outcomes,

    min_x sup_{q in ambiguity_set(nominal, radius)} E_q[(outcomes - x) ** 2]

At `radius = 0` every ambiguity set collapses to `{nominal}`, so every
formulation recovers the same ordinary empirical-risk decision (the
nominal-weighted mean of `outcomes`). As `radius` grows, each formulation's
robust decision `x*` is pulled toward the outcome that would hurt most if
an adversary reallocated probability mass there; how quickly, and how far,
depends on the divergence geometry underneath the ambiguity set.

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

# Nested outer/inner solver composition is expensive in eager PyTorch (see
# `progress/decisions.md`), so both budgets below are tuned down
# aggressively from each solver's own defaults; this is a demo of
# qualitative behavior, not a tight-tolerance convergence test.
OUTER_SOLVER = GradientDescent(step_size=0.01, max_iter=120, tol=1e-7)
KL_DUAL_SOLVER = GradientDescent(step_size=0.1, max_iter=200, tol=1e-8)
CHI_SQUARE_DUAL_SOLVER = GradientDescent(step_size=0.05, max_iter=300, tol=1e-8)
WASSERSTEIN_DUAL_SOLVER = GradientDescent(step_size=0.02, max_iter=300, tol=1e-8)


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
        "KL": lambda radius: KLAmbiguitySet(
            nominal=NOMINAL, radius=radius, dual_solver=KL_DUAL_SOLVER
        ),
        "chi-square": lambda radius: ChiSquareAmbiguitySet(
            nominal=NOMINAL, radius=radius, dual_solver=CHI_SQUARE_DUAL_SOLVER
        ),
        "total variation": lambda radius: TotalVariationAmbiguitySet(
            nominal=NOMINAL, radius=radius
        ),
        "Wasserstein": lambda radius: WassersteinAmbiguitySet(
            nominal=NOMINAL,
            cost=COST,
            radius=radius,
            dual_solver=WASSERSTEIN_DUAL_SOLVER,
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
