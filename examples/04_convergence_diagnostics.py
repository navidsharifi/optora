"""Convergence diagnostics for a KL-DRO robust decision problem.

Two complementary views of "convergence" in a DRO minimax problem:

1. Outer convergence: manually unrolling the same fixed-step gradient
   iteration `optora.solvers.GradientDescent` uses internally (instead of
   calling `MinimaxSolver.solve` as a black box) to record the robust
   decision `x` and the worst-case objective value at every iteration.
2. Inner convergence: confirming, via a fine grid search over the KL-DRO
   dual variable `eta`, that the ambiguity set's own inner dual solve
   actually lands at the true minimizer of the dual objective -- i.e. that
   the worst-case adversarial distribution has genuinely been found, not
   merely approximated by an unconverged solver.

Run:
    python examples/04_convergence_diagnostics.py
"""

import matplotlib.pyplot as plt
import torch

from optora.dro import KLAmbiguitySet

from _plotting import save_figure

OUTCOMES = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
NOMINAL = torch.full_like(OUTCOMES, 1.0 / OUTCOMES.numel())
RADIUS = 0.15


def track_outer_convergence(
    ambiguity_set: KLAmbiguitySet,
    initial_point: torch.Tensor,
    step_size: float,
    num_iterations: int,
) -> tuple[list[float], list[float]]:
    """Manually unroll a fixed-step gradient trajectory, recording history."""

    def objective(x: torch.Tensor) -> torch.Tensor:
        return ambiguity_set.worst_case_expectation((OUTCOMES - x) ** 2)

    point = initial_point.clone().requires_grad_(True)
    x_history = [point.item()]
    value_history = [objective(point).item()]
    for _ in range(num_iterations):
        value = objective(point)
        (grad,) = torch.autograd.grad(value, point)
        with torch.no_grad():
            point = point - step_size * grad
        point = point.detach().requires_grad_(True)
        x_history.append(point.item())
        value_history.append(objective(point).item())
    return x_history, value_history


def kl_dual_grid_search(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float, log_eta_grid: torch.Tensor
) -> torch.Tensor:
    """Evaluate the KL-DRO dual objective on a fine grid of `log(eta)`."""
    log_nominal = torch.log(nominal)
    eta = torch.exp(log_eta_grid)
    log_mgf = torch.logsumexp(
        log_nominal.unsqueeze(0) + loss.unsqueeze(0) / eta.unsqueeze(1), dim=-1
    )
    return eta * radius + eta * log_mgf


def main() -> None:
    ambiguity_set = KLAmbiguitySet(nominal=NOMINAL, radius=RADIUS)

    x_history, value_history = track_outer_convergence(
        ambiguity_set,
        initial_point=torch.tensor(OUTCOMES.mean().item(), dtype=torch.float64),
        step_size=0.01,
        num_iterations=150,
    )
    print(f"x trajectory:         {x_history[0]:.4f} -> {x_history[-1]:.4f}")
    print(f"objective trajectory: {value_history[0]:.4f} -> {value_history[-1]:.4f}")

    fig1, (ax_x, ax_value) = plt.subplots(1, 2, figsize=(10, 4))
    ax_x.plot(x_history)
    ax_x.set_xlabel("outer iteration")
    ax_x.set_ylabel("decision variable x")
    ax_x.set_title("robust decision trajectory")
    ax_value.plot(value_history)
    ax_value.set_xlabel("outer iteration")
    ax_value.set_ylabel("worst-case objective")
    ax_value.set_title("worst-case objective trajectory")
    fig1.tight_layout()
    save_figure(fig1, "04_outer_convergence")

    loss_at_solution = (OUTCOMES - x_history[-1]) ** 2
    log_eta_grid = torch.linspace(-6.0, 4.0, 400, dtype=torch.float64)
    dual_values = kl_dual_grid_search(NOMINAL, loss_at_solution, RADIUS, log_eta_grid)
    grid_min_value, grid_argmin_index = torch.min(dual_values, dim=0)
    library_value = ambiguity_set.worst_case_expectation(loss_at_solution)

    print(
        f"grid-search dual minimum: {grid_min_value.item():.6f} "
        f"at log(eta)={log_eta_grid[grid_argmin_index].item():.4f}"
    )
    print(f"library worst_case_expectation:  {library_value.item():.6f}")
    print(f"absolute difference:             {abs(grid_min_value.item() - library_value.item()):.2e}")

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log_eta_grid, dual_values.detach(), label="dual objective")
    ax.axvline(
        log_eta_grid[grid_argmin_index].item(),
        color="firebrick",
        linestyle="--",
        label="grid-search minimizer",
    )
    ax.set_xlabel("log(eta)")
    ax.set_ylabel("KL-DRO dual objective")
    ax.set_title("inner dual landscape at the robust solution")
    ax.legend()
    fig2.tight_layout()
    save_figure(fig2, "04_inner_dual_landscape")


if __name__ == "__main__":
    main()
