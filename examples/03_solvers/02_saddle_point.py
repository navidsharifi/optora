r"""Saddle point of a Lagrangian.

Project a point onto the half-space $\{x : \sum_i x_i \le 1\}$ by solving
the Lagrangian saddle point with `SaddlePointSolver`, using
`dual_projection` to keep the multiplier nonnegative, and compare with
the closed-form projection.
"""

import torch

from optora import SaddlePointProblem, SaddlePointSolver

torch.set_default_dtype(torch.float64)

c = torch.tensor([0.8, 0.5, 0.3])


def lagrangian(x: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
    """Return 0.5 ||x - c||^2 + multiplier * (sum(x) - 1)."""
    return 0.5 * ((x - c) ** 2).sum() + multiplier * (x.sum() - 1.0)


problem = SaddlePointProblem(
    objective=lagrangian,
    primal_initial_point=torch.zeros(3),
    dual_initial_point=torch.tensor(0.0),
    dual_projection=lambda multiplier: multiplier.clamp_min(0.0),
)
solver = SaddlePointSolver(
    primal_step_size=0.2, dual_step_size=0.2, max_iter=3000, tol=1e-9
)
result = solver.solve(problem)

# The projection onto the half-space shifts every coordinate by the same
# multiplier, max(0, (sum(c) - 1) / n).
multiplier = ((c.sum() - 1.0) / len(c)).clamp_min(0.0)
print(f"projection       : {result.primal_point}")
print(f"closed form      : {c - multiplier}")
print(f"multiplier       : {result.dual_point:.6f} (closed form {multiplier:.6f})")
print(f"converged        : {bool(result.status.converged)}")

torch.testing.assert_close(result.primal_point, c - multiplier)
torch.testing.assert_close(result.dual_point, multiplier)
