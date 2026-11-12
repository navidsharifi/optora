"""Gradient descent.

Wrap an objective and a starting point in a `MinimizationProblem`, solve
it with `GradientDescent`, and read the minimizer, the optimal value and
the convergence diagnostics from the result.
"""

import torch

from optora import GradientDescent, MinimizationProblem

torch.set_default_dtype(torch.float64)

# A strictly convex quadratic 0.5 x^T A x - b^T x with minimizer A^{-1} b.
A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
b = torch.tensor([1.0, -1.0])

problem = MinimizationProblem(
    objective=lambda x: 0.5 * x @ A @ x - b @ x,
    initial_point=torch.zeros(2),
)
# The gradient is evaluated with autograd, so only the objective is needed.
solver = GradientDescent(step_size=0.2, max_iter=500, tol=1e-10)
result = solver.solve(problem)

print(f"minimizer      : {result.point}")
print(f"optimal value  : {result.value:.6f}")
print(f"converged      : {bool(result.status.converged)}")
print(f"iterations     : {int(result.status.num_iterations)}")

torch.testing.assert_close(result.point, torch.linalg.solve(A, b))
assert result.status.converged
