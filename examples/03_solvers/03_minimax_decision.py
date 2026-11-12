"""Robust decision with MinimaxSolver.

Choose the decision $x$ that minimizes the worst-case squared error over a
KL ball of distributions on four outcomes. `MinimaxSolver` runs an outer
`GradientDescent` on $x$ and lets the ambiguity set solve the inner
maximization exactly at every step.
"""

import torch

from optora import GradientDescent, KLAmbiguitySet, MinimaxProblem, MinimaxSolver

torch.set_default_dtype(torch.float64)

nominal = torch.full((4,), 0.25)
outcomes = torch.tensor([1.0, 2.0, 3.0, 10.0])
ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

problem = MinimaxProblem(
    ambiguity_set=ambiguity_set,
    loss_fn=lambda x: (outcomes - x) ** 2,
    initial_point=torch.tensor(0.0),
)
solver = MinimaxSolver(GradientDescent(step_size=0.05, max_iter=500, tol=1e-8))
result = solver.solve(problem)

# The empirical-risk decision is the nominal mean. The robust decision
# moves toward the outlier, because the adversary upweights it.
erm_decision = (nominal * outcomes).sum()
print(f"ERM decision          : {erm_decision:.4f}")
print(f"robust decision       : {result.point:.4f}")
print(f"worst-case loss there : {result.value:.4f}")
print(f"outer iterations      : {int(result.status.num_iterations)}")
assert result.point > erm_decision

# At the robust optimum, the gradient of the worst case in x vanishes:
# sum_i q*_i (x - outcome_i) = 0, so x is the mean under q*.
loss = ((outcomes - result.point) ** 2).requires_grad_()
ambiguity_set.worst_case_expectation(loss).backward()
worst_case = loss.grad
torch.testing.assert_close(
    result.point, (worst_case * outcomes).sum(), rtol=1e-6, atol=0.0
)
