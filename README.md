# Optora

Optora is an early-stage Python library for composable research components in
optimization, uncertainty-aware decision-making, and transport-based
geometry. The first milestone is intentionally small: explicit problem,
objective, result, and solver interfaces that can later grow into stochastic
methods, differentiable programming backends, optimal transport, and Bayesian
decision geometry.

> Status: pre-alpha. APIs are designed carefully, but still expected to evolve.

## First Example

```python
import numpy as np
from optora.core import Objective, UnconstrainedOptimizationProblem
from optora.solvers.deterministic import BFGS


def fun(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + 10.0 * (x[1] + 2.0) ** 2)


def grad(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 1.0), 20.0 * (x[1] + 2.0)])


objective = Objective(fun=fun, grad=grad)
problem = UnconstrainedOptimizationProblem(
    objective=objective,
    initial_point=np.array([0.0, 0.0]),
)
solver = BFGS()

result = solver.solve(problem)

print(result.x)
print(result.success)
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy optora
```

## Roadmap

1. Deterministic differentiable optimization:
   gradient descent, line search, Newton, BFGS, L-BFGS, proximal methods.
2. Stochastic optimization:
   SGD, momentum, Adam-style methods, stochastic approximation, variance
   reduction.
3. Differentiable backends:
   optional JAX/PyTorch/CuPy support through a backend layer.
4. Research modules:
   optimal transport, entropic OT, differentiable decision-making, and RL.
