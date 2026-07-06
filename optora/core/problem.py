"""Problem abstractions for composable optimization methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

from optora.core.objective import Objective

Tensor = torch.Tensor
TensorLike: TypeAlias = (
    torch.Tensor | float | Sequence[float] | Sequence[Sequence[float]]
)


@dataclass(slots=True)
class UnconstrainedOptimizationProblem:
    """Finite-dimensional unconstrained optimization problem.

    The problem object contains the mathematical ingredients required by a
    solver, but does not prescribe a training loop, method registry, or
    workflow. Researchers instantiate the problem and pass it to the method
    implementation they want to study.

    Args:
        objective: Scalar objective with derivative information.
        initial_point: Initial iterate used by iterative methods.
    """

    objective: Objective
    initial_point: TensorLike

    def initial_tensor(self) -> Tensor:
        """Return the initial point as an independent floating-point tensor."""
        if isinstance(self.initial_point, torch.Tensor):
            tensor = self.initial_point
            if not tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.get_default_dtype())
            tensor = tensor.clone()
        else:
            tensor = torch.as_tensor(
                self.initial_point,
                dtype=torch.get_default_dtype(),
            )
        if tensor.ndim == 0:
            return tensor.reshape(1)
        return tensor
