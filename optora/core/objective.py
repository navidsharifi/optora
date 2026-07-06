"""Objective function wrapper and finite-difference gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import torch

Tensor = torch.Tensor
TensorLike: TypeAlias = (
    torch.Tensor | float | Sequence[float] | Sequence[Sequence[float]]
)
ScalarLike: TypeAlias = torch.Tensor | float | int
ObjectiveFunction = Callable[[Tensor], ScalarLike]
GradientFunction = Callable[[Tensor], TensorLike]


@dataclass(slots=True)
class Objective:
    """Wrap an objective and optional gradient with evaluation counters.

    Args:
        fun: Objective function mapping a point to a scalar value.
        grad: Optional gradient function. If omitted, central finite
            differences are used.
        finite_difference_step: Perturbation size for finite differences.
    """

    fun: ObjectiveFunction
    grad: GradientFunction | None = None
    finite_difference_step: float = 1e-8
    nfev: int = field(default=0, init=False)
    njev: int = field(default=0, init=False)

    def value(self, x: Tensor) -> Tensor:
        """Evaluate the objective at ``x`` and increment ``nfev``."""
        self.nfev += 1
        value = self.fun(x)
        if isinstance(value, torch.Tensor):
            result = value.to(device=x.device, dtype=x.dtype)
        else:
            result = torch.as_tensor(value, dtype=x.dtype, device=x.device)

        if result.numel() != 1:
            shape = tuple(result.shape)
            msg = f"objective function must return a scalar, got shape {shape}"
            raise ValueError(msg)
        return result.reshape(())

    def gradient(self, x: Tensor) -> Tensor:
        """Evaluate or approximate the gradient at ``x``."""
        self.njev += 1
        if self.grad is not None:
            value = self.grad(x)
            if isinstance(value, torch.Tensor):
                gradient = value.to(device=x.device, dtype=x.dtype)
            else:
                gradient = torch.as_tensor(value, dtype=x.dtype, device=x.device)
            if gradient.shape != x.shape:
                msg = (
                    "gradient shape must match x: "
                    f"expected {x.shape}, got {gradient.shape}"
                )
                raise ValueError(msg)
            return gradient
        return self._finite_difference_gradient(x)

    def _finite_difference_gradient(self, x: Tensor) -> Tensor:
        """Approximate the gradient with central finite differences."""
        if self.finite_difference_step <= 0.0:
            msg = "finite_difference_step must be positive"
            raise ValueError(msg)

        gradient = torch.empty_like(x)
        flat_gradient = gradient.reshape(-1)
        flat_x = x.reshape(-1)

        for index in range(flat_x.numel()):
            x_forward = flat_x.clone()
            x_backward = flat_x.clone()
            x_forward[index] += self.finite_difference_step
            x_backward[index] -= self.finite_difference_step
            forward = self.value(x_forward.reshape(x.shape))
            backward = self.value(x_backward.reshape(x.shape))
            flat_gradient[index] = (forward - backward) / (
                2.0 * self.finite_difference_step
            )

        return gradient
