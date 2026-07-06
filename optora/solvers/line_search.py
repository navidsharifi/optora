"""Shared line-search routines."""

from __future__ import annotations

from collections.abc import Callable

import torch

Tensor = torch.Tensor
ValueFunction = Callable[[Tensor], Tensor]


def backtracking_armijo(
    fun: ValueFunction,
    x: Tensor,
    direction: Tensor,
    grad: Tensor,
    *,
    initial_step: float = 1.0,
    contraction: float = 0.5,
    sufficient_decrease: float = 1e-4,
    min_step: float = 1e-12,
    max_iter: int = 50,
) -> tuple[float, Tensor]:
    """Find a step satisfying the Armijo sufficient-decrease condition.

    Args:
        fun: Objective value function.
        x: Current point.
        direction: Search direction.
        grad: Gradient at ``x``.
        initial_step: Initial trial step length.
        contraction: Multiplicative step contraction factor.
        sufficient_decrease: Armijo condition constant.
        min_step: Smallest trial step allowed before returning the best trial.
        max_iter: Maximum number of backtracking trials.

    Returns:
        A pair ``(step, value)`` for the accepted trial point.

    Raises:
        ValueError: If line-search parameters are outside their valid ranges.
    """
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    if not 0.0 < contraction < 1.0:
        raise ValueError("contraction must be between 0 and 1")
    if sufficient_decrease <= 0.0:
        raise ValueError("sufficient_decrease must be positive")
    if min_step <= 0.0:
        raise ValueError("min_step must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    f0 = fun(x)
    directional_derivative = torch.dot(grad.reshape(-1), direction.reshape(-1))
    step = initial_step
    trial_value = f0

    for _ in range(max_iter):
        trial = x + step * direction
        trial_value = fun(trial)
        sufficient_value = f0 + sufficient_decrease * step * directional_derivative
        if bool((trial_value <= sufficient_value).detach().item()):
            return step, trial_value
        step *= contraction
        if step < min_step:
            break

    return step, trial_value
