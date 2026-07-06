"""Tests for shared line-search routines."""

import torch

from optora.solvers.line_search import backtracking_armijo


def test_backtracking_armijo_decreases_quadratic() -> None:
    x = torch.tensor([2.0], dtype=torch.float64)
    grad = torch.tensor([4.0], dtype=torch.float64)
    direction = -grad

    step, value = backtracking_armijo(
        lambda point: point[0] ** 2,
        x,
        direction,
        grad,
        initial_step=1.0,
    )

    assert step < 1.0
    assert value < 4.0
