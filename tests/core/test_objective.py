"""Tests for objective evaluation helpers."""

import pytest
import torch

from optora.core.objective import Objective


def test_objective_uses_user_gradient() -> None:
    objective = Objective(
        fun=lambda x: torch.dot(x, x),
        grad=lambda x: 2.0 * x,
    )

    gradient = objective.gradient(torch.tensor([1.0, -3.0], dtype=torch.float64))

    torch.testing.assert_close(
        gradient,
        torch.tensor([2.0, -6.0], dtype=torch.float64),
    )
    assert objective.njev == 1
    assert objective.nfev == 0


def test_objective_finite_difference_gradient() -> None:
    objective = Objective(fun=lambda x: (x[0] - 3.0) ** 2)

    gradient = objective.gradient(torch.tensor([1.0], dtype=torch.float64))

    torch.testing.assert_close(
        gradient,
        torch.tensor([-4.0], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    assert objective.njev == 1
    assert objective.nfev == 2


def test_objective_rejects_gradient_shape_mismatch() -> None:
    objective = Objective(
        fun=lambda x: torch.dot(x, x),
        grad=lambda x: torch.tensor([x[0]], dtype=x.dtype, device=x.device),
    )

    with pytest.raises(ValueError, match="gradient shape"):
        objective.gradient(torch.tensor([1.0, 2.0], dtype=torch.float64))
