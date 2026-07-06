"""Tests for optimization problem tensor inputs."""

import torch

from optora.core.objective import Objective
from optora.core.problem import UnconstrainedOptimizationProblem


def test_initial_tensor_preserves_tensor_dtype_and_is_independent() -> None:
    initial_point = torch.tensor([1.0, 2.0], dtype=torch.float32)
    problem = UnconstrainedOptimizationProblem(
        objective=Objective(fun=lambda x: torch.dot(x, x)),
        initial_point=initial_point,
    )

    result = problem.initial_tensor()
    result[0] = 10.0

    assert result.dtype == torch.float32
    assert initial_point[0] == 1.0


def test_initial_tensor_converts_sequences_to_floating_tensors() -> None:
    problem = UnconstrainedOptimizationProblem(
        objective=Objective(fun=lambda x: torch.dot(x, x)),
        initial_point=[1.0, 2.0],
    )

    result = problem.initial_tensor()

    assert result.is_floating_point()
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
