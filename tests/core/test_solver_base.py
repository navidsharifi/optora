"""Tests for the `Solver` ABC contract."""

import pytest
import torch

from optora.core.solver_base import Solver


class _EchoResult:
    """Minimal result type used to exercise the `Solver` ABC."""

    def __init__(self, point: torch.Tensor) -> None:
        self.point = point


class _EchoSolver(Solver[torch.Tensor, _EchoResult]):
    """Solver that returns its input unchanged, used to exercise the ABC."""

    def solve(self, problem: torch.Tensor) -> _EchoResult:
        return _EchoResult(problem)


def test_solver_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Solver()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class _IncompleteSolver(Solver[torch.Tensor, _EchoResult]):
        pass

    with pytest.raises(TypeError):
        _IncompleteSolver()  # type: ignore[abstract]


def test_concrete_solver_solve_is_delegated_to_subclass() -> None:
    solver = _EchoSolver()
    problem = torch.tensor([1.0, 2.0])

    result = solver.solve(problem)

    assert torch.equal(result.point, problem)
