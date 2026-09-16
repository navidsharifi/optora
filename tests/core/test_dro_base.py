"""Tests for the `AmbiguitySet` ABC contract."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from optora.core.convergence import ConvergenceStatus
from optora.core.divergence_base import Divergence
from optora.core.dro_base import AmbiguitySet, DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)


class _AbsoluteDifferenceDivergence(Divergence):
    """Sum-of-absolute-differences divergence used to exercise the ABC."""

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.abs(p - q).sum()


class _MaxLossAmbiguitySet(AmbiguitySet):
    """Ambiguity set whose worst case is simply the maximum loss."""

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.max()


@pytest.fixture
def nominal() -> torch.Tensor:
    return torch.tensor([0.5, 0.5])


def test_ambiguity_set_cannot_be_instantiated_directly(nominal: torch.Tensor) -> None:
    with pytest.raises(TypeError):
        AmbiguitySet(  # type: ignore[abstract]
            nominal, _AbsoluteDifferenceDivergence(), radius=0.1
        )


def test_negative_radius_raises_value_error(nominal: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _MaxLossAmbiguitySet(nominal, _AbsoluteDifferenceDivergence(), radius=-1.0)


def test_contains_uses_divergence_and_radius(nominal: torch.Tensor) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )

    assert ambiguity_set.contains(torch.tensor([0.6, 0.4]))
    assert not ambiguity_set.contains(torch.tensor([0.9, 0.1]))


def test_contains_returns_a_tensor_without_synchronizing(
    nominal: torch.Tensor,
    host_sync_counter: Callable[[], Any],
) -> None:
    # `contains` must not bake a device-to-host copy into the ABC: callers
    # that only need a mask should not be forced to stall the accelerator.
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )

    with host_sync_counter() as syncs:
        membership = ambiguity_set.contains(torch.tensor([0.6, 0.4]))

    assert syncs == []
    assert isinstance(membership, torch.Tensor)
    assert membership.dtype == torch.bool


def test_worst_case_expectation_is_delegated_to_subclass(
    nominal: torch.Tensor,
) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )
    loss = torch.tensor([1.0, 3.0, 2.0])

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.equal(result, torch.tensor(3.0))


def test_ambiguity_set_is_an_nn_module_with_registered_buffer_and_submodule(
    nominal: torch.Tensor,
) -> None:
    divergence = _AbsoluteDifferenceDivergence()
    ambiguity_set = _MaxLossAmbiguitySet(nominal, divergence, radius=0.5)

    assert isinstance(ambiguity_set, nn.Module)
    assert "nominal" in dict(ambiguity_set.named_buffers())
    assert dict(ambiguity_set.named_modules())["divergence"] is divergence


def test_to_moves_nominal_buffer_and_divergence_submodule_together(
    nominal: torch.Tensor,
) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )

    ambiguity_set = ambiguity_set.to(dtype=torch.float64)

    assert ambiguity_set.nominal.dtype == torch.float64


# --- DualAmbiguitySet -------------------------------------------------------


class _RecordingSolver(Solver[MinimizationProblem, MinimizationResult]):
    """Fake dual solver recording every starting point it is handed."""

    def __init__(self, point: float = 3.0) -> None:
        self.point = point
        self.initial_points: list[torch.Tensor] = []

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        self.initial_points.append(problem.initial_point.clone())
        point = torch.full_like(problem.initial_point, self.point)
        return MinimizationResult(
            point=point,
            value=problem.objective(point),
            status=ConvergenceStatus(torch.tensor(True), torch.tensor(0)),
        )


class _QuadraticDualAmbiguitySet(DualAmbiguitySet):
    """Dual-solved ambiguity set with a trivially convex scalar dual."""

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        batch_shape = self._batch_shape(loss)

        def dual_objective(dual_point: torch.Tensor) -> torch.Tensor:
            return (dual_point - loss.sum(dim=-1)) ** 2

        return self._solve_dual(dual_objective, batch_shape)


def _dual_ambiguity_set(
    nominal: torch.Tensor,
    dual_solver: Solver[MinimizationProblem, MinimizationResult] | None,
    initial_dual_point: float = 0.0,
) -> _QuadraticDualAmbiguitySet:
    return _QuadraticDualAmbiguitySet(
        nominal=nominal,
        divergence=_AbsoluteDifferenceDivergence(),
        radius=0.5,
        dual_solver=dual_solver,
        initial_dual_point=torch.tensor(initial_dual_point, dtype=nominal.dtype),
    )


def test_dual_solve_without_a_solver_names_the_concrete_subclass(
    nominal: torch.Tensor,
) -> None:
    ambiguity_set = _dual_ambiguity_set(nominal, dual_solver=None)

    with pytest.raises(RuntimeError, match="dual_solver is required"):
        ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))

    with pytest.raises(RuntimeError, match="_QuadraticDualAmbiguitySet"):
        ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))


def test_first_dual_solve_starts_from_the_initial_dual_point(
    nominal: torch.Tensor,
) -> None:
    solver = _RecordingSolver()
    ambiguity_set = _dual_ambiguity_set(nominal, solver, initial_dual_point=-1.5)

    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))

    assert torch.equal(solver.initial_points[0], torch.tensor(-1.5))


def test_later_dual_solves_warm_start_from_the_previous_optimum(
    nominal: torch.Tensor,
) -> None:
    # The dual optimum barely moves between consecutive calls on a slowly
    # changing loss, so restarting from it is what removes the cold-start
    # cost of an outer minimax iteration.
    solver = _RecordingSolver(point=3.0)
    ambiguity_set = _dual_ambiguity_set(nominal, solver, initial_dual_point=-1.5)

    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))
    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.5]))
    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 3.0]))

    assert torch.equal(solver.initial_points[1], torch.tensor(3.0))
    assert torch.equal(solver.initial_points[2], torch.tensor(3.0))


def test_reset_warm_start_returns_the_next_solve_to_the_initial_dual_point(
    nominal: torch.Tensor,
) -> None:
    solver = _RecordingSolver(point=3.0)
    ambiguity_set = _dual_ambiguity_set(nominal, solver, initial_dual_point=-1.5)

    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))
    ambiguity_set.reset_warm_start()
    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))

    assert torch.equal(solver.initial_points[1], torch.tensor(-1.5))


def test_cached_warm_start_is_detached_from_the_previous_graph(
    nominal: torch.Tensor,
) -> None:
    solver = _RecordingSolver()
    ambiguity_set = _dual_ambiguity_set(nominal, solver)
    loss = torch.tensor([1.0, 2.0], requires_grad=True)

    ambiguity_set.worst_case_expectation(loss)

    warm_start = dict(ambiguity_set.named_buffers())["_dual_warm_start"]
    assert not warm_start.requires_grad


def test_dual_point_buffers_move_with_the_module(nominal: torch.Tensor) -> None:
    solver = _RecordingSolver()
    ambiguity_set = _dual_ambiguity_set(nominal, solver)
    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))

    ambiguity_set = ambiguity_set.to(dtype=torch.float64)

    buffers = dict(ambiguity_set.named_buffers())
    assert buffers["initial_dual_point"].dtype == torch.float64
    assert buffers["_dual_warm_start"].dtype == torch.float64


def test_warm_start_cache_stays_out_of_the_state_dict(nominal: torch.Tensor) -> None:
    # The cache is solver scratch space, not configuration: persisting it
    # would make a checkpoint fail to load into a freshly built set.
    solver = _RecordingSolver()
    ambiguity_set = _dual_ambiguity_set(nominal, solver)
    ambiguity_set.worst_case_expectation(torch.tensor([1.0, 2.0]))

    state_dict = ambiguity_set.state_dict()

    assert "initial_dual_point" in state_dict
    assert "_dual_warm_start" not in state_dict
