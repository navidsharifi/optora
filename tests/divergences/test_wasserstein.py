"""Tests for `SinkhornDivergence`."""

import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.special import logsumexp, xlogy
from torch.autograd import gradcheck

import optora.divergences.wasserstein as wasserstein
from optora.divergences.wasserstein import SinkhornDivergence
from tests.dro.duality import entropic_transport_divergence

TWO_POINT_COST = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

LINE = torch.linspace(0.0, 3.0, 6, dtype=torch.float64)
LINE_COST = torch.abs(LINE[:, None] - LINE[None, :])
P = torch.tensor([0.05, 0.3, 0.1, 0.25, 0.2, 0.1], dtype=torch.float64)
Q = torch.tensor([0.2, 0.1, 0.25, 0.05, 0.1, 0.3], dtype=torch.float64)
SPARSE_P = torch.tensor([0.0, 0.5, 0.1, 0.0, 0.4, 0.0], dtype=torch.float64)
SPARSE_Q = torch.tensor([0.3, 0.0, 0.0, 0.3, 0.1, 0.3], dtype=torch.float64)


def _reference_cost(p: np.ndarray, q: np.ndarray, epsilon: float) -> float:
    """Return OT_eps(p, q) up to the terms debiasing cancels, via SciPy.

    `entropic_transport_divergence` maximizes the semi-dual in the potentials
    h = g / eps + log q against the row-normalized kernel, independently of
    optora; undoing that change of variables gives the dual value.
    """
    scaled = LINE_COST.numpy() / epsilon
    normalizer = logsumexp(-scaled, axis=1)
    kernel = np.exp(-scaled - normalizer[:, None])
    semi_dual = entropic_transport_divergence(q, p, kernel, epsilon)
    return (
        semi_dual - epsilon * float(xlogy(q, q).sum()) - epsilon * float(p @ normalizer)
    )


def _reference(p: torch.Tensor, q: torch.Tensor, epsilon: float) -> float:
    p_array, q_array = p.numpy(), q.numpy()
    return (
        _reference_cost(p_array, q_array, epsilon)
        - 0.5 * _reference_cost(p_array, p_array, epsilon)
        - 0.5 * _reference_cost(q_array, q_array, epsilon)
    )


def test_zero_for_identical_distributions() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.25, 0.75])

    value = divergence(p, p)

    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


def test_nonnegative_for_distinct_distributions() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.9, 0.1])

    value = divergence(p, q)

    assert value >= 0.0


def test_symmetric_for_symmetric_cost() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.3, 0.7])

    assert torch.allclose(divergence(p, q), divergence(q, p), atol=1e-6)


def test_approximates_exact_wasserstein_distance_for_small_epsilon() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST, epsilon=0.05, max_iter=2000)
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])

    value = divergence(p, q)

    assert torch.allclose(value, torch.tensor(1.0), atol=0.05)


def test_handles_zero_entries_without_producing_nan_or_inf() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)

    assert torch.isfinite(value)


def test_non_square_cost_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=torch.zeros(2, 3))


def test_non_2d_cost_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=torch.zeros(3))


def test_negative_cost_raises_value_error_when_validation_is_requested() -> None:
    with pytest.raises(ValueError, match="cost must be nonnegative"):
        SinkhornDivergence(cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]), validate=True)


def test_negative_cost_is_not_inspected_by_default() -> None:
    SinkhornDivergence(cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]))


def test_construction_does_not_synchronize_by_default(
    host_sync_counter: Callable[[], Any],
) -> None:
    with host_sync_counter() as syncs:
        SinkhornDivergence(cost=TWO_POINT_COST)

    assert len(syncs) == 0


def test_invalid_epsilon_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=TWO_POINT_COST, epsilon=0.0)
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=TWO_POINT_COST, epsilon=-1.0)


def test_invalid_max_iter_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=TWO_POINT_COST, max_iter=0)


def test_invalid_tol_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=TWO_POINT_COST, tol=0.0)


def test_invalid_newton_max_iter_raises_value_error() -> None:
    with pytest.raises(ValueError, match="newton_max_iter must be nonnegative"):
        SinkhornDivergence(cost=TWO_POINT_COST, newton_max_iter=-1)


def test_eps_is_deprecated_and_ignored() -> None:
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])

    with pytest.warns(DeprecationWarning, match="eps is deprecated"):
        deprecated = SinkhornDivergence(cost=TWO_POINT_COST, eps=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = SinkhornDivergence(cost=TWO_POINT_COST)

    assert torch.equal(deprecated(p, q), default(p, q))


def test_invalid_check_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="check_interval must be positive"):
        SinkhornDivergence(cost=TWO_POINT_COST, check_interval=0)


@pytest.mark.parametrize("check_interval", [1, 7, 1000])
def test_check_interval_does_not_change_the_divergence(check_interval: int) -> None:
    p = torch.tensor([0.1, 0.9], dtype=torch.float64)
    q = torch.tensor([0.7, 0.3], dtype=torch.float64)
    cost = TWO_POINT_COST.to(torch.float64)
    reference = SinkhornDivergence(cost=cost, max_iter=200, check_interval=1)
    divergence = SinkhornDivergence(
        cost=cost, max_iter=200, check_interval=check_interval
    )

    assert torch.equal(divergence(p, q), reference(p, q))


def test_sinkhorn_loop_does_not_synchronize_on_every_iteration(
    host_sync_counter: Callable[[], Any],
) -> None:
    divergence = SinkhornDivergence(
        cost=TWO_POINT_COST,
        tol=1e-30,
        max_iter=60,
        check_interval=20,
        newton_max_iter=0,
    )
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.7, 0.3])

    with host_sync_counter() as syncs:
        divergence(p, q)

    # One batched Sinkhorn loop over the three debiasing transport problems,
    # checking convergence every 20 of its 60 iterations: a third of the
    # reads the three sequential loops used to cost.
    assert len(syncs) == 60 // 20


def test_debiasing_runs_one_batched_sinkhorn_loop() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.7, 0.3])
    batch_shapes = []
    solve = wasserstein.solve_semi_dual

    def recording(rows: torch.Tensor, *args: Any) -> torch.Tensor:
        batch_shapes.append(tuple(rows.shape[:-1]))
        return solve(rows, *args)

    with patch.object(wasserstein, "solve_semi_dual", recording):
        divergence(p, q)

    assert batch_shapes == [(3,)]


def test_identical_distributions_are_exactly_zero() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST, max_iter=3)
    p = torch.tensor([0.25, 0.75])

    # The three debiasing legs are the same problem in the same batched
    # kernels, so they cancel bitwise even before the loop converges.
    assert torch.equal(divergence(p, p), torch.zeros(()))


def test_batched_distributions_match_elementwise_divergences() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST, max_iter=200)
    p = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    q = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])

    value = divergence(p, q)

    assert value.shape == (3,)
    expected = torch.stack(
        [divergence(row, column) for row, column in zip(p, q, strict=True)]
    )
    assert torch.allclose(value, expected, atol=1e-6)


def test_batched_p_broadcasts_against_a_single_q() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST, max_iter=200)
    p = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    q = torch.tensor([0.7, 0.3])

    value = divergence(p, q)

    assert value.shape == (3,)
    expected = torch.stack([divergence(row, q) for row in p])
    assert torch.allclose(value, expected, atol=1e-6)


def test_batched_solve_does_not_synchronize_per_batch_element(
    host_sync_counter: Callable[[], Any],
) -> None:
    divergence = SinkhornDivergence(
        cost=TWO_POINT_COST,
        tol=1e-30,
        max_iter=60,
        check_interval=20,
        newton_max_iter=0,
    )
    p = torch.rand(16, 2)
    p = p / p.sum(dim=-1, keepdim=True)
    q = torch.tensor([0.7, 0.3])

    with host_sync_counter() as syncs:
        divergence(p, q)

    assert len(syncs) == 60 // 20


def test_mismatched_shape_raises_value_error() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.9])

    with pytest.raises(ValueError):
        divergence(p, q)
    with pytest.raises(ValueError):
        divergence(q, p)


def test_is_a_divergence_instance() -> None:
    from optora.core.divergence_base import Divergence

    assert isinstance(SinkhornDivergence(cost=TWO_POINT_COST), Divergence)


# --- The dual value and its stopping rule (issue #76) ----------------------


@pytest.mark.parametrize("epsilon", [0.02, 0.05, 0.3, 1.0])
@pytest.mark.parametrize(
    ("p", "q"),
    [(P, Q), (SPARSE_P, Q), (P, SPARSE_Q), (SPARSE_P, SPARSE_Q)],
    ids=["dense", "zero_rows", "zero_columns", "both"],
)
def test_matches_the_debiased_entropic_dual(
    p: torch.Tensor, q: torch.Tensor, epsilon: float
) -> None:
    # At epsilon = 0.02 the potential change falls below tol long before the
    # plan meets its marginals, where the former stopping rule returned the
    # cost of an unconverged plan; the Newton polish runs on these instances.
    divergence = SinkhornDivergence(LINE_COST, epsilon=epsilon, tol=1e-12)

    value = divergence(p, q)

    assert float(value) == pytest.approx(_reference(p, q, epsilon), rel=1e-9)


def test_default_tolerance_is_second_order_accurate() -> None:
    # The dual value is off by the square of the marginal violation, so the
    # default tol of 1e-6 still gives a value far more accurate than 1e-6.
    divergence = SinkhornDivergence(LINE_COST, epsilon=0.05)

    value = divergence(P, Q)

    assert float(value) == pytest.approx(_reference(P, Q, 0.05), rel=1e-7)


def test_batched_problems_match_one_at_a_time() -> None:
    divergence = SinkhornDivergence(LINE_COST, epsilon=0.05, tol=1e-12)
    candidates = torch.stack([P, SPARSE_P, P])
    references = torch.stack([Q, Q, SPARSE_Q])

    values = divergence(candidates, references)

    for index in range(3):
        assert float(values[index]) == pytest.approx(
            _reference(candidates[index], references[index], 0.05), rel=1e-9
        )


@pytest.mark.parametrize("epsilon", [0.1, 1.0])
def test_gradient_is_the_envelope_gradient_of_the_converged_dual(
    epsilon: float,
) -> None:
    # The divergence is defined on distributions, so the gradient is checked
    # along the simplex, through a softmax parametrization of each marginal.
    divergence = SinkhornDivergence(LINE_COST, epsilon=epsilon, tol=1e-12)

    def on_simplex(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        return divergence(torch.softmax(p_logits, -1), torch.softmax(q_logits, -1))

    p_logits = torch.log(P).requires_grad_(True)
    q_logits = torch.log(Q).requires_grad_(True)
    assert gradcheck(on_simplex, (p_logits, q_logits), atol=1e-6)


def test_gradient_is_finite_through_zero_entries() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.0, 1.0], requires_grad=True)
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)
    value.backward()

    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))
