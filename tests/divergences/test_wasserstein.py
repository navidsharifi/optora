"""Tests for `SinkhornDivergence`."""

import pytest
import torch

from optora.divergences.wasserstein import SinkhornDivergence

TWO_POINT_COST = torch.tensor([[0.0, 1.0], [1.0, 0.0]])


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


def test_negative_cost_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]))


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


def test_invalid_eps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SinkhornDivergence(cost=TWO_POINT_COST, eps=0.0)


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


def test_gradient_is_finite_through_clamped_zero_entries() -> None:
    divergence = SinkhornDivergence(cost=TWO_POINT_COST)
    p = torch.tensor([0.0, 1.0], requires_grad=True)
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)
    value.backward()

    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))
