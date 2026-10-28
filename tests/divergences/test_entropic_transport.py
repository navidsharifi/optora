"""Tests for `EntropicTransportDivergence`."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.divergence_base import Divergence
from optora.divergences.entropic_transport import EntropicTransportDivergence

POINTS = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
COST = torch.abs(POINTS.unsqueeze(-1) - POINTS.unsqueeze(-2))
NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
CANDIDATE = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)


def _smoothed(
    divergence: EntropicTransportDivergence, nominal: torch.Tensor
) -> torch.Tensor:
    kernel = torch.exp(divergence.log_kernel(nominal.dtype))
    return nominal @ kernel


def test_vanishes_at_the_kernel_smoothed_reference() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)

    value = divergence(_smoothed(divergence, NOMINAL), NOMINAL)

    assert float(value) == pytest.approx(0.0, abs=1e-12)


def test_does_not_vanish_at_the_reference_itself() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)

    assert float(divergence(NOMINAL, NOMINAL)) > 1e-3


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
def test_a_zero_cost_reduces_to_kl_from_the_normalized_reference(
    epsilon: float,
) -> None:
    # Every kernel row is nu, and the product coupling p x candidate attains
    # min KL(gamma || p x nu) = KL(candidate || nu).
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    divergence = EntropicTransportDivergence(
        torch.zeros_like(COST), epsilon=epsilon, reference=reference
    )
    nu = reference / reference.sum()

    value = divergence(CANDIDATE, NOMINAL)

    expected = epsilon * float(torch.sum(CANDIDATE * torch.log(CANDIDATE / nu)))
    assert float(value) == pytest.approx(expected, rel=1e-9)


def test_log_kernel_rows_are_distributions_with_reference_zeros() -> None:
    reference = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    divergence = EntropicTransportDivergence(COST, epsilon=1e-3, reference=reference)

    log_kernel = divergence.log_kernel(torch.float64)

    torch.testing.assert_close(
        torch.exp(log_kernel).sum(-1), torch.ones(4, dtype=torch.float64)
    )
    assert bool(torch.all(log_kernel[:, 1] == -torch.inf))
    assert bool(torch.all(torch.isfinite(log_kernel[:, [0, 2, 3]])))


def test_is_positive_away_from_the_centre_and_batches() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)
    candidates = torch.stack([CANDIDATE, CANDIDATE.flip(-1), NOMINAL])

    batched = divergence(candidates, NOMINAL)

    assert batched.shape == (3,)
    assert bool(torch.all(batched > 0))
    for index in range(3):
        assert float(batched[index]) == pytest.approx(
            float(divergence(candidates[index], NOMINAL)), rel=1e-9
        )


def test_converged_potentials_honour_the_sync_budget(
    host_sync_counter: Callable[[], Any],
) -> None:
    divergence = EntropicTransportDivergence(
        COST, epsilon=0.5, max_iter=40, tol=1e-300, check_interval=10
    )

    with host_sync_counter() as syncs:
        divergence(CANDIDATE, NOMINAL)

    assert len(syncs) == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epsilon": 0.0}, "epsilon must be positive"),
        ({"epsilon": math.nan}, "epsilon must be positive"),
        ({"epsilon": 1.0, "tol": 0.0}, "tol must be positive"),
        ({"epsilon": 1.0, "eps": -1.0}, "eps must be positive"),
        ({"epsilon": 1.0, "max_iter": 0}, "max_iter must be positive"),
        ({"epsilon": 1.0, "reference": torch.ones(3)}, "reference must have shape"),
    ],
)
def test_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EntropicTransportDivergence(COST, **kwargs)


def test_rejects_a_non_square_cost() -> None:
    with pytest.raises(ValueError, match="square"):
        EntropicTransportDivergence(torch.zeros(3, 4), epsilon=1.0)


def test_validate_rejects_a_non_finite_cost() -> None:
    cost = COST.clone()
    cost[0, 1] = math.inf

    with pytest.raises(ValueError, match="cost must be finite"):
        EntropicTransportDivergence(cost, epsilon=1.0, validate=True)


def test_rejects_mismatched_distribution_sizes() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=1.0)

    with pytest.raises(ValueError, match="to index cost"):
        divergence(torch.ones(3) / 3, NOMINAL)


def test_is_a_divergence_instance() -> None:
    assert isinstance(EntropicTransportDivergence(COST, epsilon=1.0), Divergence)
