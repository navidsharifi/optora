"""Tests for batched losses and batched radii across every ambiguity set."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent

NOMINAL = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)

COST = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
)

BATCHED_LOSS = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.5, 3.0], [2.0, 0.5, 1.0]], dtype=torch.float64
)

AmbiguitySetFactory = Callable[[float | torch.Tensor], AmbiguitySet]


def _kl(radius: float | torch.Tensor) -> AmbiguitySet:
    return KLAmbiguitySet(
        NOMINAL,
        radius=radius,
        dual_solver=GradientDescent(step_size=0.1, max_iter=2000, tol=1e-10),
    )


def _chi_square(radius: float | torch.Tensor) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(
        NOMINAL,
        radius=radius,
        dual_solver=GradientDescent(step_size=0.05, max_iter=3000, tol=1e-10),
    )


def _total_variation(radius: float | torch.Tensor) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(NOMINAL, radius=radius)


def _wasserstein(radius: float | torch.Tensor) -> AmbiguitySet:
    return WassersteinAmbiguitySet(
        NOMINAL,
        cost=COST,
        radius=radius,
        dual_solver=GradientDescent(step_size=0.05, max_iter=600, tol=1e-10),
    )


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "kl": _kl,
    "chi_square": _chi_square,
    "total_variation": _total_variation,
    "wasserstein": _wasserstein,
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> AmbiguitySetFactory:
    """Build one ambiguity set of each kind from a radius."""
    return FACTORIES[str(request.param)]


def _per_row_reference(
    factory: AmbiguitySetFactory, radius: float, loss: torch.Tensor
) -> torch.Tensor:
    """Evaluate `loss`'s rows one at a time, each on a freshly built set."""
    return torch.stack(
        [factory(radius).worst_case_expectation(row) for row in loss.unbind(dim=0)]
    )


def test_batched_loss_matches_a_per_row_reference(
    factory: AmbiguitySetFactory,
) -> None:
    # The duals decouple across batch elements, so one joint solve must agree
    # elementwise with independent cold solves of each row.
    expected = _per_row_reference(factory, 0.2, BATCHED_LOSS)

    result = factory(0.2).worst_case_expectation(BATCHED_LOSS)

    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_unbatched_loss_still_returns_a_scalar(factory: AmbiguitySetFactory) -> None:
    result = factory(0.2).worst_case_expectation(BATCHED_LOSS[0])

    assert result.shape == ()


def test_leading_batch_dimensions_are_preserved(
    factory: AmbiguitySetFactory,
) -> None:
    loss = BATCHED_LOSS[:2].reshape(1, 2, 3)

    result = factory(0.2).worst_case_expectation(loss)

    assert result.shape == (1, 2)
    assert torch.allclose(
        result.reshape(2), factory(0.2).worst_case_expectation(BATCHED_LOSS[:2])
    )


def test_batched_radius_sweeps_one_loss_in_a_single_call(
    factory: AmbiguitySetFactory,
) -> None:
    radii = [0.05, 0.2, 0.6]
    loss = BATCHED_LOSS[0]
    expected = torch.stack(
        [factory(radius).worst_case_expectation(loss) for radius in radii]
    )

    result = factory(torch.tensor(radii, dtype=torch.float64)).worst_case_expectation(
        loss
    )

    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_batched_radius_broadcasts_against_a_batched_loss(
    factory: AmbiguitySetFactory,
) -> None:
    radii = torch.tensor([0.05, 0.2, 0.6], dtype=torch.float64)

    result = factory(radii).worst_case_expectation(BATCHED_LOSS)

    expected = torch.stack(
        [
            factory(float(radius)).worst_case_expectation(row)
            for radius, row in zip(radii, BATCHED_LOSS.unbind(dim=0), strict=True)
        ]
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_batched_gradients_match_a_per_row_reference(
    factory: AmbiguitySetFactory,
) -> None:
    # Batched evaluation is only useful for minibatch training if the
    # envelope-theorem gradient survives batching unchanged.
    batched_loss = BATCHED_LOSS.clone().requires_grad_(True)
    factory(0.2).worst_case_expectation(batched_loss).sum().backward()

    expected = []
    for row in BATCHED_LOSS.unbind(dim=0):
        single_loss = row.clone().requires_grad_(True)
        factory(0.2).worst_case_expectation(single_loss).backward()
        assert single_loss.grad is not None
        expected.append(single_loss.grad)

    assert batched_loss.grad is not None
    assert torch.allclose(batched_loss.grad, torch.stack(expected), atol=1e-6)


def test_mismatched_support_size_raises_value_error(
    factory: AmbiguitySetFactory,
) -> None:
    loss = torch.zeros(4, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match="matching nominal's support size"):
        factory(0.2).worst_case_expectation(loss)


def test_radius_batch_that_does_not_broadcast_raises_value_error(
    factory: AmbiguitySetFactory,
) -> None:
    ambiguity_set = factory(torch.tensor([0.1, 0.2], dtype=torch.float64))

    with pytest.raises(ValueError, match="do not broadcast"):
        ambiguity_set.worst_case_expectation(BATCHED_LOSS)


def test_warm_start_cache_is_rebuilt_when_the_batch_shape_changes() -> None:
    # A cached dual optimum from a differently shaped batch cannot be
    # expanded onto the new one, so the next solve falls back to cold start.
    ambiguity_set = _kl(0.2)

    ambiguity_set.worst_case_expectation(BATCHED_LOSS)
    result = ambiguity_set.worst_case_expectation(BATCHED_LOSS[0])

    assert result.shape == ()
    assert torch.allclose(
        result, _kl(0.2).worst_case_expectation(BATCHED_LOSS[0]), atol=1e-6
    )


def test_total_variation_batched_closed_form_never_synchronizes(
    host_sync_counter: Callable[[], Any],
) -> None:
    # The closed form is solver-free, so a batched evaluation must stay a
    # pure sequence of tensor kernels with no device-to-host read at all.
    ambiguity_set = _total_variation(torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64))

    with host_sync_counter() as syncs:
        ambiguity_set.worst_case_expectation(BATCHED_LOSS)

    assert len(syncs) == 0
