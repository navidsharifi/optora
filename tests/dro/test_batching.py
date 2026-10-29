"""Tests for batched losses and batched radii across every ambiguity set."""

from collections.abc import Callable
from typing import Any, Protocol

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    TotalVariationAmbiguitySet,
)
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from tests.dro.specs import cvar_level

NOMINAL = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)

COST = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
)

BATCHED_LOSS = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.5, 3.0], [2.0, 0.5, 1.0]], dtype=torch.float64
)

# Two valid nominals stacked along a leading batch dimension, for checking
# that a nominal may carry more batch dimensions than the loss.
BATCHED_NOMINAL = torch.stack(
    [NOMINAL, torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)]
)


class AmbiguitySetFactory(Protocol):
    """Build an ambiguity set from a radius and an optional nominal."""

    def __call__(
        self, radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
    ) -> AmbiguitySet: ...


def _kl(radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL) -> AmbiguitySet:
    return KLAmbiguitySet(nominal, radius=radius)


def _chi_square(
    radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(nominal, radius=radius)


def _total_variation(
    radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(nominal, radius=radius)


def _cvar(
    radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
) -> AmbiguitySet:
    return CVaRAmbiguitySet(nominal, alpha=cvar_level(radius))


def _wasserstein(
    radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
) -> AmbiguitySet:
    return WassersteinAmbiguitySet(nominal, cost=COST, radius=radius)


def _sinkhorn(
    radius: float | torch.Tensor, nominal: torch.Tensor = NOMINAL
) -> AmbiguitySet:
    return SinkhornAmbiguitySet(nominal, cost=COST, epsilon=0.5, radius=radius)


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "kl": _kl,
    "chi_square": _chi_square,
    "total_variation": _total_variation,
    "cvar": _cvar,
    "wasserstein": _wasserstein,
    "sinkhorn": _sinkhorn,
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


def test_batched_nominal_broadcasts_against_an_unbatched_loss(
    factory: AmbiguitySetFactory,
) -> None:
    # Issue #66: the nominal may carry more batch dimensions than the loss.
    loss = BATCHED_LOSS[0]
    expected = torch.stack(
        [
            factory(0.2, nominal).worst_case_expectation(loss)
            for nominal in BATCHED_NOMINAL
        ]
    )

    result = factory(0.2, BATCHED_NOMINAL).worst_case_expectation(loss)

    assert result.shape == (2,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_batched_nominal_and_loss_broadcast_to_their_outer_product(
    factory: AmbiguitySetFactory,
) -> None:
    # A nominal batch of shape (2, 1) against a loss batch of shape (3,)
    # broadcasts to (2, 3), each entry solved independently.
    expected = torch.stack(
        [
            torch.stack(
                [
                    factory(0.2, nominal).worst_case_expectation(row)
                    for row in BATCHED_LOSS
                ]
            )
            for nominal in BATCHED_NOMINAL
        ]
    )

    result = factory(0.2, BATCHED_NOMINAL.unsqueeze(1)).worst_case_expectation(
        BATCHED_LOSS
    )

    assert result.shape == (2, 3)
    assert torch.allclose(result, expected, atol=1e-6)


def test_batched_nominal_gradient_is_the_per_nominal_maximizer(
    factory: AmbiguitySetFactory,
) -> None:
    # Summing over the nominal batch accumulates one maximizer per nominal
    # into the shared loss's gradient.
    shared_loss = BATCHED_LOSS[0].clone().requires_grad_(True)
    factory(0.2, BATCHED_NOMINAL).worst_case_expectation(shared_loss).sum().backward()

    expected = torch.zeros_like(BATCHED_LOSS[0])
    for nominal in BATCHED_NOMINAL:
        single_loss = BATCHED_LOSS[0].clone().requires_grad_(True)
        factory(0.2, nominal).worst_case_expectation(single_loss).backward()
        assert single_loss.grad is not None
        expected += single_loss.grad

    assert shared_loss.grad is not None
    assert torch.allclose(shared_loss.grad, expected, atol=1e-6)


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


def test_a_later_call_on_a_different_batch_shape_is_unaffected() -> None:
    # Nothing is cached between calls, so evaluating a batch and then a
    # single row must give the single row exactly what it gets on its own.
    ambiguity_set = _kl(0.2)

    ambiguity_set.worst_case_expectation(BATCHED_LOSS)
    result = ambiguity_set.worst_case_expectation(BATCHED_LOSS[0])

    assert result.shape == ()
    assert result == _kl(0.2).worst_case_expectation(BATCHED_LOSS[0])


@pytest.mark.parametrize(
    "make",
    [_kl, _chi_square, _total_variation, _cvar, _sinkhorn],
    ids=["kl", "chi_square", "total_variation", "cvar", "sinkhorn"],
)
def test_batched_solver_free_sets_never_synchronize(
    make: AmbiguitySetFactory, host_sync_counter: Callable[[], Any]
) -> None:
    # The tilt bisection and the total-variation and CVaR closed forms are
    # solver-free and run a trip count fixed by the dtype, so a batched
    # evaluation must stay a pure sequence of tensor kernels with no
    # device-to-host read at all.
    ambiguity_set = make(torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64))

    with host_sync_counter() as syncs:
        ambiguity_set.worst_case_expectation(BATCHED_LOSS)

    assert len(syncs) == 0
