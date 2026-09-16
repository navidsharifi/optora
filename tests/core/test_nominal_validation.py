"""Tests for opt-in validation of an ambiguity set's nominal distribution."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.f_divergence import ChiSquareDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
)
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet

COST = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
)

VALID = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)

NEGATIVE = torch.tensor([-0.25, 1.0, 0.25], dtype=torch.float64)

UNNORMALIZED = torch.tensor([0.25, 0.5, 0.5], dtype=torch.float64)

NOT_A_NUMBER = torch.tensor([float("nan"), 0.5, 0.5], dtype=torch.float64)

INFINITE = torch.tensor([float("inf"), 0.5, 0.5], dtype=torch.float64)

INVALID = {
    "negative": NEGATIVE,
    "unnormalized": UNNORMALIZED,
    "nan": NOT_A_NUMBER,
    "inf": INFINITE,
}

AmbiguitySetFactory = Callable[[torch.Tensor, bool], AmbiguitySet]


def _kl(nominal: torch.Tensor, validate: bool) -> AmbiguitySet:
    return KLAmbiguitySet(nominal, radius=0.1, validate=validate)


def _phi(nominal: torch.Tensor, validate: bool) -> AmbiguitySet:
    return PhiAmbiguitySet(
        nominal,
        divergence=ChiSquareDivergence(),
        radius=0.1,
        phi_conjugate=torch.exp,
        validate=validate,
    )


def _chi_square(nominal: torch.Tensor, validate: bool) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(nominal, radius=0.1, validate=validate)


def _total_variation(nominal: torch.Tensor, validate: bool) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(nominal, radius=0.1, validate=validate)


def _wasserstein(nominal: torch.Tensor, validate: bool) -> AmbiguitySet:
    return WassersteinAmbiguitySet(nominal, cost=COST, radius=0.1, validate=validate)


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "chi_square": _chi_square,
    "kl": _kl,
    "phi": _phi,
    "total_variation": _total_variation,
    "wasserstein": _wasserstein,
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> AmbiguitySetFactory:
    """Build one ambiguity set of each kind from a nominal and a validate flag."""
    return FACTORIES[str(request.param)]


@pytest.fixture(params=sorted(INVALID))
def invalid_nominal(request: pytest.FixtureRequest) -> torch.Tensor:
    """Iterate over the ways a nominal can fail to be a distribution."""
    return INVALID[str(request.param)]


def test_validate_accepts_a_valid_nominal(factory: AmbiguitySetFactory) -> None:
    ambiguity_set = factory(VALID, True)

    assert ambiguity_set.nominal is VALID


def test_validate_rejects_an_invalid_nominal(
    factory: AmbiguitySetFactory, invalid_nominal: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="nominal"):
        factory(invalid_nominal, True)


def test_an_invalid_nominal_is_accepted_when_validation_is_off(
    factory: AmbiguitySetFactory, invalid_nominal: torch.Tensor
) -> None:
    # Validation is opt-in, so the default path must never read the tensor.
    ambiguity_set = factory(invalid_nominal, False)

    assert ambiguity_set.nominal is invalid_nominal


def test_a_scalar_nominal_is_always_rejected(factory: AmbiguitySetFactory) -> None:
    # A scalar has no support dimension; the check is metadata-only, so it
    # costs nothing and runs whether or not validation is requested.
    with pytest.raises(ValueError, match="trailing support"):
        factory(torch.tensor(1.0, dtype=torch.float64), False)


def test_an_integer_nominal_is_rejected_by_validation() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        KLAmbiguitySet(torch.tensor([0, 1]), radius=0.1, validate=True)


def test_float32_rounding_error_is_within_tolerance() -> None:
    nominal = torch.full((1000,), 1.0 / 1000.0, dtype=torch.float32)

    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1, validate=True)

    assert ambiguity_set.nominal is nominal


def test_construction_is_free_of_host_synchronization_by_default(
    factory: AmbiguitySetFactory, host_sync_counter: Callable[[], Any]
) -> None:
    with host_sync_counter() as syncs:
        factory(VALID, False)

    assert len(syncs) == 0


def test_validation_costs_a_bounded_number_of_host_synchronizations(
    factory: AmbiguitySetFactory, host_sync_counter: Callable[[], Any]
) -> None:
    # Two reductions for the nominal, plus one for a Wasserstein set's cost.
    with host_sync_counter() as syncs:
        factory(VALID, True)

    assert len(syncs) <= 3


def test_validation_preserves_buffer_identity_and_autograd() -> None:
    nominal = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64, requires_grad=True)

    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=0.1, validate=True)

    assert ambiguity_set.nominal is nominal
    assert ambiguity_set.nominal.requires_grad
