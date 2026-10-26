"""Tests for radius and loss validation shared by every ambiguity set.

A negative or `nan` float radius is metadata and is always rejected. Checks
that read tensor values back to the host (a tensor radius at construction, the
loss at every evaluation) run only with `validate=True`.
"""

import math
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
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.instances import INSTANCES
from tests.dro.specs import SetSpec, spec_parameters

NOMINAL = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)

COST = torch.tensor(
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
)

LOSS = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

INVALID_TENSOR_RADII = {
    "nan": torch.tensor(math.nan, dtype=torch.float64),
    "negative": torch.tensor(-0.1, dtype=torch.float64),
    "negative_entry": torch.tensor([0.1, -0.1], dtype=torch.float64),
    "nan_entry": torch.tensor([0.1, math.nan], dtype=torch.float64),
}

NON_FINITE_LOSSES = {
    "nan": torch.tensor([0.0, math.nan, 2.0], dtype=torch.float64),
    "positive_infinity": torch.tensor([0.0, 1.0, math.inf], dtype=torch.float64),
    "negative_infinity": torch.tensor([-math.inf, 1.0, 2.0], dtype=torch.float64),
}

Radius = float | torch.Tensor

AmbiguitySetFactory = Callable[[Radius, bool], AmbiguitySet]


def _kl(radius: Radius, validate: bool) -> AmbiguitySet:
    return KLAmbiguitySet(NOMINAL, radius=radius, validate=validate)


def _phi(radius: Radius, validate: bool) -> AmbiguitySet:
    return PhiAmbiguitySet(
        NOMINAL,
        divergence=ChiSquareDivergence(),
        radius=radius,
        phi_conjugate=torch.exp,
        dual_solver=GradientDescent(step_size=0.05, max_iter=20, tol=1e-8),
        validate=validate,
    )


def _chi_square(radius: Radius, validate: bool) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(NOMINAL, radius=radius, validate=validate)


def _total_variation(radius: Radius, validate: bool) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(NOMINAL, radius=radius, validate=validate)


def _wasserstein(radius: Radius, validate: bool) -> AmbiguitySet:
    return WassersteinAmbiguitySet(NOMINAL, cost=COST, radius=radius, validate=validate)


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "chi_square": _chi_square,
    "kl": _kl,
    "phi": _phi,
    "total_variation": _total_variation,
    "wasserstein": _wasserstein,
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> AmbiguitySetFactory:
    """Build one ambiguity set of each kind from a radius and a validate flag."""
    return FACTORIES[str(request.param)]


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("radius", [-0.1, -math.inf, math.nan])
def test_an_invalid_float_radius_is_always_rejected(
    factory: AmbiguitySetFactory, radius: float, validate: bool
) -> None:
    with pytest.raises(ValueError, match="radius must be nonnegative"):
        factory(radius, validate)


@spec_parameters
def test_an_infinite_float_radius_reaches_the_maximum_loss(spec: SetSpec) -> None:
    nominal, loss, cost = INSTANCES[0].tensors()

    value = spec.build(nominal, cost, math.inf).worst_case_expectation(loss)

    assert float(value) == float(torch.amax(loss))


@pytest.mark.parametrize("name", sorted(INVALID_TENSOR_RADII))
def test_validate_rejects_an_invalid_tensor_radius(
    factory: AmbiguitySetFactory, name: str
) -> None:
    with pytest.raises(ValueError, match="radius must be nonnegative and not nan"):
        factory(INVALID_TENSOR_RADII[name], True)


@pytest.mark.parametrize("name", sorted(INVALID_TENSOR_RADII))
def test_an_invalid_tensor_radius_is_accepted_when_validation_is_off(
    factory: AmbiguitySetFactory, name: str, host_sync_counter: Callable[[], Any]
) -> None:
    radius = INVALID_TENSOR_RADII[name]

    with host_sync_counter() as syncs:
        ambiguity_set = factory(radius, False)

    assert ambiguity_set.radius is radius
    assert len(syncs) == 0


@pytest.mark.parametrize(
    "radius",
    [
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor([0.0, 0.1, math.inf], dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
    ],
    ids=["zero", "sweep_with_infinity", "empty"],
)
def test_validate_accepts_a_valid_tensor_radius(
    factory: AmbiguitySetFactory, radius: torch.Tensor
) -> None:
    ambiguity_set = factory(radius, True)

    assert ambiguity_set.radius is radius


@pytest.mark.parametrize("name", sorted(NON_FINITE_LOSSES))
@pytest.mark.parametrize("radius", [0.0, 0.1])
def test_validate_rejects_a_non_finite_loss(
    factory: AmbiguitySetFactory, name: str, radius: float
) -> None:
    ambiguity_set = factory(radius, True)

    with pytest.raises(ValueError, match="loss must be finite"):
        ambiguity_set.worst_case_expectation(NON_FINITE_LOSSES[name])


def test_validate_accepts_a_finite_loss(factory: AmbiguitySetFactory) -> None:
    value = factory(0.1, True).worst_case_expectation(LOSS)

    assert torch.isfinite(value)


def test_loss_validation_costs_one_host_synchronization_per_evaluation(
    factory: AmbiguitySetFactory, host_sync_counter: Callable[[], Any]
) -> None:
    with host_sync_counter() as unvalidated:
        factory(0.1, False).worst_case_expectation(LOSS)
    validated_set = factory(0.1, True)
    with host_sync_counter() as validated:
        validated_set.worst_case_expectation(LOSS)

    assert len(validated) == len(unvalidated) + 1


def test_validate_is_stored_on_the_ambiguity_set(factory: AmbiguitySetFactory) -> None:
    assert factory(0.1, True).validate
    assert not factory(0.1, False).validate
