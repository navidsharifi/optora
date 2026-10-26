"""Registry of the ambiguity sets under test and their duality references.

`SPECS` is the single list of exact ambiguity sets that the property-based
suite (`tests/dro/test_properties.py`) and the gradient suite
(`tests/dro/test_gradients.py`) run against, so adding a set to the
verification suites means adding one `SetSpec` here.

`PhiAmbiguitySet` is excluded: its general fixed-step dual is a documented
fallback whose accuracy depends on the radius and step size, so it does not
satisfy the invariants or the gradient checks to a fixed tolerance.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from tests.dro.duality import (
    Array,
    chi_square_divergence,
    chi_square_dual,
    chi_square_dual_minimum,
    kl_divergence,
    kl_dual,
    kl_dual_minimum,
    total_variation_distance,
    total_variation_dual,
    total_variation_dual_minimum,
    transport_cost,
    wasserstein_dual,
    wasserstein_dual_minimum,
)
from tests.dro.instances import Instance

Radius = float | torch.Tensor


@dataclass(frozen=True)
class SetSpec:
    """An ambiguity set together with its independent duality references.

    Attributes:
        name: Identifier used as the pytest parameter id.
        build: Constructor taking `(nominal, cost, radius)`.
        dual: Dual objective at a point indexed by a real coordinate, taking
            `(instance, radius, coordinate)`.
        dual_minimum: Exact or numerically minimized dual, taking
            `(instance, radius)`.
        divergence: Primal divergence of a candidate from the nominal,
            taking `(instance, candidate)`.
        off_support: Whether the set can move mass onto scenarios with zero
            nominal weight (true for total variation and Wasserstein, false
            for KL and chi-square, which are infinite off the support).
    """

    name: str
    build: Callable[[torch.Tensor, torch.Tensor, Radius], AmbiguitySet]
    dual: Callable[[Instance, float, float], float]
    dual_minimum: Callable[[Instance, float], float]
    divergence: Callable[[Instance, Array], float]
    off_support: bool


SPECS = (
    SetSpec(
        name="kl",
        build=lambda nominal, _, radius: KLAmbiguitySet(nominal, radius),
        dual=lambda instance, radius, coordinate: kl_dual(
            instance.nominal_array,
            instance.loss_array,
            radius,
            float(np.exp(coordinate)),
        ),
        dual_minimum=lambda instance, radius: kl_dual_minimum(
            instance.nominal_array, instance.loss_array, radius
        ),
        divergence=lambda instance, candidate: kl_divergence(
            candidate, instance.nominal_array
        ),
        off_support=False,
    ),
    SetSpec(
        name="chi_square",
        build=lambda nominal, _, radius: ChiSquareAmbiguitySet(nominal, radius),
        dual=lambda instance, radius, coordinate: chi_square_dual(
            instance.nominal_array,
            instance.loss_array,
            radius,
            float(np.sinh(coordinate)),
        ),
        dual_minimum=lambda instance, radius: chi_square_dual_minimum(
            instance.nominal_array, instance.loss_array, radius
        ),
        divergence=lambda instance, candidate: chi_square_divergence(
            candidate, instance.nominal_array
        ),
        off_support=False,
    ),
    SetSpec(
        name="total_variation",
        build=lambda nominal, _, radius: TotalVariationAmbiguitySet(nominal, radius),
        dual=lambda instance, radius, coordinate: total_variation_dual(
            instance.nominal_array,
            instance.loss_array,
            radius,
            float(np.sinh(coordinate)),
        ),
        dual_minimum=lambda instance, radius: total_variation_dual_minimum(
            instance.nominal_array, instance.loss_array, radius
        ),
        divergence=lambda instance, candidate: total_variation_distance(
            candidate, instance.nominal_array
        ),
        off_support=True,
    ),
    SetSpec(
        name="wasserstein",
        build=lambda nominal, cost, radius: WassersteinAmbiguitySet(
            nominal, cost, radius
        ),
        dual=lambda instance, radius, coordinate: wasserstein_dual(
            instance.nominal_array,
            instance.cost_array,
            instance.loss_array,
            radius,
            float(np.exp(coordinate)),
        ),
        dual_minimum=lambda instance, radius: wasserstein_dual_minimum(
            instance.nominal_array, instance.cost_array, instance.loss_array, radius
        ),
        divergence=lambda instance, candidate: transport_cost(
            candidate, instance.nominal_array, instance.cost_array
        ),
        off_support=True,
    ),
)

spec_parameters = pytest.mark.parametrize(
    "spec", SPECS, ids=[spec.name for spec in SPECS]
)
