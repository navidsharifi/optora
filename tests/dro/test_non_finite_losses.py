"""Tests for how every ambiguity set treats a `nan` or infinite loss.

Without `validate=True` a non-finite loss is outside the domain but must
never yield a finite number: the affected batch element is `nan` or
infinite, and every other batch element is exactly what it would be alone.
"""

import math
from collections.abc import Callable

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.f_divergence import ChiSquareDivergence
from optora.dro.phi_dro import PhiAmbiguitySet, chi_square_conjugate
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.instances import INSTANCES
from tests.dro.specs import SPECS, SetSpec, spec_parameters

NOMINAL, LOSS, COST = INSTANCES[0].tensors()

NON_FINITE_VALUES = {
    "nan": math.nan,
    "positive_infinity": math.inf,
    "negative_infinity": -math.inf,
}

# The lowest and the highest loss of LOSS.
POSITIONS = (0, 3)

Builder = Callable[[torch.Tensor, torch.Tensor, float], AmbiguitySet]


def _phi(nominal: torch.Tensor, _: torch.Tensor, radius: float) -> AmbiguitySet:
    return PhiAmbiguitySet(
        nominal,
        divergence=ChiSquareDivergence(),
        radius=radius,
        phi_conjugate=chi_square_conjugate,
        dual_solver=GradientDescent(step_size=0.1, max_iter=50, tol=1e-8),
    )


BUILDERS: dict[str, Builder] = {spec.name: spec.build for spec in SPECS} | {"phi": _phi}


def _with_entry(position: int, entry: float) -> torch.Tensor:
    loss = LOSS.clone()
    loss[position] = entry
    return loss


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("entry", sorted(NON_FINITE_VALUES))
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("radius", [0.0, 0.3, 30.0])
def test_a_non_finite_loss_never_yields_a_finite_worst_case(
    name: str, entry: str, position: int, radius: float
) -> None:
    ambiguity_set = BUILDERS[name](NOMINAL, COST, radius)
    loss = _with_entry(position, NON_FINITE_VALUES[entry])

    value = ambiguity_set.worst_case_expectation(loss)

    assert not bool(torch.isfinite(value))


@spec_parameters
@pytest.mark.parametrize("entry", sorted(NON_FINITE_VALUES))
@pytest.mark.parametrize("radius", [0.0, 0.3])
def test_a_non_finite_row_leaves_the_rest_of_the_batch_unchanged(
    spec: SetSpec, entry: str, radius: float
) -> None:
    ambiguity_set = spec.build(NOMINAL, COST, radius)
    loss = torch.stack((LOSS, _with_entry(3, NON_FINITE_VALUES[entry]), 2.0 * LOSS))

    value = ambiguity_set.worst_case_expectation(loss)

    assert torch.equal(value[0], ambiguity_set.worst_case_expectation(LOSS))
    assert not bool(torch.isfinite(value[1]))
    assert torch.equal(value[2], ambiguity_set.worst_case_expectation(2.0 * LOSS))


@spec_parameters
def test_a_nan_on_a_zero_mass_scenario_still_propagates(spec: SetSpec) -> None:
    # A set that cannot move mass onto the scenario still reports nan rather
    # than a number that silently ignores a corrupted loss.
    nominal, loss, cost = INSTANCES[3].tensors()
    loss[0] = math.nan

    value = spec.build(nominal, cost, 0.3).worst_case_expectation(loss)

    assert torch.isnan(value)
