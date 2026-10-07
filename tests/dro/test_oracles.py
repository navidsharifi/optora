r"""Cross-checks of every ambiguity set against an independent reference.

Each test here compares `optora.dro`'s worst-case expectation against a
reference in `tests.dro.oracles` that shares no code with it: the
total-variation and Wasserstein sets are checked against exact primal
linear programs, the KL and chi-square sets against primal closed forms.
Unlike the fine-grid cross-checks in the per-set test modules, which
re-evaluate optora's own dual formula, these catch an error in the dual
derivation itself.

The sets fall into two groups:

- `TotalVariationAmbiguitySet` and `WassersteinAmbiguitySet` evaluate exact
  combinatorial solutions (a sorted cumulative sweep, a bisection on a
  monotone piecewise-constant derivative), and match their oracles to
  machine precision at every radius from zero to far past saturation.
- `KLAmbiguitySet` and `ChiSquareAmbiguitySet` pin a monotone tilt of the
  nominal by bisection. Writing $P^\star$ for the nominal mass on the
  highest-loss scenarios, the radius constraint is tight below the
  saturation radius $-\log P^\star$ (KL) or $1/P^\star - 1$ (chi-square);
  from there the whole tilt path is feasible and the worst case is exactly
  `max(loss)`. They match their oracles to machine precision across that
  whole range, on a loss of any scale, and down to a vanishing radius.

Because these two sets are now solved in the primal, the oracles share the
*derivation* of the maximizer with them, where before they shared nothing.
They remain independent implementations — SciPy `brentq` and an active-set
scan over NumPy arrays, against a bracketed torch bisection — and the
per-set modules add fine-grid cross-checks against the convex duals, which
the primal path never touches.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from tests.dro.oracles import (
    chi_square_worst_case,
    kl_worst_case,
    total_variation_worst_case,
    wasserstein_worst_case,
)

ORACLE_TOLERANCE = 1e-9

# Radii spanning the whole range an exactly solved set must handle: zero,
# far below any structural threshold, around the point where the worst case
# saturates at max(loss), and far beyond it.
EXACT_RADII = (0.0, 1e-8, 1e-3, 0.1, 0.5, 1.0, 2.0, 1e3)

# Fractions of a set's saturation radius, the range over which the radius
# constraint is tight and the tilt is pinned strictly inside its bracket.
# The leading entries are the vanishing-radius regime of issue #49, where
# the equivalent dual's minimizer runs off to infinity.
TILT_RADIUS_FRACTIONS = (1e-10, 1e-6, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99)

# Multiples of the saturation radius from exactly at it to far beyond it,
# where the worst case is the maximum loss.
SATURATION_MULTIPLES = (1.0, 1.5, 4.0, 20.0, 1e3)


@dataclass(frozen=True)
class Instance:
    """A nominal distribution, a loss vector, and a ground-cost geometry.

    Attributes:
        name: Identifier used as the pytest parameter id.
        nominal: Reference distribution over the support.
        loss: Per-scenario losses, one per support point.
        position: Coordinate of each support point on the real line, from
            which the Wasserstein ground cost is built as the pairwise
            absolute difference.
    """

    name: str
    nominal: tuple[float, ...]
    loss: tuple[float, ...]
    position: tuple[float, ...]

    @property
    def nominal_array(self) -> np.ndarray:
        """Return the nominal distribution as a float64 NumPy array."""
        return np.array(self.nominal, dtype=np.float64)

    @property
    def loss_array(self) -> np.ndarray:
        """Return the loss vector as a float64 NumPy array."""
        return np.array(self.loss, dtype=np.float64)

    @property
    def cost_array(self) -> np.ndarray:
        """Return the pairwise absolute-difference ground cost matrix."""
        position = np.array(self.position, dtype=np.float64)
        return np.abs(position[:, None] - position[None, :])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(nominal, loss, cost)` as float64 tensors."""
        return (
            torch.tensor(self.nominal_array, dtype=torch.float64),
            torch.tensor(self.loss_array, dtype=torch.float64),
            torch.tensor(self.cost_array, dtype=torch.float64),
        )


INSTANCES = (
    Instance(
        name="asymmetric",
        nominal=(0.1, 0.2, 0.3, 0.4),
        loss=(0.0, 1.0, 2.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="tied_maximum",
        nominal=(0.1, 0.2, 0.3, 0.4),
        loss=(0.0, 1.0, 5.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="duplicate_support",
        nominal=(0.25, 0.25, 0.25, 0.25),
        loss=(1.0, 1.0, 3.0, 4.0),
        position=(0.0, 0.0, 1.0, 2.0),
    ),
    Instance(
        name="zero_weight_maximum",
        nominal=(0.0, 0.3, 0.3, 0.4),
        loss=(9.0, 1.0, 2.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="wide_spread",
        nominal=(0.5, 0.5),
        loss=(0.0, 100.0),
        position=(0.0, 1.0),
    ),
    Instance(
        name="single_point",
        nominal=(1.0,),
        loss=(2.5,),
        position=(0.0,),
    ),
)

# Instances with positive nominal mass strictly between zero and one on the
# highest-loss scenarios, so the radius constraint is tight below the
# saturation radius. A highest-loss scenario with zero nominal mass never
# saturates, and one carrying all the mass saturates at radius zero; both
# are covered by their own tests below.
TILT_INSTANCES = tuple(
    instance
    for instance in INSTANCES
    if instance.name
    in {"asymmetric", "tied_maximum", "duplicate_support", "wide_spread"}
)

# Instances for which the worst case saturates at some finite radius.
SATURATING_INSTANCES = TILT_INSTANCES + tuple(
    instance for instance in INSTANCES if instance.name == "single_point"
)

INSTANCE_IDS = [instance.name for instance in INSTANCES]
TILT_INSTANCE_IDS = [instance.name for instance in TILT_INSTANCES]
SATURATING_INSTANCE_IDS = [instance.name for instance in SATURATING_INSTANCES]

ASYMMETRIC = next(instance for instance in INSTANCES if instance.name == "asymmetric")


def saturation_mass(instance: Instance) -> float:
    """Return the nominal mass carried by the instance's highest-loss scenarios."""
    nominal, loss = instance.nominal_array, instance.loss_array
    return float(nominal[loss >= loss.max()].sum())


def kl_saturation_radius(instance: Instance) -> float:
    r"""Return the KL radius $-\log P^\star$ at which the worst case saturates."""
    return float(-np.log(saturation_mass(instance)))


def chi_square_saturation_radius(instance: Instance) -> float:
    r"""Return the chi-square radius $1/P^\star - 1$ at which the worst case saturates."""  # noqa: E501
    return 1.0 / saturation_mass(instance) - 1.0


# --- Exactly solved sets: total variation and Wasserstein ------------------


@pytest.mark.parametrize("instance", INSTANCES, ids=INSTANCE_IDS)
@pytest.mark.parametrize("radius", EXACT_RADII)
def test_total_variation_matches_the_primal_linear_program(
    instance: Instance, radius: float
) -> None:
    nominal, loss, _ = instance.tensors()
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = total_variation_worst_case(
        instance.nominal_array, instance.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", INSTANCES, ids=INSTANCE_IDS)
@pytest.mark.parametrize("radius", EXACT_RADII)
def test_wasserstein_matches_the_primal_transport_linear_program(
    instance: Instance, radius: float
) -> None:
    nominal, loss, cost = instance.tensors()
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = wasserstein_worst_case(
        instance.nominal_array, instance.cost_array, instance.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", INSTANCES, ids=INSTANCE_IDS)
def test_batched_radius_sweep_matches_the_oracles_elementwise(
    instance: Instance,
) -> None:
    # A tensor radius solves the whole sweep in one call; each entry must
    # reproduce the value the same radius gives on its own. A tensor radius
    # also skips the zero-radius shortcut, so the leading entry exercises
    # the general path at radius zero.
    nominal, loss, cost = instance.tensors()
    radii = torch.tensor(EXACT_RADII, dtype=torch.float64)

    total_variation = TotalVariationAmbiguitySet(
        nominal, radius=radii
    ).worst_case_expectation(loss)
    wasserstein = WassersteinAmbiguitySet(
        nominal, cost=cost, radius=radii
    ).worst_case_expectation(loss)

    for index, radius in enumerate(EXACT_RADII):
        assert (
            abs(
                float(total_variation[index])
                - total_variation_worst_case(
                    instance.nominal_array, instance.loss_array, radius
                )
            )
            <= ORACLE_TOLERANCE
        )
        assert (
            abs(
                float(wasserstein[index])
                - wasserstein_worst_case(
                    instance.nominal_array,
                    instance.cost_array,
                    instance.loss_array,
                    radius,
                )
            )
            <= ORACLE_TOLERANCE
        )


# --- Tilt-solved sets: KL and chi-square -----------------------------------


@pytest.mark.parametrize("instance", TILT_INSTANCES, ids=TILT_INSTANCE_IDS)
@pytest.mark.parametrize("fraction", TILT_RADIUS_FRACTIONS)
def test_kl_matches_the_exponential_tilt_closed_form(
    instance: Instance, fraction: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = fraction * kl_saturation_radius(instance)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(instance.nominal_array, instance.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", TILT_INSTANCES, ids=TILT_INSTANCE_IDS)
@pytest.mark.parametrize("fraction", TILT_RADIUS_FRACTIONS)
def test_chi_square_matches_the_active_set_closed_form(
    instance: Instance, fraction: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = fraction * chi_square_saturation_radius(instance)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = chi_square_worst_case(
        instance.nominal_array, instance.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", SATURATING_INSTANCES, ids=SATURATING_INSTANCE_IDS)
@pytest.mark.parametrize("multiple", SATURATION_MULTIPLES)
def test_kl_at_and_beyond_its_saturation_radius_is_the_maximum_loss(
    instance: Instance, multiple: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = multiple * kl_saturation_radius(instance)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(instance.nominal_array, instance.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", SATURATING_INSTANCES, ids=SATURATING_INSTANCE_IDS)
@pytest.mark.parametrize("multiple", SATURATION_MULTIPLES)
def test_chi_square_at_and_beyond_its_saturation_radius_is_the_maximum_loss(
    instance: Instance, multiple: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = multiple * chi_square_saturation_radius(instance)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    assert float(result) == float(instance.loss_array.max())


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_tilt_solved_sets_are_exact_on_a_loss_of_any_scale_and_offset(
    scale: float, offset: float
) -> None:
    # One step size serves every scale because the dual is solved on the
    # standardized loss; before that, a wide spread overflowed the
    # chi-square conjugate and returned nan (issue #48).
    nominal, _, _ = ASYMMETRIC.tensors()
    loss_array = offset + scale * ASYMMETRIC.loss_array
    loss = torch.tensor(loss_array, dtype=torch.float64)
    kl_radius = 0.5 * kl_saturation_radius(ASYMMETRIC)
    chi_square_radius = 0.5 * chi_square_saturation_radius(ASYMMETRIC)

    kl = KLAmbiguitySet(nominal, radius=kl_radius).worst_case_expectation(loss)
    chi_square = ChiSquareAmbiguitySet(
        nominal, radius=chi_square_radius
    ).worst_case_expectation(loss)

    tolerance = ORACLE_TOLERANCE * max(1.0, abs(offset) + scale)
    assert (
        abs(float(kl) - kl_worst_case(ASYMMETRIC.nominal_array, loss_array, kl_radius))
        <= tolerance
    )
    assert (
        abs(
            float(chi_square)
            - chi_square_worst_case(
                ASYMMETRIC.nominal_array, loss_array, chi_square_radius
            )
        )
        <= tolerance
    )


@pytest.mark.parametrize("instance", TILT_INSTANCES, ids=TILT_INSTANCE_IDS)
def test_zero_radius_matches_the_oracles_exactly(instance: Instance) -> None:
    nominal, loss, _ = instance.tensors()
    nominal_array, loss_array = instance.nominal_array, instance.loss_array

    kl = KLAmbiguitySet(nominal, radius=0.0).worst_case_expectation(loss)
    chi_square = ChiSquareAmbiguitySet(nominal, radius=0.0).worst_case_expectation(loss)

    assert abs(float(kl) - kl_worst_case(nominal_array, loss_array, 0.0)) <= 1e-15
    assert (
        abs(float(chi_square) - chi_square_worst_case(nominal_array, loss_array, 0.0))
        <= 1e-15
    )


def test_batched_kl_radius_sweep_matches_the_closed_form_elementwise() -> None:
    # The batched dual stops on the joint gradient norm over the whole
    # sweep, so every element must still land on its own optimum. The sweep
    # straddles the saturation radius, so it also mixes solved and
    # saturated elements in one batch.
    nominal, loss, _ = ASYMMETRIC.tensors()
    radii_list = [
        multiple * kl_saturation_radius(ASYMMETRIC)
        for multiple in (*TILT_RADIUS_FRACTIONS, 1.0, 4.0)
    ]
    radii = torch.tensor(radii_list, dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radii)

    result = ambiguity_set.worst_case_expectation(loss)

    for index, radius in enumerate(radii_list):
        reference = kl_worst_case(
            ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius
        )
        assert abs(float(result[index]) - reference) <= ORACLE_TOLERANCE


def test_batched_chi_square_radius_sweep_matches_the_closed_form_elementwise() -> None:
    nominal, loss, _ = ASYMMETRIC.tensors()
    radii_list = [
        multiple * chi_square_saturation_radius(ASYMMETRIC)
        for multiple in (*TILT_RADIUS_FRACTIONS, 1.0, 20.0)
    ]
    radii = torch.tensor(radii_list, dtype=torch.float64)
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radii)

    result = ambiguity_set.worst_case_expectation(loss)

    for index, radius in enumerate(radii_list):
        reference = chi_square_worst_case(
            ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius
        )
        assert abs(float(result[index]) - reference) <= ORACLE_TOLERANCE


# --- A vanishing radius (issue #49) ------------------------------------------


@pytest.mark.parametrize("instance", TILT_INSTANCES, ids=TILT_INSTANCE_IDS)
@pytest.mark.parametrize("radius", [1e-10, 1e-8, 1e-6])
def test_a_vanishing_radius_matches_the_oracles(
    instance: Instance, radius: float
) -> None:
    # A radius this small leaves a worst case within O(sqrt(radius)) of the
    # nominal expectation, so the tolerance is checked against the excess
    # rather than against the value: an absolute check would pass on a
    # solver that simply returned the nominal expectation.
    #
    # 1e-10 is the floor of what the oracles can referee, not of what the
    # sets can do. Below it the oracles lose accuracy first: `brentq` roots
    # a divergence of order `radius` that NumPy evaluates by direct
    # summation, so its own noise swamps the answer. The regime below this
    # is pinned instead against the exact small-radius expansion, in
    # `tests/dro/test_kl_dro.py` and `tests/dro/test_phi_dro.py`.
    nominal, loss, _ = instance.tensors()
    nominal_array, loss_array = instance.nominal_array, instance.loss_array
    nominal_expectation = float(np.sum(nominal_array * loss_array))

    kl = KLAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)
    chi_square = ChiSquareAmbiguitySet(nominal, radius=radius).worst_case_expectation(
        loss
    )

    kl_reference = kl_worst_case(nominal_array, loss_array, radius)
    chi_square_reference = chi_square_worst_case(nominal_array, loss_array, radius)
    kl_excess = kl_reference - nominal_expectation
    chi_square_excess = chi_square_reference - nominal_expectation
    assert abs(float(kl) - kl_reference) <= 1e-6 * kl_excess
    assert abs(float(chi_square) - chi_square_reference) <= 1e-6 * chi_square_excess
