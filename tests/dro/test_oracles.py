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
- `KLAmbiguitySet` and `ChiSquareAmbiguitySet` minimize a smooth dual with
  an injected first-order solver, so they match only while that dual has a
  finite minimizer. Writing $P^\star$ for the nominal mass on the
  highest-loss scenarios, the minimizer escapes to $\eta \to 0$ once
  `radius` reaches $-\log P^\star$ (KL) or $1/P^\star - 1$ (chi-square),
  and to $\eta \to \infty$ as `radius` approaches zero. Both ends are
  covered by strict `xfail` tests that record the current gap and will
  fail as soon as it is closed.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent
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

# Fractions of a set's saturation radius, the range over which its dual has
# a finite minimizer and a first-order solver can find it.
DUAL_RADIUS_FRACTIONS = (0.01, 0.1, 0.3, 0.6, 0.9, 0.99)


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

# Instances whose nominal puts positive mass on a unique highest-loss
# scenario at a moderate loss scale, so the KL and chi-square duals have a
# finite, well-conditioned minimizer below the saturation radius. The
# others are covered by the degenerate-regime tests below.
DUAL_INSTANCES = tuple(
    instance
    for instance in INSTANCES
    if instance.name in {"asymmetric", "duplicate_support"}
)

INSTANCE_IDS = [instance.name for instance in INSTANCES]
DUAL_INSTANCE_IDS = [instance.name for instance in DUAL_INSTANCES]

WIDE_SPREAD = next(instance for instance in INSTANCES if instance.name == "wide_spread")
SINGLE_POINT = next(
    instance for instance in INSTANCES if instance.name == "single_point"
)
ASYMMETRIC = next(instance for instance in INSTANCES if instance.name == "asymmetric")


def dual_solver(step_size: float = 0.5, max_iter: int = 20_000) -> GradientDescent:
    """Return the first-order dual solver the KL / chi-square checks use.

    Args:
        step_size: Gradient step the solver takes. The duals here are
            minimized over `log(eta)`, whose curvature scales with the
            spread of `loss`, so a loss spread far from order one needs a
            correspondingly smaller step.
        max_iter: Gradient-step budget. The default is generous enough that
            an agreement check measures the formulation rather than the
            solver; the degenerate-regime checks lower it, since there the
            minimizer sits at infinity and no budget reaches it.

    Returns:
        A configured `GradientDescent` instance.
    """
    return GradientDescent(step_size=step_size, max_iter=max_iter, tol=1e-14)


# Budget used wherever the dual minimizer is known to be unattainable, so
# the check costs a few milliseconds instead of exhausting a budget that
# cannot help.
DEGENERATE_MAX_ITER = 2_000


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


# --- Dual-solved sets: KL and chi-square -----------------------------------


@pytest.mark.parametrize("instance", DUAL_INSTANCES, ids=DUAL_INSTANCE_IDS)
@pytest.mark.parametrize("fraction", DUAL_RADIUS_FRACTIONS)
def test_kl_matches_the_exponential_tilt_closed_form(
    instance: Instance, fraction: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = fraction * kl_saturation_radius(instance)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radius, dual_solver=dual_solver())

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(instance.nominal_array, instance.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", DUAL_INSTANCES, ids=DUAL_INSTANCE_IDS)
@pytest.mark.parametrize("fraction", DUAL_RADIUS_FRACTIONS)
def test_chi_square_matches_the_active_set_closed_form(
    instance: Instance, fraction: float
) -> None:
    nominal, loss, _ = instance.tensors()
    radius = fraction * chi_square_saturation_radius(instance)
    ambiguity_set = ChiSquareAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver()
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = chi_square_worst_case(
        instance.nominal_array, instance.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("instance", DUAL_INSTANCES, ids=DUAL_INSTANCE_IDS)
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
    # sweep, so every element must still land on its own optimum.
    nominal, loss, _ = ASYMMETRIC.tensors()
    radii = torch.tensor(
        [
            fraction * kl_saturation_radius(ASYMMETRIC)
            for fraction in DUAL_RADIUS_FRACTIONS
        ],
        dtype=torch.float64,
    )
    ambiguity_set = KLAmbiguitySet(nominal, radius=radii, dual_solver=dual_solver())

    result = ambiguity_set.worst_case_expectation(loss)

    for index, radius in enumerate(radii.tolist()):
        reference = kl_worst_case(
            ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius
        )
        assert abs(float(result[index]) - reference) <= ORACLE_TOLERANCE


# --- Degenerate regimes of the dual-solved sets ----------------------------


@pytest.mark.parametrize("instance", DUAL_INSTANCES, ids=DUAL_INSTANCE_IDS)
def test_kl_above_its_saturation_radius_stays_an_upper_bound(
    instance: Instance,
) -> None:
    # Weak duality survives the regime below: any dual point the solver
    # stops at still bounds the worst case from above, so the reported
    # value is conservative rather than unsafe.
    nominal, loss, _ = instance.tensors()
    radius = 4.0 * kl_saturation_radius(instance)
    ambiguity_set = KLAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver(max_iter=DEGENERATE_MAX_ITER)
    )

    result = float(ambiguity_set.worst_case_expectation(loss))

    assert result >= kl_worst_case(instance.nominal_array, instance.loss_array, radius)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "past its saturation radius the KL dual minimizer runs to eta -> 0, so "
        "a gradient-norm stopping test leaves an uncontrolled gap above "
        "max(loss)"
    ),
)
def test_kl_above_its_saturation_radius_matches_the_closed_form() -> None:
    nominal, loss, _ = ASYMMETRIC.tensors()
    radius = 4.0 * kl_saturation_radius(ASYMMETRIC)
    ambiguity_set = KLAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver(max_iter=DEGENERATE_MAX_ITER)
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "as the radius vanishes the KL dual minimizer runs to eta -> infinity, "
        "so a gradient-norm stopping test leaves a gap far larger than the "
        "radius itself"
    ),
)
def test_kl_at_a_vanishing_radius_matches_the_closed_form() -> None:
    nominal, loss, _ = ASYMMETRIC.tensors()
    radius = 1e-10
    ambiguity_set = KLAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver(max_iter=DEGENERATE_MAX_ITER)
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "past its saturation radius the chi-square dual minimizer runs to "
        "eta -> 0, where the conjugate's quadratic branch overflows and the "
        "reported worst case degrades to nan"
    ),
)
def test_chi_square_above_its_saturation_radius_matches_the_closed_form() -> None:
    nominal, loss, _ = ASYMMETRIC.tensors()
    radius = 20.0 * chi_square_saturation_radius(ASYMMETRIC)
    ambiguity_set = ChiSquareAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver(max_iter=DEGENERATE_MAX_ITER)
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = chi_square_worst_case(
        ASYMMETRIC.nominal_array, ASYMMETRIC.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a single-point support saturates at radius zero, so every positive "
        "radius puts the KL dual in the eta -> 0 regime even though the only "
        "distribution in the set is the nominal one"
    ),
)
def test_kl_on_a_single_point_support_matches_the_closed_form() -> None:
    nominal, loss, _ = SINGLE_POINT.tensors()
    ambiguity_set = KLAmbiguitySet(
        nominal, radius=1.0, dual_solver=dual_solver(max_iter=DEGENERATE_MAX_ITER)
    )

    result = ambiguity_set.worst_case_expectation(loss)

    assert abs(float(result) - SINGLE_POINT.loss[0]) <= ORACLE_TOLERANCE


@pytest.mark.parametrize("fraction", DUAL_RADIUS_FRACTIONS)
def test_kl_with_a_wide_loss_spread_matches_the_closed_form(fraction: float) -> None:
    # The dual's curvature scales with the loss spread, so a step size that
    # suits an order-one loss diverges here; one scaled to the spread
    # recovers machine-precision agreement.
    nominal, loss, _ = WIDE_SPREAD.tensors()
    radius = fraction * kl_saturation_radius(WIDE_SPREAD)
    ambiguity_set = KLAmbiguitySet(
        nominal, radius=radius, dual_solver=dual_solver(step_size=0.05)
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = kl_worst_case(WIDE_SPREAD.nominal_array, WIDE_SPREAD.loss_array, radius)
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the chi-square conjugate grows quadratically in (loss - lam) / eta, "
        "so a wide loss spread overflows the dual objective before the "
        "injected first-order solver reaches its minimizer"
    ),
)
@pytest.mark.parametrize("step_size", [0.5, 0.05, 0.005])
def test_chi_square_with_a_wide_loss_spread_matches_the_closed_form(
    step_size: float,
) -> None:
    nominal, loss, _ = WIDE_SPREAD.tensors()
    radius = 0.3 * chi_square_saturation_radius(WIDE_SPREAD)
    ambiguity_set = ChiSquareAmbiguitySet(
        nominal,
        radius=radius,
        dual_solver=dual_solver(step_size=step_size, max_iter=DEGENERATE_MAX_ITER),
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = chi_square_worst_case(
        WIDE_SPREAD.nominal_array, WIDE_SPREAD.loss_array, radius
    )
    assert abs(float(result) - reference) <= ORACLE_TOLERANCE
