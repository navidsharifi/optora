r"""Tests for the tilt bisection shared by the KL and chi-square ambiguity sets.

`optora.core.dro_base.TiltedAmbiguitySet` locates the worst-case distribution
by bisecting a monotone tilt path on the loss standardized to unit spread.
These tests pin the properties that make that safe, using the real
formulations: exact affine equivariance, no `nan` on any loss scale or
support size (issues #48 and #51), a value that never exceeds `max(loss)`,
accuracy that survives a vanishing radius (issue #49), saturated elements
that cannot disturb the rest of a batch, gradients that stay valid
distributions inside the ambiguity set, and no mass ever placed on a
scenario the nominal gives zero weight.
"""

import math

import pytest
import torch

from optora.core.dro_base import TiltedAmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet
from tests.dro.oracles import chi_square_worst_case, kl_worst_case
from tests.dro.tolerances import feasibility_bound

NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
LOSS = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)

# Saturation radii of NOMINAL: the top-loss scenario carries mass 0.4.
KL_SATURATION = -math.log(0.4)
CHI_SQUARE_SATURATION = 1.0 / 0.4 - 1.0


def _kl(radius: float | torch.Tensor) -> KLAmbiguitySet:
    return KLAmbiguitySet(NOMINAL, radius=radius)


def _chi_square(radius: float | torch.Tensor) -> ChiSquareAmbiguitySet:
    return ChiSquareAmbiguitySet(NOMINAL, radius=radius)


# --- Issue #48: no nan on any loss scale -------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1.0, 100.0, 1e6])
def test_chi_square_is_finite_and_exact_on_a_wide_loss_spread(scale: float) -> None:
    radius = 0.3 * CHI_SQUARE_SATURATION

    result = _chi_square(radius).worst_case_expectation(scale * LOSS)

    reference = chi_square_worst_case(NOMINAL.numpy(), scale * LOSS.numpy(), radius)
    assert torch.isfinite(result)
    assert abs(float(result) - reference) <= 1e-9 * max(1.0, scale)


@pytest.mark.parametrize("multiple", [1.0, 4.0, 20.0, 1e3, 1e6])
def test_chi_square_is_finite_at_and_beyond_its_saturation_radius(
    multiple: float,
) -> None:
    result = _chi_square(multiple * CHI_SQUARE_SATURATION).worst_case_expectation(LOSS)

    assert float(result) == float(LOSS.max())


# --- Issue #51: no nan and no overshoot on a large support -------------------


@pytest.mark.parametrize("support_size", [2, 10, 50, 100, 1000, 10000])
@pytest.mark.parametrize("fraction", [1e-6, 0.1, 0.5, 0.9, 1.0, 2.0])
def test_a_large_support_stays_finite_and_below_the_maximum_loss(
    support_size: int, fraction: float
) -> None:
    # The old first-order dual solve overshot on a support this large: the
    # chi-square set returned values above max(loss) from n = 10 and nan
    # from n = 50. Bisection keeps the feasible endpoint at every step, so
    # both bounds hold by construction rather than by tuning.
    nominal = torch.full((support_size,), 1.0 / support_size, dtype=torch.float64)
    loss = torch.linspace(0.0, 1.0, support_size, dtype=torch.float64)
    nominal_expectation = torch.sum(nominal * loss)

    kl = KLAmbiguitySet(
        nominal, radius=fraction * math.log(support_size)
    ).worst_case_expectation(loss)
    chi_square = ChiSquareAmbiguitySet(
        nominal, radius=fraction * (support_size - 1.0)
    ).worst_case_expectation(loss)

    for result in (kl, chi_square):
        assert torch.isfinite(result)
        assert result <= loss.max()
        assert result >= nominal_expectation
    maximum = float(loss.max())
    for result in (kl, chi_square):
        if fraction > 1.0:
            assert float(result) == maximum
        elif fraction == 1.0:
            # Exactly at saturation the feasibility test compares two values
            # equal in exact arithmetic; reduction order may cost a few ulps.
            assert float(result) >= maximum - 4 * math.ulp(maximum)


# --- Issue #49: accuracy at a vanishing radius -------------------------------


@pytest.mark.parametrize("radius", [1e-16, 1e-14, 1e-12, 1e-10, 1e-8])
def test_a_vanishing_radius_tracks_the_square_root_expansion(radius: float) -> None:
    # Both worst cases expand as `E_p[loss] + sqrt(k * radius * Var_p(loss))`
    # for small radius, with `k = 2` for KL and `k = 1` for chi-square. The
    # excess over the nominal expectation is what must stay accurate: the
    # value itself is within O(sqrt(radius)) of the nominal expectation, so
    # an absolute check would pass on a solver that never moved at all.
    mean = torch.sum(NOMINAL * LOSS)
    variance = torch.sum(NOMINAL * (LOSS - mean) ** 2)

    kl_excess = float(_kl(radius).worst_case_expectation(LOSS) - mean)
    chi_square_excess = float(_chi_square(radius).worst_case_expectation(LOSS) - mean)

    assert kl_excess == pytest.approx(float(torch.sqrt(2.0 * radius * variance)), 1e-3)
    assert chi_square_excess == pytest.approx(
        float(torch.sqrt(radius * variance)), 1e-6
    )


@pytest.mark.parametrize("radius", [1e-16, 1e-12, 1e-8])
def test_a_vanishing_radius_is_strictly_above_the_nominal_expectation(
    radius: float,
) -> None:
    # The bisection must resolve a tilt this small rather than collapsing to
    # the zero-radius answer, which is what a gradient-norm stopping test on
    # the dual did once its minimizer ran off to infinity.
    mean = torch.sum(NOMINAL * LOSS)

    assert _kl(radius).worst_case_expectation(LOSS) > mean
    assert _chi_square(radius).worst_case_expectation(LOSS) > mean


# --- Exact equivariance of the standardization -------------------------------


@pytest.mark.parametrize("scale", [1e-3, 0.5, 7.0, 1e4])
@pytest.mark.parametrize("offset", [-1e3, 0.0, 2.5, 1e5])
def test_worst_case_is_equivariant_under_a_positive_affine_loss_transform(
    scale: float, offset: float
) -> None:
    # sup_q E_q[a * loss + b] = a * sup_q E_q[loss] + b for a > 0, because the
    # set of candidate distributions does not depend on the loss.
    radii = torch.tensor(
        [0.2 * KL_SATURATION, 0.7 * KL_SATURATION, 3.0 * KL_SATURATION],
        dtype=torch.float64,
    )
    for make in (_kl, _chi_square):
        base = make(radii).worst_case_expectation(LOSS)
        transformed = make(radii).worst_case_expectation(scale * LOSS + offset)

        tolerance = 1e-9 * max(1.0, scale, abs(offset))
        assert torch.allclose(transformed, scale * base + offset, atol=tolerance)


# --- A saturated element cannot disturb the rest of its batch ----------------


def test_a_saturated_element_does_not_disturb_the_batch() -> None:
    solved_radius = 0.5 * CHI_SQUARE_SATURATION
    radii = torch.tensor(
        [solved_radius, 20.0 * CHI_SQUARE_SATURATION], dtype=torch.float64
    )
    ambiguity_set = _chi_square(radii)

    result = ambiguity_set.worst_case_expectation(LOSS)

    assert torch.isfinite(result).all()
    alone = _chi_square(solved_radius).worst_case_expectation(LOSS)
    assert float(result[0]) == float(alone)
    assert float(result[1]) == float(LOSS.max())
    again = ambiguity_set.worst_case_expectation(LOSS)
    assert torch.equal(again, result)


def test_batch_of_losses_mixes_saturated_and_solved_elements() -> None:
    # The same radius saturates for a loss whose maximum is unique and heavy
    # but not for one that spreads the maximum over several scenarios.
    heavy = torch.tensor([0.0, 0.0, 0.0, 9.0], dtype=torch.float64)
    spread = torch.tensor([0.0, 9.0, 0.0, 9.0], dtype=torch.float64)
    radius = 1.2
    losses = torch.stack([heavy, spread])

    batched = _kl(radius).worst_case_expectation(losses)

    for index, loss in enumerate(losses):
        expected = kl_worst_case(NOMINAL.numpy(), loss.numpy(), radius)
        assert abs(float(batched[index]) - expected) <= 1e-9


# --- Gradients ---------------------------------------------------------------


@pytest.mark.parametrize("make", [_kl, _chi_square], ids=["kl", "chi_square"])
@pytest.mark.parametrize("multiple", [1e-8, 0.3, 0.9, 1.0, 4.0])
def test_gradient_is_a_distribution_inside_the_ambiguity_set(
    make: type[KLAmbiguitySet] | type[ChiSquareAmbiguitySet], multiple: float
) -> None:
    # By Danskin's theorem the gradient of the worst case with respect to
    # the loss is the worst-case distribution itself, tilted or saturated
    # alike, so it must be a probability vector the set contains.
    saturation = KL_SATURATION if make is _kl else CHI_SQUARE_SATURATION
    radius = multiple * saturation
    ambiguity_set = make(radius)  # type: ignore[operator]
    loss = LOSS.clone().requires_grad_(True)

    (gradient,) = torch.autograd.grad(ambiguity_set.worst_case_expectation(loss), loss)

    assert torch.isfinite(gradient).all()
    assert bool((gradient >= 0.0).all())
    assert abs(float(gradient.sum()) - 1.0) <= 1e-12
    divergence = float(ambiguity_set.divergence(gradient, NOMINAL))
    assert divergence <= feasibility_bound(radius, NOMINAL.numel(), NOMINAL.dtype)


def test_saturated_gradient_is_the_nominal_restricted_to_the_top_scenarios() -> None:
    # Tied maxima: an even split across the ties would leave the set, while
    # the restricted nominal is the worst-case distribution at the boundary.
    loss = torch.tensor([0.0, 1.0, 5.0, 5.0], dtype=torch.float64, requires_grad=True)
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=2.0)

    value = ambiguity_set.worst_case_expectation(loss)
    (gradient,) = torch.autograd.grad(value, loss)

    expected = torch.tensor([0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0], dtype=torch.float64)
    assert float(value.detach()) == 5.0
    assert torch.allclose(gradient, expected, atol=1e-12)


def test_chi_square_gradient_matches_finite_differences_on_a_wide_spread() -> None:
    radius = 0.4 * CHI_SQUARE_SATURATION
    loss = 100.0 * LOSS
    ambiguity_set = _chi_square(radius)

    leaf = loss.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(ambiguity_set.worst_case_expectation(leaf), leaf)

    step = 1e-3
    numerical = torch.zeros_like(loss)
    for index in range(loss.numel()):
        bump = torch.zeros_like(loss)
        bump[index] = step
        numerical[index] = (
            _chi_square(radius).worst_case_expectation(loss + bump)
            - _chi_square(radius).worst_case_expectation(loss - bump)
        ) / (2 * step)
    assert torch.allclose(gradient, numerical, atol=1e-6)


# --- Degenerate losses -------------------------------------------------------


@pytest.mark.parametrize("make", [_kl, _chi_square], ids=["kl", "chi_square"])
def test_a_constant_loss_returns_the_constant(
    make: type[KLAmbiguitySet] | type[ChiSquareAmbiguitySet],
) -> None:
    # With no spread every distribution has the same expectation, and the
    # set saturates at radius zero. The gradient is the nominal itself.
    loss = torch.full((4,), 3.5, dtype=torch.float64, requires_grad=True)
    ambiguity_set = make(0.7)  # type: ignore[operator]

    value = ambiguity_set.worst_case_expectation(loss)
    (gradient,) = torch.autograd.grad(value, loss)

    assert float(value.detach()) == 3.5
    assert torch.allclose(gradient, NOMINAL, atol=1e-15)


@pytest.mark.parametrize(
    "ambiguity_type", [KLAmbiguitySet, ChiSquareAmbiguitySet], ids=["kl", "chi_square"]
)
@pytest.mark.parametrize("radius", [1e-10, 0.3, 30.0, 1e3])
def test_a_zero_weight_maximum_is_unreachable_at_any_radius(
    ambiguity_type: type[KLAmbiguitySet] | type[ChiSquareAmbiguitySet],
    radius: float,
) -> None:
    # Both divergences are infinite off the nominal's support, so the loss of
    # 9 on the zero-mass scenario must never enter the worst case. KL used to
    # clamp that mass to 1e-12 and leak onto the scenario once the tilt was
    # steep enough, reporting nearly 9 at radius 1e3; the exact answer there
    # is the largest loss on the support, 5.
    nominal = torch.tensor([0.0, 0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([9.0, 1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True)
    ambiguity_set = ambiguity_type(nominal, radius=radius)

    value = ambiguity_set.worst_case_expectation(loss)
    (gradient,) = torch.autograd.grad(value, loss)

    reference_fn = (
        kl_worst_case if ambiguity_type is KLAmbiguitySet else (chi_square_worst_case)
    )
    reference = reference_fn(nominal.numpy(), loss.detach().numpy(), radius)
    assert abs(float(value.detach()) - reference) <= 1e-9
    assert float(gradient[0]) == 0.0


@pytest.mark.parametrize("make", [_kl, _chi_square], ids=["kl", "chi_square"])
def test_float32_wide_spread_is_finite(
    make: type[KLAmbiguitySet] | type[ChiSquareAmbiguitySet],
) -> None:
    nominal = NOMINAL.to(torch.float32)
    loss = (1e4 * LOSS).to(torch.float32)
    saturation = KL_SATURATION if make is _kl else CHI_SQUARE_SATURATION
    radius = 0.3 * saturation
    ambiguity_set = type(make(0.1))(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    reference_fn = kl_worst_case if make is _kl else chi_square_worst_case
    reference = reference_fn(NOMINAL.numpy(), 1e4 * LOSS.numpy(), radius)
    assert torch.isfinite(result)
    assert abs(float(result) - reference) <= 1e-3 * abs(reference)


# --- Plan 8d: KL divergence across its whole range ---------------------------

# Float32 accuracy budgets, in units of float32 epsilon times the larger of the
# worst case and the loss spread (documented in `docs/numerics.md`). A worst
# case concentrated on a mass far below epsilon needs a tilt rate near the top
# of its bracket, where the bisection resolves the rate only to a relative
# `eps / (1 - t)`, so that regime gets the looser budget.
FLOAT32_TOLERANCE_EPS = 128.0
FLOAT32_TAIL_TOLERANCE_EPS = 2048.0


def _float32_error(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float
) -> tuple[torch.Tensor, float]:
    """Return a float32 KL worst case and its error in budget units."""
    value = KLAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)
    reference = KLAmbiguitySet(nominal.double(), radius=radius)
    expected = float(reference.worst_case_expectation(loss.double()))
    unit = torch.finfo(torch.float32).eps * max(
        abs(expected), float(torch.amax(loss) - torch.amin(loss))
    )
    return value, abs(float(value) - expected) / unit


@pytest.mark.parametrize("fraction", [0.3, 0.5, 0.9])
def test_kl_divergence_far_above_one_is_not_lost_to_cancellation(
    fraction: float,
) -> None:
    # The centered form reads exp(-D) as one plus a sum near -1, which rounds
    # to exactly -1 once D exceeds about 36 in float64; the saturation radius
    # here is 230.
    mass = 1e-100
    nominal = torch.tensor([1.0 - mass, mass], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    radius = -fraction * math.log(mass)

    value = KLAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)

    reference = kl_worst_case(nominal.numpy(), loss.numpy(), radius)
    assert abs(float(value) - reference) <= 1e-9


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.99])
def test_float32_kl_near_saturation_on_a_large_support(fraction: float) -> None:
    # A uniform nominal on 10^5 points saturates at log(10^5) = 11.5, past the
    # float32 cancellation threshold of the centered form.
    generator = torch.Generator().manual_seed(0)
    size = 100_000
    nominal = torch.full((size,), 1.0 / size)
    loss = torch.randn(size, generator=generator)

    value, error = _float32_error(nominal, loss, fraction * math.log(size))

    assert torch.isfinite(value)
    assert error <= FLOAT32_TOLERANCE_EPS


@pytest.mark.parametrize("mass", [1e-20, 1e-38, 1e-44], ids=["1e-20", "1e-38", "1e-44"])
@pytest.mark.parametrize("fraction", [1e-3, 0.5, 0.999])
def test_float32_kl_with_a_tiny_top_loss_mass(mass: float, fraction: float) -> None:
    # The highest loss sits on a mass down to a float32 subnormal, so the
    # centered exponent exceeds the float32 range of expm1 although every
    # product nominal * exp(exponent) is at most one.
    nominal = torch.tensor([1.0, mass], dtype=torch.float32)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float32)
    saturation = -math.log(float(nominal[1]))

    value, error = _float32_error(nominal, loss, fraction * saturation)

    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0
    assert error <= FLOAT32_TAIL_TOLERANCE_EPS


@pytest.mark.parametrize("offset", [-1e20, 1e6, 1e30])
@pytest.mark.parametrize("radius", [1e-8, 0.1, 3.0])
def test_float32_kl_on_a_large_offset(offset: float, radius: float) -> None:
    generator = torch.Generator().manual_seed(1)
    nominal = torch.softmax(torch.randn(64, generator=generator), dim=0)
    loss = offset + abs(offset) * 1e-6 * torch.randn(64, generator=generator)

    value, error = _float32_error(nominal, loss, radius)

    assert torch.isfinite(value)
    assert error <= FLOAT32_TOLERANCE_EPS


def test_tilted_ambiguity_set_is_abstract_over_its_formulation() -> None:
    with pytest.raises(TypeError):
        TiltedAmbiguitySet(  # type: ignore[abstract]
            NOMINAL,
            divergence=KLAmbiguitySet(NOMINAL, radius=0.1).divergence,
            radius=0.1,
        )


class _UndifferentiatedTilt(TiltedAmbiguitySet):
    """A linear tilt without parameter derivatives, as a third-party subclass."""

    def _tilted_distribution(
        self, gap: torch.Tensor, tilt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nominal = self._constant_nominal
        weight = torch.clamp(1.0 - self._tilt_rate(tilt) * gap, min=0.0)
        distribution = nominal * weight / torch.sum(nominal * weight, -1, keepdim=True)
        ratio = distribution / nominal
        return distribution, torch.sum(nominal * (ratio - 1.0) ** 2, dim=-1)


def test_tilted_subclass_without_parameter_derivatives_fails_loudly() -> None:
    nominal = NOMINAL.clone()
    ambiguity_set = _UndifferentiatedTilt(
        nominal,
        divergence=ChiSquareAmbiguitySet(NOMINAL, radius=0.1).divergence,
        radius=0.1,
    )
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)

    expected = ChiSquareAmbiguitySet(NOMINAL, radius=0.1).worst_case_expectation(loss)
    torch.testing.assert_close(ambiguity_set.worst_case_expectation(loss), expected)
    nominal.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="_UndifferentiatedTilt"):
        ambiguity_set.worst_case_expectation(loss)
