r"""Tests for the loss standardization and saturation shortcut of dual-solved sets.

`optora.core.dro_base.DualAmbiguitySet` solves the KL and chi-square duals on
the loss standardized to unit spread and returns `max(loss)` once the radius
reaches the saturation radius. These tests pin the properties that make that
safe, using the real formulations: exact affine equivariance, no `nan` on any
loss scale (issue #48), saturated elements that cannot poison the rest of a
batch, and gradients that stay valid distributions inside the ambiguity set.
"""

import math

import pytest
import torch

from optora.core.convergence import ConvergenceStatus
from optora.core.dro_base import DualAmbiguitySet
from optora.core.solver_base import (
    MinimizationProblem,
    MinimizationResult,
    Solver,
)
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.oracles import chi_square_worst_case, kl_worst_case

NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
LOSS = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)

# Saturation radii of NOMINAL: the top-loss scenario carries mass 0.4.
KL_SATURATION = -math.log(0.4)
CHI_SQUARE_SATURATION = 1.0 / 0.4 - 1.0


def _solver() -> GradientDescent:
    return GradientDescent(step_size=0.2, max_iter=20_000, tol=1e-11)


def _kl(radius: float | torch.Tensor) -> KLAmbiguitySet:
    return KLAmbiguitySet(NOMINAL, radius=radius, dual_solver=_solver())


def _chi_square(radius: float | torch.Tensor) -> ChiSquareAmbiguitySet:
    return ChiSquareAmbiguitySet(NOMINAL, radius=radius, dual_solver=_solver())


class _RecordingSolver(Solver[MinimizationProblem, MinimizationResult]):
    """Fake dual solver recording the point it is asked to start from."""

    def __init__(self) -> None:
        self.initial_point: torch.Tensor | None = None

    def solve(self, problem: MinimizationProblem) -> MinimizationResult:
        self.initial_point = problem.initial_point.clone()
        return MinimizationResult(
            point=problem.initial_point,
            value=problem.objective(problem.initial_point),
            status=ConvergenceStatus(torch.tensor(True), torch.tensor(0)),
        )


# --- Issue #48: no nan on any loss scale or radius ---------------------------


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


# --- A saturated element cannot poison the rest of its batch -----------------


def test_a_saturated_element_does_not_poison_the_batch() -> None:
    solved_radius = 0.5 * CHI_SQUARE_SATURATION
    radii = torch.tensor(
        [solved_radius, 20.0 * CHI_SQUARE_SATURATION], dtype=torch.float64
    )
    ambiguity_set = _chi_square(radii)

    result = ambiguity_set.worst_case_expectation(LOSS)

    assert torch.isfinite(result).all()
    alone = _chi_square(solved_radius).worst_case_expectation(LOSS)
    assert abs(float(result[0]) - float(alone)) <= 1e-12
    assert float(result[1]) == float(LOSS.max())
    # The cached warm start must stay usable for the next call.
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
@pytest.mark.parametrize("multiple", [0.3, 0.9, 1.0, 4.0])
def test_gradient_is_a_distribution_inside_the_ambiguity_set(
    make: type[KLAmbiguitySet] | type[ChiSquareAmbiguitySet], multiple: float
) -> None:
    # The gradient of the worst case with respect to the loss is the
    # worst-case distribution, solved or saturated alike.
    saturation = KL_SATURATION if make is _kl else CHI_SQUARE_SATURATION
    ambiguity_set = make(multiple * saturation)  # type: ignore[operator]
    loss = LOSS.clone().requires_grad_(True)

    (gradient,) = torch.autograd.grad(ambiguity_set.worst_case_expectation(loss), loss)

    assert torch.isfinite(gradient).all()
    assert bool((gradient >= -1e-12).all())
    assert abs(float(gradient.sum()) - 1.0) <= 1e-9
    assert bool(ambiguity_set.contains(gradient.detach().clamp_min(0.0)))


def test_saturated_gradient_is_the_nominal_restricted_to_the_top_scenarios() -> None:
    # Tied maxima: an even split across the ties would leave the set, while
    # the restricted nominal is the worst-case distribution at the boundary.
    nominal = NOMINAL
    loss = torch.tensor([0.0, 1.0, 5.0, 5.0], dtype=torch.float64, requires_grad=True)
    ambiguity_set = KLAmbiguitySet(nominal, radius=2.0, dual_solver=_solver())

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


def test_float32_wide_spread_is_finite() -> None:
    nominal = NOMINAL.to(torch.float32)
    loss = (1e4 * LOSS).to(torch.float32)
    ambiguity_set = ChiSquareAmbiguitySet(
        nominal, radius=0.3 * CHI_SQUARE_SATURATION, dual_solver=_solver()
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = chi_square_worst_case(
        NOMINAL.numpy(), 1e4 * LOSS.numpy(), 0.3 * CHI_SQUARE_SATURATION
    )
    assert torch.isfinite(result)
    assert abs(float(result) - reference) <= 1e-3 * reference


# --- Starting point ----------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_default_start_is_the_same_standardized_point_on_any_scale(
    scale: float,
) -> None:
    # A raw-unit default of eta = 1 would sit far from the optimum of a loss
    # of spread 1e6, which is what made a wide spread overflow.
    solver = _RecordingSolver()
    ambiguity_set = ChiSquareAmbiguitySet(NOMINAL, radius=0.5, dual_solver=solver)

    ambiguity_set.worst_case_expectation(scale * LOSS + 3.0)

    assert solver.initial_point is not None
    assert torch.equal(solver.initial_point, torch.zeros(2, dtype=torch.float64))


def test_explicit_start_values_are_read_in_raw_loss_units() -> None:
    solver = _RecordingSolver()
    ambiguity_set = KLAmbiguitySet(
        NOMINAL, radius=0.5, dual_solver=solver, initial_log_eta=math.log(10.0)
    )

    ambiguity_set.worst_case_expectation(10.0 * LOSS)

    # eta = 10 on a loss of spread 50 is eta / spread = 0.2 once standardized.
    assert solver.initial_point is not None
    assert torch.allclose(
        solver.initial_point,
        torch.tensor(math.log(10.0 / 50.0), dtype=torch.float64),
    )


def test_chi_square_initial_values_must_be_given_together() -> None:
    with pytest.raises(ValueError, match="together"):
        ChiSquareAmbiguitySet(NOMINAL, radius=0.5, initial_log_eta=0.0)
    with pytest.raises(ValueError, match="together"):
        ChiSquareAmbiguitySet(NOMINAL, radius=0.5, initial_lam=0.0)


def test_dual_ambiguity_set_is_abstract_over_its_formulation() -> None:
    with pytest.raises(TypeError):
        DualAmbiguitySet(  # type: ignore[abstract]
            NOMINAL,
            divergence=KLAmbiguitySet(NOMINAL, radius=0.1).divergence,
            radius=0.1,
            dual_solver=None,
            initial_dual_point=torch.tensor(0.0),
        )
