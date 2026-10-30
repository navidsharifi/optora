r"""Tests for `MMDAmbiguitySet`.

The worst case over an MMD ball is a second-order cone program, so every
value here is refereed by `tests.dro.oracles.mmd_worst_case`, which solves
the same primal with Clarabel, an interior-point implementation that shares
no code with optora's. Grams are Laplacian, $e^{-\lVert x - y \rVert}$,
which is positive definite on distinct points and well conditioned; a
duplicated point makes one singular on purpose. Values are compared in units
of the loss spread, since the set solves the program on the standardized
loss to a relative duality gap of `tol`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from optora.core.dro_base import AmbiguitySet
from optora.divergences.mmd import MMDDivergence
from optora.dro.mmd_dro import MMDAmbiguitySet
from tests.dro.oracles import mmd_worst_case

DTYPE = torch.float64

# Agreement with the oracle, in units of the loss spread.
ORACLE_TOLERANCE = 1e-9

# Fractions of the kernel diameter of the simplex, from a vanishing radius to
# past saturation.
RADIUS_FRACTIONS = (1e-6, 1e-3, 0.05, 0.3, 0.9, 1.5)


@dataclass(frozen=True)
class Problem:
    """A nominal, a Laplacian Gram on random points, and a loss."""

    name: str
    nominal: torch.Tensor
    gram: torch.Tensor
    loss: torch.Tensor

    @property
    def diameter(self) -> float:
        diagonal = torch.diagonal(self.gram)
        squared = diagonal[:, None] + diagonal[None, :] - 2.0 * self.gram
        return math.sqrt(float(torch.amax(squared)))

    def reference(self, radius: float, loss: torch.Tensor | None = None) -> float:
        loss = self.loss if loss is None else loss
        if radius == 0.0:
            return float(self.nominal @ loss)
        return mmd_worst_case(
            self.nominal.numpy(), self.gram.numpy(), loss.numpy(), radius
        )


def make_problem(
    name: str,
    size: int,
    seed: int,
    zero_mass: bool = False,
    duplicate: bool = False,
    loss_scale: float = 1.0,
) -> Problem:
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(size, 2))
    if duplicate:
        points[1] = points[0]
    nominal = generator.dirichlet(np.ones(size))
    if zero_mass:
        nominal[: max(1, size // 3)] = 0.0
        nominal /= nominal.sum()
    gram = np.exp(-np.linalg.norm(points[:, None] - points[None], axis=-1))
    loss = loss_scale * generator.normal(size=size)
    return Problem(
        name,
        torch.tensor(nominal, dtype=DTYPE),
        torch.tensor(gram, dtype=DTYPE),
        torch.tensor(loss, dtype=DTYPE),
    )


PROBLEMS = (
    make_problem("two_points", 2, 0),
    make_problem("small", 4, 1),
    make_problem("zero_mass", 6, 2, zero_mass=True),
    make_problem("duplicate_point", 5, 3, duplicate=True),
    make_problem("wide_loss", 12, 4, loss_scale=1e3),
    make_problem("moderate", 40, 5, zero_mass=True),
)
PROBLEM_IDS = [problem.name for problem in PROBLEMS]
SMOOTH = make_problem("smooth", 7, 6)

problem_parameters = pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEM_IDS)


def build(
    problem: Problem, radius: float | torch.Tensor, **kwargs: Any
) -> MMDAmbiguitySet:
    return MMDAmbiguitySet(problem.nominal, problem.gram, radius, **kwargs)


def spread(loss: torch.Tensor) -> float:
    return float(loss.max() - loss.min())


# --- Values against the oracle ----------------------------------------------


@problem_parameters
@pytest.mark.parametrize("fraction", RADIUS_FRACTIONS)
def test_matches_the_second_order_cone_oracle(
    problem: Problem, fraction: float
) -> None:
    radius = fraction * problem.diameter

    value = build(problem, radius).worst_case_expectation(problem.loss)

    error = abs(float(value) - problem.reference(radius))
    assert error <= ORACLE_TOLERANCE * spread(problem.loss)


@pytest.mark.parametrize("fraction", [1e-3, 0.1, 0.5])
def test_matches_the_oracle_on_a_large_support(fraction: float) -> None:
    # Fixed-budget barrier and Mehrotra prototypes without Nesterov-Todd
    # scaling failed here (issue #67).
    problem = make_problem("large", 256, 7, zero_mass=True)
    radius = fraction * problem.diameter

    value = build(problem, radius).worst_case_expectation(problem.loss)

    error = abs(float(value) - problem.reference(radius))
    assert error <= ORACLE_TOLERANCE * spread(problem.loss)


@problem_parameters
def test_a_zero_radius_is_the_nominal_expectation(problem: Problem) -> None:
    value = build(problem, 0.0).worst_case_expectation(problem.loss)

    assert float(value) == float(torch.sum(problem.nominal * problem.loss))


@problem_parameters
@pytest.mark.parametrize("multiple", [1.0, 2.0, 1e6])
def test_a_radius_past_the_diameter_is_the_maximum_loss(
    problem: Problem, multiple: float
) -> None:
    value = build(problem, multiple * problem.diameter).worst_case_expectation(
        problem.loss
    )

    assert float(value) == pytest.approx(
        float(problem.loss.max()), abs=ORACLE_TOLERANCE * spread(problem.loss)
    )


def test_a_mixed_tensor_radius_matches_scalar_radii() -> None:
    problem = PROBLEMS[2]
    radii = torch.tensor([0.0, 0.01, 0.2, 10.0], dtype=DTYPE) * problem.diameter

    values = build(problem, radii).worst_case_expectation(problem.loss)

    for radius, value in zip(radii.tolist(), values.tolist(), strict=True):
        assert value == pytest.approx(
            problem.reference(radius), abs=ORACLE_TOLERANCE * spread(problem.loss)
        )
    assert values[0] == torch.sum(problem.nominal * problem.loss)


def test_batched_losses_match_one_at_a_time() -> None:
    problem = PROBLEMS[3]
    generator = torch.Generator().manual_seed(0)
    losses = torch.randn(5, 5, generator=generator, dtype=DTYPE)
    ambiguity_set = build(problem, 0.3 * problem.diameter)

    values = ambiguity_set.worst_case_expectation(losses)

    assert values.shape == (5,)
    for loss, value in zip(losses, values, strict=True):
        assert float(value) == pytest.approx(
            float(ambiguity_set.worst_case_expectation(loss)), abs=1e-10
        )


@problem_parameters
def test_is_monotone_in_the_radius_and_bracketed_by_the_mean_and_maximum(
    problem: Problem,
) -> None:
    radii = torch.tensor(RADIUS_FRACTIONS, dtype=DTYPE) * problem.diameter
    tolerance = ORACLE_TOLERANCE * spread(problem.loss)

    values = build(problem, radii).worst_case_expectation(problem.loss)

    assert bool(torch.all(values[1:] >= values[:-1] - tolerance))
    assert float(values[0]) >= float(problem.nominal @ problem.loss) - tolerance
    assert float(values[-1]) <= float(problem.loss.max()) + tolerance


@pytest.mark.parametrize(("scale", "offset"), [(1e-3, 5.0), (7.0, -2.0), (1e4, 1e4)])
def test_is_equivariant_to_a_positive_affine_loss(scale: float, offset: float) -> None:
    ambiguity_set = build(SMOOTH, 0.2 * SMOOTH.diameter)

    base = ambiguity_set.worst_case_expectation(SMOOTH.loss)
    transformed = ambiguity_set.worst_case_expectation(scale * SMOOTH.loss + offset)

    assert float(transformed) == pytest.approx(
        scale * float(base) + offset, abs=1e-9 * scale * spread(SMOOTH.loss)
    )


def test_a_constant_loss_is_its_value() -> None:
    loss = torch.full((7,), 3.25, dtype=DTYPE)

    value = build(SMOOTH, 0.2 * SMOOTH.diameter).worst_case_expectation(loss)

    assert float(value) == pytest.approx(3.25, abs=1e-12)


# --- Derivatives in the loss ------------------------------------------------


@problem_parameters
@pytest.mark.parametrize("fraction", [0.05, 0.3])
def test_gradient_is_a_distribution_inside_the_ball_attaining_the_worst_case(
    problem: Problem, fraction: float
) -> None:
    radius = fraction * problem.diameter
    loss = problem.loss.clone().requires_grad_(True)

    value = build(problem, radius).worst_case_expectation(loss)
    value.backward()

    assert loss.grad is not None
    candidate = loss.grad
    assert float(candidate.min()) >= 0.0
    assert float(candidate.sum()) == pytest.approx(1.0, abs=1e-12)
    discrepancy = float(MMDDivergence(problem.gram)(candidate, problem.nominal))
    assert discrepancy <= radius * (1.0 + 1e-6) + 1e-9
    assert float(candidate @ problem.loss) == pytest.approx(
        float(value.detach()), abs=1e-12 * spread(problem.loss)
    )


def test_first_and_second_derivatives_match_finite_differences() -> None:
    function = build(SMOOTH, 0.2 * SMOOTH.diameter).worst_case_expectation
    loss = SMOOTH.loss.clone().requires_grad_(True)

    assert gradcheck(function, (loss,), eps=1e-6, atol=1e-6, rtol=1e-3)
    assert gradgradcheck(function, (loss,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_hessian_is_symmetric_positive_semidefinite_and_shift_invariant() -> None:
    hessian = torch.autograd.functional.hessian(
        build(SMOOTH, 0.2 * SMOOTH.diameter).worst_case_expectation, SMOOTH.loss
    )

    torch.testing.assert_close(hessian, hessian.mT, rtol=0.0, atol=1e-10)
    assert float(torch.linalg.eigvalsh(hessian).min()) >= -1e-8
    assert float(torch.linalg.eigvalsh(hessian).max()) > 1e-3
    torch.testing.assert_close(
        hessian @ torch.ones(7, dtype=DTYPE),
        torch.zeros(7, dtype=DTYPE),
        atol=1e-8,
        rtol=0,
    )


def test_hessian_vanishes_once_the_radius_saturates() -> None:
    hessian = torch.autograd.functional.hessian(
        build(SMOOTH, 3.0 * SMOOTH.diameter).worst_case_expectation, SMOOTH.loss
    )

    assert float(hessian.abs().max()) <= 1e-8


def test_curvature_term_is_skipped_without_a_loss_requiring_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_: object) -> torch.Tensor:
        raise AssertionError("the loss Hessian was built")

    monkeypatch.setattr("optora.dro.mmd_dro._loss_hessian", unexpected)
    ambiguity_set = build(SMOOTH, 0.2 * SMOOTH.diameter)

    ambiguity_set.worst_case_expectation(SMOOTH.loss)
    with pytest.raises(AssertionError, match="Hessian was built"):
        ambiguity_set.worst_case_expectation(SMOOTH.loss.clone().requires_grad_(True))


# --- Derivatives in the nominal and the radius --------------------------------


def test_parameter_gradients_match_finite_differences() -> None:
    radius = torch.tensor(0.2 * SMOOTH.diameter, dtype=DTYPE, requires_grad=True)
    logits = torch.log(SMOOTH.nominal).requires_grad_(True)

    def function(logits: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        nominal = torch.softmax(logits, dim=-1)
        return MMDAmbiguitySet(nominal, SMOOTH.gram, radius).worst_case_expectation(
            SMOOTH.loss
        )

    assert gradcheck(function, (logits, radius), eps=1e-6, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("fraction", [0.05, 0.3])
def test_radius_gradient_matches_the_oracle(fraction: float) -> None:
    problem = PROBLEMS[2]
    radius = torch.tensor(fraction * problem.diameter, dtype=DTYPE, requires_grad=True)

    build(problem, radius).worst_case_expectation(problem.loss).backward()

    step = 1e-6 * problem.diameter
    expected = (
        problem.reference(float(radius) + step)
        - problem.reference(float(radius) - step)
    ) / (2.0 * step)
    assert radius.grad is not None
    assert float(radius.grad) == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert float(radius.grad) > 0.0


def test_radius_gradient_vanishes_once_the_radius_saturates() -> None:
    radius = torch.tensor(5.0 * SMOOTH.diameter, dtype=DTYPE, requires_grad=True)

    build(SMOOTH, radius).worst_case_expectation(SMOOTH.loss).backward()

    assert radius.grad is not None
    assert float(radius.grad) == pytest.approx(0.0, abs=1e-8)


def test_zero_radius_gradients_are_those_of_the_nominal_expectation() -> None:
    nominal = SMOOTH.nominal.clone().requires_grad_(True)
    radius = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)

    MMDAmbiguitySet(nominal, SMOOTH.gram, radius).worst_case_expectation(
        SMOOTH.loss
    ).backward()

    assert nominal.grad is not None and radius.grad is not None
    torch.testing.assert_close(nominal.grad, SMOOTH.loss)
    # Richardson extrapolation of forward differences cancels their O(h)
    # curvature error, which a one-sided difference cannot avoid at zero.
    step = 1e-5 * SMOOTH.diameter
    base = SMOOTH.reference(0.0)
    coarse = (SMOOTH.reference(step) - base) / step
    fine = (SMOOTH.reference(0.5 * step) - base) / (0.5 * step)
    assert float(radius.grad) == pytest.approx(2.0 * fine - coarse, rel=1e-4)


# --- torch.func --------------------------------------------------------------


def test_torch_func_transforms_match_autograd() -> None:
    ambiguity_set = build(SMOOTH, 0.2 * SMOOTH.diameter)
    function = ambiguity_set.worst_case_expectation
    leaf = SMOOTH.loss.clone().requires_grad_(True)
    (expected_gradient,) = torch.autograd.grad(function(leaf), leaf)
    expected_hessian = torch.autograd.functional.hessian(function, SMOOTH.loss)

    torch.testing.assert_close(
        torch.func.grad(function)(SMOOTH.loss), expected_gradient, rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        torch.func.hessian(function)(SMOOTH.loss), expected_hessian, rtol=0.0, atol=1e-9
    )
    batch = torch.stack([SMOOTH.loss, -SMOOTH.loss, SMOOTH.loss.flip(-1)])
    torch.testing.assert_close(
        torch.func.vmap(function)(batch), function(batch), rtol=0.0, atol=1e-9
    )


# --- Precision, plumbing, and validation ------------------------------------


def test_float32_agrees_with_float64_to_its_precision() -> None:
    radius = 0.2 * SMOOTH.diameter
    expected = SMOOTH.reference(radius)

    value = MMDAmbiguitySet(
        SMOOTH.nominal.float(), SMOOTH.gram.float(), radius
    ).worst_case_expectation(SMOOTH.loss.float())

    assert value.dtype == torch.float32
    assert float(value) == pytest.approx(expected, abs=1e-4 * spread(SMOOTH.loss))


def test_reads_the_convergence_flag_only_at_checkpoints(
    host_sync_counter: Callable[[], Any],
) -> None:
    ambiguity_set = build(SMOOTH, 0.2 * SMOOTH.diameter, max_iter=40, check_interval=40)

    with host_sync_counter() as syncs:
        ambiguity_set.worst_case_expectation(SMOOTH.loss)

    assert len(syncs) == 1


def test_an_unconverged_budget_still_returns_a_distribution() -> None:
    loss = SMOOTH.loss.clone().requires_grad_(True)

    value = build(SMOOTH, 0.2 * SMOOTH.diameter, max_iter=2).worst_case_expectation(
        loss
    )
    value.backward()

    assert loss.grad is not None
    assert math.isfinite(float(value))
    assert float(loss.grad.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(loss.grad.min()) >= 0.0


def test_contains_uses_the_maximum_mean_discrepancy() -> None:
    ambiguity_set = build(SMOOTH, 0.1)
    near = 0.99 * SMOOTH.nominal + 0.01 / 7
    far = torch.zeros(7, dtype=DTYPE)
    far[0] = 1.0

    inside = ambiguity_set.contains(torch.stack([near, far]))

    assert inside.tolist() == [True, False]
    assert isinstance(ambiguity_set.divergence, MMDDivergence)
    assert ambiguity_set.gram is ambiguity_set.divergence.gram


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": 0}, "max_iter must be positive"),
        ({"tol": 0.0}, "tol must be positive"),
        ({"check_interval": 0}, "check_interval must be positive"),
    ],
)
def test_rejects_invalid_hyperparameters(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build(SMOOTH, 0.1, **kwargs)


def test_rejects_a_gram_that_does_not_match_the_nominal() -> None:
    with pytest.raises(ValueError, match="gram must have shape"):
        MMDAmbiguitySet(SMOOTH.nominal, SMOOTH.gram[:3, :3], 0.1)


def test_rejects_a_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius must be nonnegative"):
        build(SMOOTH, -0.1)


def test_validate_rejects_an_indefinite_gram() -> None:
    gram = SMOOTH.gram.clone()
    gram[0, 1] = gram[1, 0] = 5.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        MMDAmbiguitySet(SMOOTH.nominal, gram, 0.1, validate=True)


def test_is_an_ambiguity_set_and_moves_with_the_module() -> None:
    ambiguity_set = build(SMOOTH, 0.1)

    moved = ambiguity_set.to(torch.float32)

    assert isinstance(ambiguity_set, AmbiguitySet)
    assert moved.kernel_factor.dtype == torch.float32
    assert moved.gram.dtype == torch.float32
