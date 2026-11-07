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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from optora.core.dro_base import AmbiguitySet
from optora.divergences.mmd import MMDDivergence
from optora.dro import mmd_dro
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
    """A nominal, a Laplacian Gram on random points, and a loss.

    `duplicate` marks a Gram made singular by repeating the first support
    point as the second.
    """

    name: str
    nominal: torch.Tensor
    gram: torch.Tensor
    loss: torch.Tensor
    duplicate: bool = False

    @property
    def diameter(self) -> float:
        diagonal = torch.diagonal(self.gram)
        squared = diagonal[:, None] + diagonal[None, :] - 2.0 * self.gram
        return math.sqrt(float(torch.amax(squared)))

    def reference(self, radius: float, loss: torch.Tensor | None = None) -> float:
        loss = self.loss if loss is None else loss
        # A singular Gram's ball of radius zero is a face of the simplex,
        # which Clarabel solves exactly as the degenerate cone program.
        if radius == 0.0 and not self.duplicate:
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
        duplicate=duplicate,
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
DEFINITE = tuple(problem for problem in PROBLEMS if not problem.duplicate)
SINGULAR = (
    PROBLEMS[3],
    make_problem("duplicate_large", 12, 7, duplicate=True),
    make_problem("duplicate_zero_mass", 40, 8, zero_mass=True, duplicate=True),
)

problem_parameters = pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEM_IDS)
definite_parameters = pytest.mark.parametrize(
    "problem", DEFINITE, ids=[problem.name for problem in DEFINITE]
)
singular_parameters = pytest.mark.parametrize(
    "problem", SINGULAR, ids=[problem.name for problem in SINGULAR]
)


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


@definite_parameters
def test_a_zero_radius_is_the_nominal_expectation(problem: Problem) -> None:
    value = build(problem, 0.0).worst_case_expectation(problem.loss)

    assert float(value) == float(torch.sum(problem.nominal * problem.loss))


def merged_duplicates(problem: Problem) -> float:
    """Return the zero-radius worst case of a duplicated pair by hand.

    Mass moves between the two copies of a support point at no discrepancy,
    so the pair's whole mass sits on the larger of its two losses.
    """
    nominal, loss = problem.nominal, problem.loss
    pair_mass = float(nominal[:2].sum())
    return float(nominal @ loss - nominal[:2] @ loss[:2]) + pair_mass * float(
        loss[:2].max()
    )


@singular_parameters
@pytest.mark.parametrize("as_tensor", [False, True], ids=["float", "tensor"])
def test_a_zero_radius_of_a_singular_gram_is_the_worst_case_over_its_face(
    problem: Problem, as_tensor: bool
) -> None:
    # A duplicated support point makes the Gram singular: moving mass between
    # the copies costs no discrepancy, so the ball of radius zero is a face
    # of the simplex rather than {nominal} and the shortcut to the nominal
    # expectation understated the worst case (#98). The oracle solves the
    # degenerate cone program exactly.
    radius = torch.tensor(0.0, dtype=DTYPE) if as_tensor else 0.0
    expected = problem.reference(0.0)

    value = build(problem, radius).worst_case_expectation(problem.loss)

    assert expected == pytest.approx(
        merged_duplicates(problem), abs=1e-9 * spread(problem.loss)
    )
    assert expected >= float(problem.nominal @ problem.loss) - 1e-12
    assert float(value) == pytest.approx(
        expected, abs=ORACLE_TOLERANCE * spread(problem.loss)
    )


def test_a_singular_gram_with_a_mixed_tensor_radius_extrapolates_its_zero_entries() -> (
    None
):
    problem = SINGULAR[0]
    radii = torch.tensor([0.0, 0.3 * problem.diameter], dtype=DTYPE)
    expected = torch.tensor(
        [problem.reference(float(radius)) for radius in radii], dtype=DTYPE
    )

    values = build(problem, radii).worst_case_expectation(problem.loss)

    assert float(expected[0]) > float(problem.nominal @ problem.loss)
    torch.testing.assert_close(
        values, expected, rtol=0.0, atol=ORACLE_TOLERANCE * spread(problem.loss)
    )


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


@pytest.mark.parametrize(
    "problem", SINGULAR[:2], ids=[problem.name for problem in SINGULAR[:2]]
)
def test_zero_radius_gradients_of_a_singular_gram_are_those_of_its_face(
    problem: Problem,
) -> None:
    # By Danskin's theorem the loss gradient is the face's maximizer, the
    # duplicated pair's mass on its larger loss. The nominal and radius
    # derivatives come from the probe solve, so they are first order: the
    # nominal gradient lands within about ten probe radii (sqrt(tol) times
    # the radius cap, 4e-5 here) of the face's exact gradient, and the
    # radius derivative within 3e-4 of a Richardson extrapolation of the
    # oracle.
    nominal = problem.nominal.clone().requires_grad_(True)
    radius = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    loss = problem.loss.clone().requires_grad_(True)

    MMDAmbiguitySet(nominal, problem.gram, radius).worst_case_expectation(
        loss
    ).backward()

    assert loss.grad is not None and nominal.grad is not None
    assert radius.grad is not None
    maximizer = problem.nominal.clone()
    larger = int(torch.argmax(problem.loss[:2]))
    maximizer[:2] = 0.0
    maximizer[larger] = problem.nominal[:2].sum()
    torch.testing.assert_close(loss.grad, maximizer, rtol=0.0, atol=1e-4)
    face_gradient = problem.loss.clone()
    face_gradient[:2] = problem.loss[:2].max()
    torch.testing.assert_close(
        nominal.grad, face_gradient, rtol=0.0, atol=1e-3 * spread(problem.loss)
    )
    step = 1e-5 * problem.diameter
    base = problem.reference(0.0)
    coarse = (problem.reference(step) - base) / step
    fine = (problem.reference(0.5 * step) - base) / (0.5 * step)
    assert float(radius.grad) == pytest.approx(2.0 * fine - coarse, rel=3e-4)


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


@pytest.fixture(params=[1, 2, 3, 4], ids=lambda count: f"{count}-threads")
def thread_count(request: pytest.FixtureRequest) -> Iterator[int]:
    """Run under each intra-op thread count, restoring the previous one."""
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    yield request.param
    torch.set_num_threads(previous)


def laplacian_problem(
    bandwidth: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a Laplacian-kernel Gram on 256 grid points, 16 losses, directions."""
    size, batch = 256, 16
    points = torch.linspace(0.0, 1.0, size, dtype=DTYPE)
    gram = torch.exp(-torch.abs(points[:, None] - points[None, :]) / bandwidth)
    generator = torch.Generator().manual_seed(seed)
    loss = torch.randn(batch, size, generator=generator, dtype=DTYPE)
    direction = torch.randn(batch, size, generator=generator, dtype=DTYPE)
    return gram, loss, direction


def condition_number(gram: torch.Tensor) -> float:
    """Return the spectral condition number of a positive definite Gram."""
    eigenvalues = torch.linalg.eigvalsh(gram)
    return float(eigenvalues[-1] / eigenvalues[0])


@pytest.mark.parametrize("bandwidth", [0.1, 0.2, 1.0])
def test_float32_radius_multiplier_agrees_with_float64(
    bandwidth: float, thread_count: int
) -> None:
    # With two refinement passes the float32 interior-point solve stalled on
    # rows that depended on the platform and thread count (a failed Cholesky
    # near the solution, #88) and their radius multiplier was 2e-2 to 1e1
    # off. Even on rows it marks converged the iterate is held only by the
    # dual-residual tolerance, and drifted to 1.4e-2 on a Windows runner
    # with PyTorch 2.2. The multiplier recovered from complementary
    # stationarity is accurate to about 1e-3 here on every row, unrelated
    # to the Gram conditioning that bounds the Hessian.
    gram, loss, _ = laplacian_problem(bandwidth, 0)
    batch, size = loss.shape

    def radius_gradient(dtype: torch.dtype) -> torch.Tensor:
        radius = torch.full((batch,), 0.1, dtype=dtype, requires_grad=True)
        ambiguity_set = MMDAmbiguitySet(
            torch.full((size,), 1.0 / size, dtype=dtype), gram.to(dtype), radius
        )
        value = ambiguity_set.worst_case_expectation(loss.to(dtype)).sum()
        (gradient,) = torch.autograd.grad(value, radius)
        return gradient.double()

    single = radius_gradient(torch.float32)
    expected = radius_gradient(torch.float64)

    assert bool(torch.all(expected > 0.0))
    errors = torch.abs(single - expected) / expected
    assert float(errors.max()) < 1e-2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_radius_gradient_vanishes_where_the_ball_holds_the_worst_vertex(
    dtype: torch.dtype,
) -> None:
    # Below the radius cap but past the MMD from the nominal to the vertex
    # of the largest loss, the worst case is that vertex and the radius
    # constraint is slack. The slope recovered from stationarity is then set
    # by rounding-level masses, up to 3.4 here in float32, so it may replace
    # the interior-point multiplier only where the constraint is active.
    gram, loss, _ = laplacian_problem(1.0, 0)
    batch, size = loss.shape
    nominal = torch.full((size,), 1.0 / size, dtype=DTYPE)
    vertex = torch.nn.functional.one_hot(torch.argmax(loss, dim=-1), size).to(DTYPE)
    offset = vertex - nominal
    vertex_distance = torch.sqrt(torch.sum((offset @ gram) * offset, dim=-1))
    radius = torch.full((batch,), 0.8, dtype=dtype, requires_grad=True)
    ambiguity_set = MMDAmbiguitySet(nominal.to(dtype), gram.to(dtype), radius)
    assert bool(torch.all(vertex_distance < 0.8))
    assert bool(torch.all(radius < ambiguity_set.radius_ceiling))

    value = ambiguity_set.worst_case_expectation(loss.to(dtype))
    (gradient,) = torch.autograd.grad(value.sum(), radius)

    assert torch.allclose(value.double(), torch.amax(loss, dim=-1), atol=1e-3)
    assert float(torch.amax(torch.abs(gradient))) < 1e-3 * spread(loss)


def test_float32_second_derivatives_survive_a_thread_count_change(
    thread_count: int,
) -> None:
    # On Windows MKL, any torch.set_num_threads call left the batched float32
    # LU of the Hessian system returning invalid pivots: the backward pass
    # raised, or got nan and silently zeroed the Hessian. Later, one row was
    # 0.69 off on macOS (#87): a row whose float32 solve had stalled on a
    # failed Cholesky kept stale orthant multipliers, and the support test
    # `mass > multiplier` behind the Hessian picked a different support. The
    # Hessian jumps where the support changes, and a converged float32 solve
    # tells a mass from its multiplier only down to about 2e-4, the square
    # root of its gap floor; on this problem the smallest support mass and
    # off-support multiplier both exceed 1e-3, and the two are at least a
    # factor 100 apart on every row.
    gram, loss, direction = laplacian_problem(1.0, 0)
    size = loss.shape[-1]

    def hessian_vector_product(dtype: torch.dtype) -> torch.Tensor:
        ambiguity_set = MMDAmbiguitySet(
            torch.full((size,), 1.0 / size, dtype=dtype), gram.to(dtype), 0.1
        )
        point = loss.to(dtype).requires_grad_()
        value = ambiguity_set.worst_case_expectation(point).sum()
        (gradient,) = torch.autograd.grad(value, point, create_graph=True)
        (product,) = torch.autograd.grad(
            torch.sum(gradient * direction.to(dtype)), point
        )
        return product.double()

    single = hessian_vector_product(torch.float32)
    expected = hessian_vector_product(torch.float64)

    # Float32 accuracy is set by the Gram matrix's condition number, about
    # 1e5 here (see docs/numerics.md).
    norms = torch.linalg.vector_norm(expected, dim=-1)
    assert bool(torch.all(norms > 0.0))
    errors = torch.linalg.vector_norm(single - expected, dim=-1) / norms
    bound = 4.0 * condition_number(gram) * torch.finfo(torch.float32).eps
    assert float(errors.max()) < bound


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


def test_a_failed_factorization_freezes_its_element_and_ends_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An element whose normal matrix fails to factor cannot move: the same
    # point gives the same matrix at the next iteration. It used to be
    # retried until max_iter while the converged elements waited (#88); now
    # it is frozen where it stands and the loop ends once the others
    # converge.
    generator = torch.Generator().manual_seed(0)
    loss = torch.randn(3, 7, generator=generator, dtype=DTYPE)
    radius = 0.2 * SMOOTH.diameter
    search_direction = mmd_dro._search_direction
    calls = 0

    def failing_search_direction(*args: Any) -> tuple[Any, torch.Tensor]:
        nonlocal calls
        calls += 1
        direction, failed = search_direction(*args)
        if calls > 3:
            failed = failed | torch.tensor([True, False, False])
        return direction, failed

    clean = build(SMOOTH, radius).worst_case_expectation(loss)
    monkeypatch.setattr(mmd_dro, "_search_direction", failing_search_direction)
    frozen = build(SMOOTH, radius, max_iter=200).worst_case_expectation(loss)

    assert calls < 200
    torch.testing.assert_close(frozen[1:], clean[1:], rtol=0.0, atol=1e-9)
    assert math.isfinite(float(frozen[0]))
    assert float(loss[0].min()) <= float(frozen[0]) <= float(loss[0].max())


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
