"""Device and dtype coverage for every ambiguity set.

The precision tests back the dtype support table in `docs/numerics.md`: a
half-precision worst case is exactly the float32 solve rounded once, and a
float32 worst case agrees with a float64 solve of the same rounded inputs to
within `FLOAT32_TOLERANCE_EPS` units of float32 epsilon.
"""

import math
from collections.abc import Callable

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.f_divergence import PhiDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.mmd_dro import MMDAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
)
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.instances import INSTANCES, Instance
from tests.dro.specs import SetSpec, spec_parameters

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device is available"
)

NOMINAL = (0.25, 0.5, 0.25)

CANDIDATE = (0.2, 0.5, 0.3)

LOSS = (0.0, 1.0, 2.0)

COST = ((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0))

RADIUS = 0.2

RADII = (1e-6, 0.1, 1.0, 10.0)

# Documented float32 accuracy, in units of float32 epsilon times the larger of
# the worst case's magnitude and the loss spread. The worst draw measured over
# 2400 random problems was 93 (Wasserstein); typical errors are a few units.
FLOAT32_TOLERANCE_EPS = 128.0

AmbiguitySetFactory = Callable[[torch.dtype], AmbiguitySet]


def _kl_generator(ratio: torch.Tensor) -> torch.Tensor:
    """Convex generator `phi(t) = t log(t) - t + 1` of the KL divergence."""
    return ratio * torch.log(ratio) - ratio + 1.0


def _kl_conjugate(scaled_shift: torch.Tensor) -> torch.Tensor:
    """Convex conjugate `phi*(s) = exp(s) - 1` of `_kl_generator`."""
    return torch.exp(scaled_shift) - 1.0


def _solver() -> GradientDescent:
    """Return a short dual solve: these tests check dtypes, not convergence."""
    return GradientDescent(step_size=0.05, max_iter=50, tol=1e-8)


def _nominal(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(NOMINAL, dtype=dtype)


def _kl(dtype: torch.dtype) -> AmbiguitySet:
    return KLAmbiguitySet(_nominal(dtype), radius=RADIUS)


def _phi(dtype: torch.dtype) -> AmbiguitySet:
    return PhiAmbiguitySet(
        _nominal(dtype),
        divergence=PhiDivergence(phi=_kl_generator),
        radius=RADIUS,
        phi_conjugate=_kl_conjugate,
        dual_solver=_solver(),
    )


def _chi_square(dtype: torch.dtype) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(_nominal(dtype), radius=RADIUS)


def _total_variation(dtype: torch.dtype) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(_nominal(dtype), radius=RADIUS)


def _cvar(dtype: torch.dtype) -> AmbiguitySet:
    # A tensor level exercises the buffers; its entries are dyadic so every
    # dtype, half precision included, holds exactly the same levels.
    return CVaRAmbiguitySet(
        _nominal(dtype), alpha=torch.tensor([0.75, 0.5, 0.125], dtype=dtype)
    )


def _wasserstein(dtype: torch.dtype) -> AmbiguitySet:
    return WassersteinAmbiguitySet(
        _nominal(dtype),
        cost=torch.tensor(COST, dtype=dtype),
        radius=RADIUS,
    )


def _sinkhorn(dtype: torch.dtype) -> AmbiguitySet:
    return SinkhornAmbiguitySet(
        _nominal(dtype),
        cost=torch.tensor(COST, dtype=dtype),
        epsilon=0.5,
        radius=RADIUS,
    )


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "chi_square": _chi_square,
    "cvar": _cvar,
    "sinkhorn": _sinkhorn,
    "kl": _kl,
    "phi": _phi,
    "total_variation": _total_variation,
    "wasserstein": _wasserstein,
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> AmbiguitySetFactory:
    """Build one ambiguity set of each kind from a dtype."""
    return FACTORIES[str(request.param)]


def test_worst_case_expectation_preserves_input_dtype_and_device(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    device = torch.device("cpu")
    ambiguity_set = factory(dtype)
    loss = torch.tensor(LOSS, dtype=dtype, device=device)

    value = ambiguity_set.worst_case_expectation(loss)

    assert value.dtype == dtype
    assert value.device == loss.device


def test_contains_returns_a_bool_tensor_on_the_input_device(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    device = torch.device("cpu")
    ambiguity_set = factory(dtype)
    candidate = torch.tensor(CANDIDATE, dtype=dtype, device=device)

    membership = ambiguity_set.contains(candidate)

    assert membership.dtype == torch.bool
    assert membership.device == candidate.device


@requires_cuda
def test_buffers_follow_the_module_to_cuda(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    # Ambiguity sets are nn.Modules precisely so the nominal distribution, the
    # initial dual point, and any tensor state their divergence holds move
    # together on a single .to(device) call.
    ambiguity_set = factory(dtype).to(torch.device("cuda"))

    names = dict(ambiguity_set.named_buffers())
    assert "nominal" in names
    for name, buffer in names.items():
        assert buffer.device.type == "cuda", name
        assert buffer.dtype == dtype, name


@requires_cuda
def test_cuda_worst_case_expectation_matches_cpu(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    cpu_value = factory(dtype).worst_case_expectation(torch.tensor(LOSS, dtype=dtype))
    device = torch.device("cuda")
    ambiguity_set = factory(dtype).to(device)
    loss = torch.tensor(LOSS, dtype=dtype, device=device)

    value = ambiguity_set.worst_case_expectation(loss)

    assert value.dtype == dtype
    assert value.device.type == "cuda"
    torch.testing.assert_close(value.cpu(), cpu_value, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_tensor_state_stays_on_the_solving_device(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    device = torch.device("cuda")
    ambiguity_set = factory(dtype).to(device)
    loss = torch.tensor(LOSS, dtype=dtype, device=device)

    ambiguity_set.worst_case_expectation(loss)

    for name, buffer in ambiguity_set.named_buffers():
        assert buffer.device.type == "cuda", name


def _rounded(instance: Instance, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Return `(nominal, loss, cost)` rounded to `dtype`."""
    return tuple(tensor.to(dtype) for tensor in instance.tensors())


def _random_problem(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Draw a float32 problem spanning support sizes, scales, and radii.

    Returns:
        `(nominal, loss, cost, radius)`, with nominal masses down to `1e-6`,
        a loss spread between `1e-3` and `1e3` around an offset of order
        ten, and a radius between `1e-6` and `10`.
    """
    generator = torch.Generator().manual_seed(seed)

    def uniform() -> float:
        return float(torch.rand((), generator=generator, dtype=torch.float64))

    size = int(torch.randint(1, 200, (), generator=generator))
    logits = 1.5 * torch.randn(size, generator=generator, dtype=torch.float64)
    nominal = torch.clamp(torch.softmax(logits, dim=0), min=1e-6)
    nominal = nominal / torch.sum(nominal)
    scale = 10.0 ** (6.0 * uniform() - 3.0)
    offset = 10.0 * (2.0 * uniform() - 1.0)
    loss = offset + scale * torch.randn(size, generator=generator, dtype=torch.float64)
    position = torch.rand(size, generator=generator, dtype=torch.float64)
    cost = torch.abs(position[:, None] - position[None, :])
    radius = 10.0 ** (7.0 * uniform() - 6.0)
    return nominal.float(), loss.float(), cost.float(), radius


Builder = Callable[[torch.Tensor, torch.Tensor, float], AmbiguitySet]


def _sinkhorn_set(
    nominal: torch.Tensor, cost: torch.Tensor, radius: float
) -> AmbiguitySet:
    return SinkhornAmbiguitySet(nominal, cost, epsilon=0.5, radius=radius)


def _mmd_set(nominal: torch.Tensor, cost: torch.Tensor, radius: float) -> AmbiguitySet:
    return MMDAmbiguitySet(nominal, torch.exp(-cost), radius)


def _gram_condition(cost: torch.Tensor) -> float:
    """Return the condition number of `_mmd_set`'s Laplacian Gram matrix."""
    eigenvalues = torch.linalg.eigvalsh(torch.exp(-cost.double()))
    return float(eigenvalues[-1] / torch.clamp(eigenvalues[0], min=1e-300))


# Sets outside `SPECS` (Sinkhorn is centred off the nominal; MMD needs a
# kernel), each with its float32 bound in units of epsilon as a function of
# the cost. MMD's float32 error is that of factoring its Gram matrix in
# float32, up to about its condition number in ulps: measured within half of
# `128 + condition` over 428 problems with conditions from 1 to 3e15.
EXTRA_FLOAT32_SETS: dict[str, tuple[Builder, Callable[[torch.Tensor], float]]] = {
    "sinkhorn": (_sinkhorn_set, lambda cost: FLOAT32_TOLERANCE_EPS),
    "mmd": (_mmd_set, lambda cost: FLOAT32_TOLERANCE_EPS + _gram_condition(cost)),
}


def _assert_float32_accuracy(
    spec: SetSpec | Builder,
    nominal: torch.Tensor,
    loss: torch.Tensor,
    cost: torch.Tensor,
    radius: float,
    tolerance_eps: float = FLOAT32_TOLERANCE_EPS,
) -> None:
    """Check a float32 solve against float64 on the same float32 inputs."""
    build = spec.build if isinstance(spec, SetSpec) else spec
    value = build(nominal, cost, radius).worst_case_expectation(loss)
    reference = build(nominal.double(), cost.double(), radius)
    expected = float(reference.worst_case_expectation(loss.double()))
    spread = float(torch.amax(loss) - torch.amin(loss))
    tolerance = (
        tolerance_eps
        * torch.finfo(torch.float32).eps
        * max(abs(expected), spread, math.ulp(0.0))
    )

    assert value.dtype == torch.float32
    assert abs(float(value) - expected) <= tolerance


@spec_parameters
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda instance: instance.name)
@pytest.mark.parametrize("radius", RADII)
def test_float32_matches_float64_on_the_same_inputs(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    nominal, loss, cost = _rounded(instance, torch.float32)

    _assert_float32_accuracy(spec, nominal, loss, cost, radius)


@spec_parameters
@pytest.mark.parametrize("seed", range(16))
def test_float32_matches_float64_on_random_problems(spec: SetSpec, seed: int) -> None:
    nominal, loss, cost, radius = _random_problem(seed)

    _assert_float32_accuracy(spec, nominal, loss, cost, radius)


@pytest.mark.parametrize("name", sorted(EXTRA_FLOAT32_SETS))
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda instance: instance.name)
@pytest.mark.parametrize("radius", RADII)
def test_float32_matches_float64_on_the_same_inputs_beyond_specs(
    name: str, instance: Instance, radius: float
) -> None:
    build, tolerance_eps = EXTRA_FLOAT32_SETS[name]
    nominal, loss, cost = _rounded(instance, torch.float32)

    _assert_float32_accuracy(build, nominal, loss, cost, radius, tolerance_eps(cost))


@pytest.mark.parametrize("name", sorted(EXTRA_FLOAT32_SETS))
@pytest.mark.parametrize("seed", range(16))
def test_float32_matches_float64_on_random_problems_beyond_specs(
    name: str, seed: int
) -> None:
    build, tolerance_eps = EXTRA_FLOAT32_SETS[name]
    nominal, loss, cost, radius = _random_problem(seed)

    _assert_float32_accuracy(build, nominal, loss, cost, radius, tolerance_eps(cost))


def test_mmd_float32_error_tracks_the_gram_condition_number() -> None:
    # Random problem 13 has an ill-conditioned Gram matrix (points nearly
    # coincide, condition about 5e6): its float32 error is set by the Gram
    # factorization and stays below its condition-number bound. How far it
    # exceeds the 128-ulp bound of the other sets depends on the platform's
    # BLAS and thread count (160 to 1700 ulps on one machine, 77 on another),
    # so the bound's slack is asserted on the condition number, not on the
    # measured error.
    nominal, loss, cost, radius = _random_problem(13)
    condition = _gram_condition(cost)
    assert condition > FLOAT32_TOLERANCE_EPS
    value = float(_mmd_set(nominal, cost, radius).worst_case_expectation(loss))
    reference = _mmd_set(nominal.double(), cost.double(), radius)
    expected = float(reference.worst_case_expectation(loss.double()))
    spread = float(torch.amax(loss) - torch.amin(loss))
    ulps = abs(value - expected) / (
        torch.finfo(torch.float32).eps * max(abs(expected), spread)
    )

    assert ulps <= FLOAT32_TOLERANCE_EPS + condition


@spec_parameters
@pytest.mark.parametrize("instance", INSTANCES, ids=lambda instance: instance.name)
@pytest.mark.parametrize("radius", RADII)
def test_half_precision_is_the_float32_solve_rounded_once(
    spec: SetSpec, instance: Instance, radius: float, half_dtype: torch.dtype
) -> None:
    nominal, loss, cost = _rounded(instance, half_dtype)

    value = spec.build(nominal, cost, radius).worst_case_expectation(loss)

    float32_set = spec.build(nominal.float(), cost.float(), radius)
    expected = float32_set.worst_case_expectation(loss.float())
    assert value.dtype == half_dtype
    assert torch.equal(value, expected.to(half_dtype))


def test_every_set_returns_a_half_precision_result(
    factory: AmbiguitySetFactory, half_dtype: torch.dtype
) -> None:
    ambiguity_set = factory(half_dtype)
    loss = torch.tensor(LOSS, dtype=half_dtype)

    value = ambiguity_set.worst_case_expectation(loss)

    expected = factory(torch.float32).worst_case_expectation(loss.float())
    assert value.dtype == half_dtype
    assert torch.equal(value, expected.to(half_dtype))


def test_a_half_precision_loss_on_a_float32_set_returns_float32(
    factory: AmbiguitySetFactory, half_dtype: torch.dtype
) -> None:
    loss = torch.tensor(LOSS, dtype=half_dtype)

    value = factory(torch.float32).worst_case_expectation(loss)

    expected = factory(torch.float32).worst_case_expectation(loss.float())
    assert value.dtype == torch.float32
    assert torch.equal(value, expected)


def test_a_float64_loss_on_a_float32_set_is_solved_in_float64(
    factory: AmbiguitySetFactory,
) -> None:
    loss = torch.tensor(LOSS, dtype=torch.float64)

    value = factory(torch.float32).worst_case_expectation(loss)

    assert value.dtype == torch.float64


@spec_parameters
def test_half_precision_gradient_is_the_float32_gradient_rounded(
    spec: SetSpec, half_dtype: torch.dtype
) -> None:
    nominal, loss, cost = _rounded(INSTANCES[0], half_dtype)
    half_loss = loss.clone().requires_grad_(True)
    float32_loss = loss.float().requires_grad_(True)

    spec.build(nominal, cost, 0.3).worst_case_expectation(half_loss).backward()
    float32_set = spec.build(nominal.float(), cost.float(), 0.3)
    float32_set.worst_case_expectation(float32_loss).backward()

    assert half_loss.grad is not None
    assert float32_loss.grad is not None
    assert half_loss.grad.dtype == half_dtype
    assert torch.equal(half_loss.grad, float32_loss.grad.to(half_dtype))


@spec_parameters
def test_cpu_autocast_does_not_change_the_worst_case(spec: SetSpec) -> None:
    nominal, loss, cost = _rounded(INSTANCES[0], torch.float32)
    ambiguity_set = spec.build(nominal, cost, 0.3)
    expected = ambiguity_set.worst_case_expectation(loss)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = ambiguity_set.worst_case_expectation(loss)

    assert value.dtype == torch.float32
    assert torch.equal(value, expected)
