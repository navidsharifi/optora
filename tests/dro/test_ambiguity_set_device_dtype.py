"""Device and dtype coverage for every ambiguity set."""

from collections.abc import Callable

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.f_divergence import PhiDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
)
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device is available"
)

NOMINAL = (0.25, 0.5, 0.25)

CANDIDATE = (0.2, 0.5, 0.3)

LOSS = (0.0, 1.0, 2.0)

COST = ((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0))

RADIUS = 0.2

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
    return KLAmbiguitySet(_nominal(dtype), radius=RADIUS, dual_solver=_solver())


def _phi(dtype: torch.dtype) -> AmbiguitySet:
    return PhiAmbiguitySet(
        _nominal(dtype),
        divergence=PhiDivergence(phi=_kl_generator),
        radius=RADIUS,
        phi_conjugate=_kl_conjugate,
        dual_solver=_solver(),
    )


def _chi_square(dtype: torch.dtype) -> AmbiguitySet:
    return ChiSquareAmbiguitySet(_nominal(dtype), radius=RADIUS, dual_solver=_solver())


def _total_variation(dtype: torch.dtype) -> AmbiguitySet:
    return TotalVariationAmbiguitySet(_nominal(dtype), radius=RADIUS)


def _wasserstein(dtype: torch.dtype) -> AmbiguitySet:
    return WassersteinAmbiguitySet(
        _nominal(dtype),
        cost=torch.tensor(COST, dtype=dtype),
        radius=RADIUS,
        dual_solver=_solver(),
    )


FACTORIES: dict[str, AmbiguitySetFactory] = {
    "chi_square": _chi_square,
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
def test_warm_start_cache_stays_on_the_solving_device(
    factory: AmbiguitySetFactory, dtype: torch.dtype
) -> None:
    device = torch.device("cuda")
    ambiguity_set = factory(dtype).to(device)
    loss = torch.tensor(LOSS, dtype=dtype, device=device)

    ambiguity_set.worst_case_expectation(loss)

    for name, buffer in ambiguity_set.named_buffers():
        assert buffer.device.type == "cuda", name
