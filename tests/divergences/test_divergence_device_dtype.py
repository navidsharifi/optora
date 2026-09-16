"""Device and dtype coverage for every divergence."""

from collections.abc import Callable

import pytest
import torch

from optora.core.divergence_base import Divergence
from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)
from optora.divergences.kl import KLDivergence
from optora.divergences.wasserstein import SinkhornDivergence

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device is available"
)

CANDIDATE = (0.2, 0.5, 0.3)

REFERENCE = (0.25, 0.5, 0.25)

COST = ((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0))

DivergenceFactory = Callable[[torch.dtype], Divergence]


def _squared_deviation(ratio: torch.Tensor) -> torch.Tensor:
    """Convex generator `phi(t) = (t - 1)^2` exercising the generic phi path."""
    return (ratio - 1.0) ** 2


def _phi(dtype: torch.dtype) -> Divergence:
    return PhiDivergence(phi=_squared_deviation)


def _chi_square(dtype: torch.dtype) -> Divergence:
    return ChiSquareDivergence()


def _total_variation(dtype: torch.dtype) -> Divergence:
    return TotalVariationDivergence()


def _kl(dtype: torch.dtype) -> Divergence:
    return KLDivergence()


def _sinkhorn(dtype: torch.dtype) -> Divergence:
    return SinkhornDivergence(cost=torch.tensor(COST, dtype=dtype), epsilon=0.1)


FACTORIES: dict[str, DivergenceFactory] = {
    "chi_square": _chi_square,
    "kl": _kl,
    "phi": _phi,
    "sinkhorn": _sinkhorn,
    "total_variation": _total_variation,
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> DivergenceFactory:
    """Build one divergence of each kind from a dtype."""
    return FACTORIES[str(request.param)]


def _distributions(
    dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a candidate and a reference distribution on `device`."""
    return (
        torch.tensor(CANDIDATE, dtype=dtype, device=device),
        torch.tensor(REFERENCE, dtype=dtype, device=device),
    )


def test_output_preserves_input_dtype_and_device(
    factory: DivergenceFactory, dtype: torch.dtype
) -> None:
    device = torch.device("cpu")
    divergence = factory(dtype)
    p, q = _distributions(dtype, device)

    value = divergence(p, q)

    assert value.dtype == dtype
    assert value.device == p.device


@requires_cuda
def test_buffers_follow_the_module_to_cuda(
    factory: DivergenceFactory, dtype: torch.dtype
) -> None:
    # Divergences are nn.Modules precisely so tensor state such as
    # SinkhornDivergence's ground cost moves with a single .to(device) call.
    divergence = factory(dtype).to(torch.device("cuda"))

    for name, buffer in divergence.named_buffers():
        assert buffer.device.type == "cuda", name
        assert buffer.dtype == dtype, name


@requires_cuda
def test_cuda_output_matches_cpu(
    factory: DivergenceFactory, dtype: torch.dtype
) -> None:
    cpu_value = factory(dtype)(*_distributions(dtype, torch.device("cpu")))
    device = torch.device("cuda")
    divergence = factory(dtype).to(device)
    p, q = _distributions(dtype, device)

    value = divergence(p, q)

    assert value.dtype == dtype
    assert value.device.type == "cuda"
    torch.testing.assert_close(value.cpu(), cpu_value)
