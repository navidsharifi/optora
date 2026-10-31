"""Device, dtype, and batching coverage for every divergence.

The half-precision tests back the dtype support table in `docs/numerics.md`:
a divergence of half-precision distributions is exactly the float32 value
rounded once. The batching tests pin the `Divergence` contract: inputs of
shape `(..., n)` that broadcast together give one value per pair, of shape
`(...)`, reducing the last dimension only.
"""

import math
from collections.abc import Callable

import pytest
import torch

from optora.core.divergence_base import Divergence
from optora.divergences.entropic_transport import EntropicTransportDivergence
from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)
from optora.divergences.kl import KLDivergence
from optora.divergences.max_divergence import MaxDivergence
from optora.divergences.mmd import MMDDivergence
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


def _entropic_transport(dtype: torch.dtype) -> Divergence:
    return EntropicTransportDivergence(
        cost=torch.tensor(COST, dtype=dtype), epsilon=0.5
    )


def _max(dtype: torch.dtype) -> Divergence:
    return MaxDivergence()


def _mmd(dtype: torch.dtype) -> Divergence:
    cost = torch.tensor(COST, dtype=dtype)
    return MMDDivergence(gram=torch.exp(-cost))


def _sinkhorn(dtype: torch.dtype) -> Divergence:
    return SinkhornDivergence(cost=torch.tensor(COST, dtype=dtype), epsilon=0.1)


FACTORIES: dict[str, DivergenceFactory] = {
    "chi_square": _chi_square,
    "entropic_transport": _entropic_transport,
    "kl": _kl,
    "max": _max,
    "mmd": _mmd,
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


def test_half_precision_is_the_float32_value_rounded_once(
    factory: DivergenceFactory, half_dtype: torch.dtype
) -> None:
    p, q = _distributions(half_dtype, torch.device("cpu"))

    value = factory(half_dtype)(p, q)

    expected = factory(torch.float32)(p.float(), q.float())
    assert value.dtype == half_dtype
    assert torch.equal(value, expected.to(half_dtype))


def test_mixed_precision_arguments_return_the_promoted_dtype(
    factory: DivergenceFactory, half_dtype: torch.dtype
) -> None:
    p, q = _distributions(torch.float32, torch.device("cpu"))

    value = factory(torch.float32)(p.to(half_dtype), q)

    assert value.dtype == torch.float32


def test_kl_of_a_zero_entry_is_finite_in_float16() -> None:
    # The default eps of 1e-12 underflows to zero in float16, so clamping in
    # float16 would evaluate 0 * log(0) = nan.
    p = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float16)
    q = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float16)

    value = KLDivergence()(p, q)

    assert value.dtype == torch.float16
    assert torch.isfinite(value)
    assert abs(float(value) - 0.5 * math.log(2.0)) <= 1e-3


# --- Batching ----------------------------------------------------------------

BATCH_CANDIDATES = (
    (0.2, 0.5, 0.3),
    (0.1, 0.1, 0.8),
    (0.6, 0.3, 0.1),
    (0.25, 0.5, 0.25),
)

BATCH_REFERENCES = (
    (0.25, 0.5, 0.25),
    (0.3, 0.3, 0.4),
)


# Absolute tolerance of a batched value against the same pair solved alone.
# Closed forms agree to rounding. Sinkhorn's loop stops on the joint marginal
# residual of the whole batch, so its elements agree to its `tol` (1e-9).
BATCH_TOLERANCE = {"sinkhorn": 1e-8}


def _batch(rows: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


@pytest.fixture
def batch_tolerance(request: pytest.FixtureRequest) -> float:
    """Return the batched-vs-single tolerance of the current divergence."""
    name = str(request.node.callspec.params["factory"])
    return BATCH_TOLERANCE.get(name, 1e-14)


def test_a_batch_of_pairs_gives_one_value_per_pair(
    factory: DivergenceFactory, batch_tolerance: float
) -> None:
    divergence = factory(torch.float64)
    p = _batch(BATCH_CANDIDATES)
    q = _batch(BATCH_REFERENCES * 2)

    values = divergence(p, q)

    expected = torch.stack(
        [divergence(row, reference) for row, reference in zip(p, q, strict=True)]
    )
    assert values.shape == (len(BATCH_CANDIDATES),)
    torch.testing.assert_close(values, expected, rtol=0.0, atol=batch_tolerance)


def test_a_single_reference_broadcasts_against_a_batch(
    factory: DivergenceFactory, batch_tolerance: float
) -> None:
    divergence = factory(torch.float64)
    p = _batch(BATCH_CANDIDATES).reshape(2, 2, 3)
    q = _batch(BATCH_REFERENCES[:1])[0]

    values = divergence(p, q)

    expected = torch.stack([divergence(row, q) for row in p.reshape(4, 3)])
    assert values.shape == (2, 2)
    torch.testing.assert_close(
        values.reshape(4), expected, rtol=0.0, atol=batch_tolerance
    )


def test_batched_references_broadcast_against_one_candidate(
    factory: DivergenceFactory, batch_tolerance: float
) -> None:
    divergence = factory(torch.float64)
    p = _batch(BATCH_CANDIDATES)[0]
    q = _batch(BATCH_REFERENCES)

    values = divergence(p, q)

    expected = torch.stack([divergence(p, reference) for reference in q])
    assert values.shape == (len(BATCH_REFERENCES),)
    torch.testing.assert_close(values, expected, rtol=0.0, atol=batch_tolerance)


def test_an_unbatched_pair_gives_a_scalar(factory: DivergenceFactory) -> None:
    p, q = _distributions(torch.float64, torch.device("cpu"))

    assert factory(torch.float64)(p, q).shape == ()
