"""Performance guard for the divergences in `optora.divergences`.

The closed-form divergences never synchronize. The Sinkhorn divergence runs
its three debiasing transport problems as one batched loop to `max_iter`,
paying one host synchronization per `check_interval` iterations. The entropic
transport divergence pays the same for its Sinkhorn phase, then one per
iteration of its damped Newton phase.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.divergences.entropic_transport import EntropicTransportDivergence
from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    TotalVariationDivergence,
)
from optora.divergences.kl import KLDivergence
from optora.divergences.max_divergence import MaxDivergence
from optora.divergences.wasserstein import SinkhornDivergence

# Small enough that both Sinkhorn loops are still far from converged after
# `max_iter` iterations (potentials moving by ~1e-4), so they always run to
# `max_iter`. At 0.1 they reach the float32 noise floor early, and whether
# rounding then settles on an exact fixed point, ending the loop a checkpoint
# early, depends on the platform.
SINKHORN_EPSILON = 0.02


def _distribution(generator: torch.Generator, *shape: int) -> torch.Tensor:
    weights = torch.rand(*shape, generator=generator) + 0.5
    return weights / weights.sum(dim=-1, keepdim=True)


def test_closed_form_divergences(guarded_benchmark: Callable[..., Any]) -> None:
    support = 4096
    generator = torch.Generator().manual_seed(0)
    p, q = _distribution(generator, support), _distribution(generator, support)
    divergences = (
        KLDivergence(),
        ChiSquareDivergence(),
        TotalVariationDivergence(),
        MaxDivergence(),
    )

    values = guarded_benchmark(
        lambda: [divergence(p, q) for divergence in divergences], host_syncs=0
    )

    assert all(value.shape == () for value in values)


def test_sinkhorn_divergence(guarded_benchmark: Callable[..., Any]) -> None:
    batch, support, max_iter, check_interval = 4, 128, 200, 50
    generator = torch.Generator().manual_seed(0)
    p, q = _distribution(generator, batch, support), _distribution(generator, support)
    points = torch.linspace(0.0, 1.0, support)
    divergence = SinkhornDivergence(
        cost=torch.abs(points[:, None] - points[None, :]),
        epsilon=SINKHORN_EPSILON,
        max_iter=max_iter,
        tol=1e-30,
        check_interval=check_interval,
        newton_max_iter=0,
    )

    value = guarded_benchmark(
        lambda: divergence(p, q), host_syncs=max_iter // check_interval
    )

    assert value.shape == (batch,)


# Far fewer Newton iterations than the solve needs to converge, so the phase
# always runs to its budget; zero benchmarks the Sinkhorn phase alone.
@pytest.mark.parametrize("newton_max_iter", [0, 3], ids=["sinkhorn", "newton"])
def test_entropic_transport_divergence(
    guarded_benchmark: Callable[..., Any], newton_max_iter: int
) -> None:
    batch, support, max_iter, check_interval = 4, 128, 200, 50
    generator = torch.Generator().manual_seed(0)
    p, q = _distribution(generator, batch, support), _distribution(generator, support)
    points = torch.linspace(0.0, 1.0, support)
    divergence = EntropicTransportDivergence(
        cost=torch.abs(points[:, None] - points[None, :]),
        epsilon=SINKHORN_EPSILON,
        max_iter=max_iter,
        newton_max_iter=newton_max_iter,
        tol=1e-30,
        check_interval=check_interval,
    )

    value = guarded_benchmark(
        lambda: divergence(p, q),
        host_syncs=max_iter // check_interval + newton_max_iter,
    )

    assert value.shape == (batch,)
