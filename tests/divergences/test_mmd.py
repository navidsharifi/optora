"""Tests for `MMDDivergence`."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.autograd import gradcheck

from optora.core.divergence_base import Divergence
from optora.divergences.mmd import MMDDivergence

POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.5, 1.5]])
GRAM = torch.exp(-torch.cdist(POINTS, POINTS)).to(torch.float64)
P = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
Q = torch.tensor([0.4, 0.0, 0.5, 0.1], dtype=torch.float64)


def test_is_the_rkhs_distance_of_the_mean_embeddings() -> None:
    difference = P - Q

    value = MMDDivergence(GRAM)(P, Q)

    assert float(value) == pytest.approx(
        math.sqrt(float(difference @ GRAM @ difference)), rel=1e-15
    )


def test_is_zero_for_identical_distributions_and_symmetric() -> None:
    divergence = MMDDivergence(GRAM)

    assert float(divergence(P, P)) == 0.0
    assert float(divergence(P, Q)) == float(divergence(Q, P))


def test_satisfies_the_triangle_inequality() -> None:
    generator = torch.Generator().manual_seed(0)
    divergence = MMDDivergence(GRAM)
    a, b, c = torch.softmax(
        torch.randn(3, 64, 4, generator=generator, dtype=torch.float64), dim=-1
    )

    assert bool(torch.all(divergence(a, c) <= divergence(a, b) + divergence(b, c)))


def test_is_finite_between_distributions_with_disjoint_supports() -> None:
    left = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    right = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    value = MMDDivergence(GRAM)(left, right)

    assert math.isfinite(float(value))
    assert float(value) == pytest.approx(
        math.sqrt(float(GRAM[0, 0] + GRAM[3, 3] - 2.0 * GRAM[0, 3]))
    )


def test_batches_and_broadcasts() -> None:
    divergence = MMDDivergence(GRAM)
    candidates = torch.stack([P, Q, 0.5 * (P + Q)])

    values = divergence(candidates, Q)

    # A batched quadratic form runs as a matrix product and a single one as a
    # matrix-vector product, whose BLAS reduction orders may differ by an ulp.
    assert values.shape == (3,)
    for index in range(3):
        assert float(values[index]) == pytest.approx(
            float(divergence(candidates[index], Q)), rel=1e-15
        )


def test_rounding_below_zero_is_clamped_on_a_singular_gram() -> None:
    # Duplicate points make the Gram matrix singular; a difference in its
    # null space has an exactly zero discrepancy that rounds either way.
    duplicate = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    gram = torch.exp(-torch.cdist(duplicate, duplicate))
    left = torch.tensor([0.7, 0.1, 0.2], dtype=torch.float64)
    right = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)

    value = MMDDivergence(gram)(left, right)

    assert float(value) == pytest.approx(0.0, abs=1e-7)
    assert not math.isnan(float(value))


def test_gradient_matches_finite_differences_away_from_zero() -> None:
    candidate = P.clone().requires_grad_(True)
    reference = Q.clone().requires_grad_(True)

    assert gradcheck(MMDDivergence(GRAM), (candidate, reference))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_are_computed_in_float32(dtype: torch.dtype) -> None:
    divergence = MMDDivergence(GRAM.to(dtype))

    value = divergence(P.to(dtype), Q.to(dtype))

    assert value.dtype == dtype
    assert float(value) == pytest.approx(float(MMDDivergence(GRAM)(P, Q)), rel=1e-2)


def test_rejects_a_non_square_gram() -> None:
    with pytest.raises(ValueError, match="square"):
        MMDDivergence(torch.zeros(3, 4))


def test_rejects_mismatched_distribution_sizes() -> None:
    divergence = MMDDivergence(GRAM)

    with pytest.raises(ValueError, match="p must have shape"):
        divergence(torch.ones(3) / 3, Q)
    with pytest.raises(ValueError, match="q must have shape"):
        divergence(P, torch.ones(3) / 3)


@pytest.mark.parametrize(
    ("gram", "match"),
    [
        (torch.tensor([[1.0, 0.5], [0.4, 1.0]]), "symmetric"),
        (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
        (torch.tensor([[1.0, math.nan], [math.nan, 1.0]]), "finite"),
    ],
    ids=["asymmetric", "indefinite", "nan"],
)
def test_validate_rejects_an_invalid_gram(gram: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MMDDivergence(gram, validate=True)
    MMDDivergence(gram)


def test_construction_does_not_synchronize_by_default(
    host_sync_counter: Callable[[], Any],
) -> None:
    with host_sync_counter() as syncs:
        MMDDivergence(GRAM)

    assert len(syncs) == 0


def test_is_a_divergence_instance() -> None:
    assert isinstance(MMDDivergence(GRAM), Divergence)
