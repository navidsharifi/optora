"""Tests for `PhiDivergence`, `ChiSquareDivergence`, and `TotalVariationDivergence`."""

import pytest
import torch

from optora.divergences.f_divergence import (
    ChiSquareDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)

DIVERGENCE_FACTORIES = [ChiSquareDivergence, TotalVariationDivergence]


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_zero_for_identical_distributions(divergence_cls: type) -> None:
    divergence = divergence_cls()
    p = torch.tensor([0.25, 0.25, 0.5])

    value = divergence(p, p)

    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_nonnegative_for_distinct_distributions(divergence_cls: type) -> None:
    divergence = divergence_cls()
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.9, 0.1])

    value = divergence(p, q)

    assert value >= 0.0


def test_chi_square_is_asymmetric() -> None:
    divergence = ChiSquareDivergence()
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.3, 0.7])

    assert not torch.allclose(divergence(p, q), divergence(q, p))


def test_total_variation_is_symmetric() -> None:
    divergence = TotalVariationDivergence()
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.3, 0.7])

    assert torch.allclose(divergence(p, q), divergence(q, p))


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_handles_zero_entries_without_producing_nan_or_inf(
    divergence_cls: type,
) -> None:
    divergence = divergence_cls()
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)

    assert torch.isfinite(value)


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_reference_zero_entry_stays_finite_due_to_clamping(
    divergence_cls: type,
) -> None:
    divergence = divergence_cls()
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])

    value = divergence(p, q)

    assert torch.isfinite(value)
    assert value > 0.0


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_invalid_eps_raises_value_error(divergence_cls: type) -> None:
    with pytest.raises(ValueError):
        divergence_cls(eps=0.0)
    with pytest.raises(ValueError):
        divergence_cls(eps=-1.0)


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_is_a_divergence_instance(divergence_cls: type) -> None:
    from optora.core.divergence_base import Divergence

    assert isinstance(divergence_cls(), Divergence)


@pytest.mark.parametrize("divergence_cls", DIVERGENCE_FACTORIES)
def test_gradient_is_finite_through_clamped_zero_entries(
    divergence_cls: type,
) -> None:
    divergence = divergence_cls()
    p = torch.tensor([0.0, 1.0], requires_grad=True)
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)
    value.backward()

    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))


def test_chi_square_matches_pearson_closed_form() -> None:
    divergence = ChiSquareDivergence()
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.4, 0.6])

    value = divergence(p, q)

    expected = torch.sum((p - q) ** 2 / q)
    assert torch.allclose(value, expected, atol=1e-6)


def test_total_variation_matches_closed_form() -> None:
    divergence = TotalVariationDivergence()
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.4, 0.6])

    value = divergence(p, q)

    expected = 0.5 * torch.sum(torch.abs(p - q))
    assert torch.allclose(value, expected, atol=1e-6)


def test_total_variation_is_bounded_by_one() -> None:
    divergence = TotalVariationDivergence()
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])

    value = divergence(p, q)

    assert torch.allclose(value, torch.tensor(1.0), atol=1e-6)


def test_phi_divergence_matches_manual_computation_for_custom_generator() -> None:
    def cube_generator(ratio: torch.Tensor) -> torch.Tensor:
        return (ratio - 1.0) ** 2 + (ratio - 1.0) ** 4

    divergence = PhiDivergence(phi=cube_generator)
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.4, 0.6])

    value = divergence(p, q)

    ratio = p / q
    expected = torch.sum(q * cube_generator(ratio))
    assert torch.allclose(value, expected, atol=1e-6)


def test_phi_divergence_zero_for_identical_distributions() -> None:
    divergence = PhiDivergence(phi=lambda ratio: (ratio - 1.0) ** 2)
    p = torch.tensor([0.2, 0.3, 0.5])

    value = divergence(p, p)

    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


def test_phi_divergence_invalid_eps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        PhiDivergence(phi=lambda ratio: (ratio - 1.0) ** 2, eps=0.0)
    with pytest.raises(ValueError):
        PhiDivergence(phi=lambda ratio: (ratio - 1.0) ** 2, eps=-1.0)


def test_chi_square_and_total_variation_are_phi_divergence_instances() -> None:
    assert isinstance(ChiSquareDivergence(), PhiDivergence)
    assert isinstance(TotalVariationDivergence(), PhiDivergence)
