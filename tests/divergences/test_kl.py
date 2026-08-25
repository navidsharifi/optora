"""Tests for `KLDivergence`."""

import pytest
import torch

from optora.divergences.kl import KLDivergence


def test_zero_for_identical_distributions() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.25, 0.25, 0.5])

    value = divergence(p, p)

    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


def test_matches_closed_form_for_two_point_distributions() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.4, 0.6])

    value = divergence(p, q)

    expected = 0.7 * torch.log(torch.tensor(0.7 / 0.4)) + 0.3 * torch.log(
        torch.tensor(0.3 / 0.6)
    )
    assert torch.allclose(value, expected, atol=1e-6)


def test_nonnegative_for_distinct_distributions() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.9, 0.1])

    value = divergence(p, q)

    assert value >= 0.0


def test_asymmetric() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.1, 0.9])
    q = torch.tensor([0.3, 0.7])

    assert not torch.allclose(divergence(p, q), divergence(q, p))


def test_handles_zero_entries_without_producing_nan_or_inf() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)

    assert torch.isfinite(value)


def test_reference_zero_entry_does_not_diverge_to_inf_due_to_clamping() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])

    value = divergence(p, q)

    assert torch.isfinite(value)
    assert value > 0.0


def test_invalid_eps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        KLDivergence(eps=0.0)
    with pytest.raises(ValueError):
        KLDivergence(eps=-1.0)


def test_is_a_divergence_instance() -> None:
    from optora.core.divergence_base import Divergence

    assert isinstance(KLDivergence(), Divergence)


def test_gradient_is_finite_through_clamped_zero_entries() -> None:
    divergence = KLDivergence()
    p = torch.tensor([0.0, 1.0], requires_grad=True)
    q = torch.tensor([0.5, 0.5])

    value = divergence(p, q)
    value.backward()

    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))
