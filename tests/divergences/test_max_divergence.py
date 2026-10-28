"""Tests for `MaxDivergence`."""

import math

import pytest
import torch

from optora.core.divergence_base import Divergence
from optora.divergences.max_divergence import MaxDivergence


def test_zero_for_identical_distributions() -> None:
    p = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)

    assert float(MaxDivergence()(p, p)) == 0.0


def test_is_the_log_of_the_largest_likelihood_ratio() -> None:
    p = torch.tensor([0.7, 0.3], dtype=torch.float64)
    q = torch.tensor([0.4, 0.6], dtype=torch.float64)

    value = MaxDivergence()(p, q)

    assert float(value) == pytest.approx(math.log(0.7 / 0.4), abs=1e-15)


def test_is_asymmetric() -> None:
    p = torch.tensor([0.1, 0.9], dtype=torch.float64)
    q = torch.tensor([0.3, 0.7], dtype=torch.float64)

    assert float(MaxDivergence()(p, q)) == pytest.approx(math.log(0.9 / 0.7))
    assert float(MaxDivergence()(q, p)) == pytest.approx(math.log(3.0))


def test_dominates_the_kl_divergence() -> None:
    # Renyi divergences are nondecreasing in their order, so D_inf >= D_KL.
    generator = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(5, 16, generator=generator), dim=-1).double()
    q = torch.softmax(torch.randn(5, 16, generator=generator), dim=-1).double()

    for p_row, q_row in zip(p, q, strict=True):
        kl = float(torch.sum(p_row * torch.log(p_row / q_row)))
        assert float(MaxDivergence()(p_row, q_row)) >= kl


def test_zero_candidate_entries_are_ignored() -> None:
    p = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    q = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)

    assert float(MaxDivergence()(p, q)) == pytest.approx(math.log(2.0))


def test_leaving_the_reference_support_is_large_but_finite() -> None:
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0], dtype=torch.float64)

    value = MaxDivergence(eps=1e-12)(p, q)

    assert torch.isfinite(value)
    assert float(value) == pytest.approx(math.log(0.5 / 1e-12))


def test_rounding_below_zero_is_clamped() -> None:
    p = torch.tensor([0.3, 0.7], dtype=torch.float64)
    q = p * (1.0 + 1e-16)

    assert float(MaxDivergence()(p, q)) == 0.0


def test_gradient_flows_to_the_candidate() -> None:
    p = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)

    MaxDivergence()(p, q).backward()

    assert p.grad is not None
    torch.testing.assert_close(
        p.grad, torch.tensor([0.0, 1.0 / 0.8], dtype=torch.float64)
    )


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_invalid_eps_raises_value_error(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        MaxDivergence(eps=eps)


def test_is_a_divergence_instance() -> None:
    assert isinstance(MaxDivergence(), Divergence)
