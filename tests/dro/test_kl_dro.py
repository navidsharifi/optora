"""Tests for `KLAmbiguitySet`."""

import pytest
import torch

from optora.core.dro_base import AmbiguitySet, TiltedAmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet

NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
LOSS = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)


def _grid_search_dual_minimum(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float
) -> torch.Tensor:
    """Cross-check the primal tilt against a fine grid over the KL-DRO dual.

    The ambiguity set is now solved in the primal, so the convex dual
    `inf_eta eta * radius + eta * log E[exp(loss / eta)]` is an independent
    route to the same number: strong duality makes the two equal, and a
    bug in the tilt path would not move the dual's grid minimum.

    Args:
        nominal: Reference distribution of shape `(n,)`.
        loss: Per-scenario losses of shape `(n,)`.
        radius: Nonnegative KL budget in nats.

    Returns:
        The smallest dual value over a fine logarithmic grid of `eta`.
    """
    log_nominal = torch.log(nominal)
    eta_grid = torch.logspace(-4, 4, steps=200001, dtype=nominal.dtype)
    log_mgf = torch.logsumexp(
        log_nominal.unsqueeze(0) + loss.unsqueeze(0) / eta_grid.unsqueeze(1),
        dim=-1,
    )
    return torch.min(eta_grid * radius + eta_grid * log_mgf)


def _kl_saturation_radius(nominal: torch.Tensor, loss: torch.Tensor) -> float:
    """Return `-log P*`, the radius from which the worst case is `max(loss)`."""
    top_mass = nominal[loss >= loss.max()].sum()
    return float(-torch.log(top_mass))


def test_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


def test_worst_case_expectation_matches_grid_search_over_the_dual() -> None:
    radius = 0.2
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=radius)

    result = ambiguity_set.worst_case_expectation(LOSS)

    assert torch.allclose(
        result, _grid_search_dual_minimum(NOMINAL, LOSS, radius), atol=1e-6
    )


def test_worst_case_expectation_reaches_max_loss_for_large_radius() -> None:
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=2.0)

    result = ambiguity_set.worst_case_expectation(LOSS)

    assert result == LOSS.max()


def test_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    radii = torch.logspace(-6, 1, steps=200, dtype=torch.float64)

    values = KLAmbiguitySet(nominal, radius=radii).worst_case_expectation(loss)

    assert torch.all(torch.diff(values) >= 0.0)


def test_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss)


def test_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.5)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max()


def test_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=-1.0)


def test_invalid_eps_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=0.1, eps=0.0)
    with pytest.raises(ValueError):
        KLAmbiguitySet(nominal, radius=0.1, eps=-1.0)


def test_contains_uses_kl_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])

    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1)

    assert isinstance(ambiguity_set, AmbiguitySet)
    assert isinstance(ambiguity_set, TiltedAmbiguitySet)


def test_log_nominal_is_cached_as_a_clamped_non_persistent_buffer() -> None:
    nominal = torch.tensor([0.5, 0.5, 0.0])
    ambiguity_set = KLAmbiguitySet(nominal, radius=0.1, eps=1e-10)

    log_nominal = dict(ambiguity_set.named_buffers())["log_nominal"]

    expected = torch.log(torch.tensor([0.5, 0.5, 1e-10]))
    assert torch.allclose(log_nominal, expected)
    assert "log_nominal" not in ambiguity_set.state_dict()


def test_log_nominal_buffer_moves_with_the_module() -> None:
    ambiguity_set = KLAmbiguitySet(torch.tensor([0.5, 0.5]), radius=0.1)

    ambiguity_set = ambiguity_set.to(dtype=torch.float64)

    assert ambiguity_set.log_nominal.dtype == torch.float64


# --- The tilt path -----------------------------------------------------------


def test_repeated_calls_are_deterministic() -> None:
    # The bisection carries no state between calls, so there is no warm
    # start that could make a repeat solve differ from the first one.
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=0.2)

    first = ambiguity_set.worst_case_expectation(LOSS)
    second = ambiguity_set.worst_case_expectation(LOSS)

    assert first == second


def test_the_tilt_is_an_exponential_reweighting_of_the_nominal() -> None:
    # q ∝ nominal * exp(beta * loss) means the log-ratio log(q / nominal) is
    # affine in the loss, which pins the whole shape of the maximizer.
    radius = 0.4
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=radius)
    differentiable = LOSS.clone().requires_grad_(True)

    ambiguity_set.worst_case_expectation(differentiable).backward()

    assert differentiable.grad is not None
    log_ratio = torch.log(differentiable.grad / NOMINAL)
    slopes = torch.diff(log_ratio) / torch.diff(LOSS)
    assert torch.allclose(slopes, slopes[0].expand_as(slopes), atol=1e-9)


@pytest.mark.parametrize("fraction", [1e-12, 1e-6, 0.01, 0.5, 0.9, 0.999])
def test_the_returned_distribution_stays_inside_the_radius(fraction: float) -> None:
    # Bisection keeps the feasible endpoint, so the reported value is always
    # a primal objective value at a distribution the set actually contains.
    # Below saturation the radius constraint is tight, so the slack is
    # checked relatively: the divergence may sit on the boundary but must
    # never cross it by more than the rounding of its own evaluation.
    radius = fraction * _kl_saturation_radius(NOMINAL, LOSS)
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=radius)
    differentiable = LOSS.clone().requires_grad_(True)

    ambiguity_set.worst_case_expectation(differentiable).backward()

    assert differentiable.grad is not None
    divergence = ambiguity_set.divergence(differentiable.grad, NOMINAL)
    assert float(divergence) <= radius * (1.0 + 1e-9)


@pytest.mark.parametrize("support_size", [1, 2, 100, 1000, 10000])
def test_a_large_support_stays_finite_and_bounded(support_size: int) -> None:
    nominal = torch.full((support_size,), 1.0 / support_size, dtype=torch.float64)
    loss = torch.linspace(0.0, 1.0, support_size, dtype=torch.float64)
    radii = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64) * max(
        _kl_saturation_radius(nominal, loss), 1e-12
    )

    values = KLAmbiguitySet(nominal, radius=radii).worst_case_expectation(loss)

    assert torch.all(torch.isfinite(values))
    assert torch.all(values <= loss.max())
    assert torch.all(values >= torch.sum(nominal * loss) - 1e-12)


@pytest.mark.parametrize("radius", [1e-14, 1e-10, 1e-6])
def test_a_vanishing_radius_matches_the_quadratic_expansion(radius: float) -> None:
    # Issue #49. For a small radius the worst case expands as
    # E_p[loss] + sqrt(2 * radius * Var_p(loss)) + O(radius), so the excess
    # over the nominal expectation must track sqrt(radius) to high relative
    # accuracy rather than being swamped by a solver's stopping gap.
    nominal = torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
    nominal_expectation = torch.sum(nominal * loss)
    variance = torch.sum(nominal * (loss - nominal_expectation) ** 2)

    result = KLAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)

    excess = float(result - nominal_expectation)
    expected = float(torch.sqrt(2.0 * radius * variance))
    assert abs(excess - expected) <= 1e-3 * expected
