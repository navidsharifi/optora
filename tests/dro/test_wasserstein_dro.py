"""Tests for `WassersteinAmbiguitySet`."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.wasserstein import SinkhornDivergence
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet, _bisection_steps

TWO_POINT_COST = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)

FOUR_POINT_COST = torch.tensor(
    [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ],
    dtype=torch.float64,
)


def _grid_search_wasserstein_dual_minimum(
    nominal: torch.Tensor, loss: torch.Tensor, cost: torch.Tensor, radius: float
) -> torch.Tensor:
    """Independent fine-grid cross-check of the Wasserstein-DRO dual minimum."""
    gamma_grid = torch.linspace(0.0, 10.0, steps=400_001, dtype=nominal.dtype)
    shifted = loss.view(1, 1, -1) - gamma_grid.view(-1, 1, 1) * cost.view(
        1, *cost.shape
    )
    row_max = torch.amax(shifted, dim=-1)
    dual_values = gamma_grid * radius + torch.sum(nominal.view(1, -1) * row_max, dim=-1)
    return dual_values.min()


# --- Core dual solve behavior ----------------------------------------------


def test_zero_radius_returns_exact_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])
    loss = torch.tensor([1.0, 2.0, 3.0])
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
    )
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(nominal * loss), atol=1e-6)


@pytest.mark.parametrize("radius", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
def test_worst_case_expectation_matches_grid_search_over_dual_variable(
    radius: float,
) -> None:
    # The dual is piecewise linear in gamma, so a fixed-step gradient method
    # oscillates around its kink; bisecting the monotone derivative instead
    # lands on the minimum to machine precision, not to a solver tolerance.
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=radius
    )

    result = ambiguity_set.worst_case_expectation(loss)

    reference = _grid_search_wasserstein_dual_minimum(
        nominal, loss, FOUR_POINT_COST, radius
    )
    assert torch.allclose(result, reference, atol=1e-12)


def test_dual_bracket_contains_the_minimizer() -> None:
    # The bisection is only exact because the bracket is: the dual must still
    # be decreasing at gamma = 0 and no longer decreasing at the upper bound.
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=FOUR_POINT_COST, radius=0.2)

    upper = ambiguity_set._dual_upper_bound(loss)
    lower_derivative = ambiguity_set._dual_derivative(loss, torch.zeros_like(upper))
    upper_derivative = ambiguity_set._dual_derivative(loss, upper)

    assert lower_derivative <= 0.0
    assert upper_derivative >= 0.0
    lower, bracketed_upper = ambiguity_set._dual_bracket(loss, torch.Size())
    assert 0.0 <= lower <= bracketed_upper <= upper
    assert ambiguity_set._dual_derivative(loss, lower) <= 0.0
    assert ambiguity_set._dual_derivative(loss, bracketed_upper) >= 0.0


def test_dual_solve_costs_a_fixed_number_of_derivative_evaluations() -> None:
    # The whole point of the bisection: bounded, dtype-determined work,
    # instead of a fixed-step gradient method burning its iteration budget.
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=FOUR_POINT_COST, radius=0.2)
    evaluations = 0
    derivative = ambiguity_set._dual_derivative

    def counting_derivative(loss: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        nonlocal evaluations
        evaluations += 1
        return derivative(loss, gamma)

    ambiguity_set._dual_derivative = counting_derivative  # type: ignore[method-assign]
    ambiguity_set.worst_case_expectation(loss)

    assert evaluations == _bisection_steps(torch.float64)
    assert evaluations < 60


def test_bisection_step_count_tracks_the_working_precision() -> None:
    assert _bisection_steps(torch.float32) < _bisection_steps(torch.float64)
    for dtype in (torch.float32, torch.float64):
        assert 2.0 ** -_bisection_steps(dtype) < torch.finfo(dtype).eps


def test_worst_case_expectation_does_not_synchronize_with_the_host(
    host_sync_counter: Callable[[], Any],
) -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=FOUR_POINT_COST, radius=0.2)

    with host_sync_counter() as syncs:
        ambiguity_set.worst_case_expectation(loss)

    assert len(syncs) == 0


def test_large_radius_puts_the_dual_minimizer_exactly_at_zero() -> None:
    # Once the radius covers the cost of moving all mass onto the highest-loss
    # point the constraint gamma >= 0 is active, and the bracket's left end is
    # the answer: the worst case is the maximum loss itself.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=TWO_POINT_COST, radius=1.5)

    lower, _ = ambiguity_set._dual_bracket(loss, torch.Size())

    assert torch.equal(lower, torch.zeros_like(lower))
    assert torch.equal(ambiguity_set.worst_case_expectation(loss), loss.max())


def test_worst_case_distribution_spends_the_radius_exactly() -> None:
    # The dual minimizer is a kink: the transport plans on either side of it
    # over- and under-spend the radius, and only their mixture is optimal.
    # Reading the value off the dual there instead would break the gradient.
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    outcomes = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
    cost = (outcomes.unsqueeze(0) - outcomes.unsqueeze(1)) ** 2
    radius = 0.02
    decision = torch.tensor(3.98, dtype=torch.float64, requires_grad=True)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    value = ambiguity_set.worst_case_expectation((outcomes - decision) ** 2)
    (gradient,) = torch.autograd.grad(value, decision)

    # The gradient is the one of the true worst-case distribution, so it must
    # agree with a central difference of the (exact) value function.
    step = 1e-5
    shifted = torch.tensor([decision.item() - step, decision.item() + step])
    losses = (outcomes - shifted.to(torch.float64).unsqueeze(-1)) ** 2
    values = ambiguity_set.worst_case_expectation(losses)
    assert torch.allclose(gradient, (values[1] - values[0]) / (2.0 * step), atol=1e-6)


def test_worst_case_expectation_is_differentiable_through_the_dual_optimum() -> None:
    # The bisection is detached, so the gradient comes entirely from
    # re-evaluating the dual at the minimizer (envelope theorem). Comparing
    # against a difference quotient in a direction that keeps the same
    # maximizers avoids the kinks where the value is only subdifferentiable.
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    direction = torch.ones_like(loss)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=FOUR_POINT_COST, radius=0.2)
    variable = loss.clone().requires_grad_(True)

    value = ambiguity_set.worst_case_expectation(variable)
    (gradient,) = torch.autograd.grad(value, variable)

    # Shifting every loss by a constant shifts the worst case by the same
    # constant, so the directional derivative along `direction` must be one.
    assert torch.allclose(
        torch.sum(gradient * direction), torch.tensor(1.0, dtype=torch.float64)
    )
    step = 1e-6
    shifted = ambiguity_set.worst_case_expectation(loss + step * direction)
    assert torch.allclose(
        shifted - value.detach(),
        torch.tensor(step, dtype=torch.float64),
        atol=1e-12,
    )


# --- Hand-computed closed-form examples -------------------------------------


def test_matches_hand_computed_two_point_example_below_threshold() -> None:
    # nominal=[0.5,0.5], loss=[0,1], cost=[[0,2],[2,0]]: moving mass m at cost
    # 2*m per unit reallocated from point 0 to point 1 under budget 0.3 gives
    # m = 0.15, so Q = [0.35, 0.65] and E_Q[loss] = 0.65 (verified by hand and
    # against a fine grid search over gamma).
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=TWO_POINT_COST, radius=0.3)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(0.65, dtype=torch.float64), atol=1e-12)


def test_matches_hand_computed_two_point_example_at_threshold() -> None:
    # The budget needed to move all mass to the max-loss point is
    # sum_i p_i * cost_{i, argmax(loss)} = 0.5 * 2 + 0.5 * 0 = 1.0; at exactly
    # that radius the dual optimum sits at gamma = 0 and the worst case is the
    # max loss.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=TWO_POINT_COST, radius=1.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_worst_case_expectation_is_attained_by_explicit_transport_plan() -> None:
    # The coupling pi = [[0.35, 0.15], [0.0, 0.5]] has row marginal
    # nominal=[0.5,0.5], column marginal Q=[0.35,0.65], and exact transport
    # cost sum(pi * cost) = 0.15 * 2 = 0.3 = radius, so Q lies exactly on the
    # boundary of the ambiguity set and attains the worst-case expectation.
    nominal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    candidate = torch.tensor([0.35, 0.65], dtype=torch.float64)
    coupling = torch.tensor([[0.35, 0.15], [0.0, 0.5]], dtype=torch.float64)

    assert torch.allclose(coupling.sum(dim=1), nominal, atol=1e-9)
    assert torch.allclose(coupling.sum(dim=0), candidate, atol=1e-9)
    exact_transport_cost = torch.sum(coupling * TWO_POINT_COST)
    assert exact_transport_cost <= 0.3 + 1e-9

    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=TWO_POINT_COST, radius=0.3)
    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.sum(candidate * loss), atol=1e-12)


def test_worst_case_expectation_reaches_max_loss_for_large_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=FOUR_POINT_COST, radius=2.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-12)


def test_single_support_point_has_no_transport_and_no_bracket() -> None:
    nominal = torch.tensor([1.0], dtype=torch.float64)
    loss = torch.tensor([3.0], dtype=torch.float64)
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=torch.zeros(1, 1, dtype=torch.float64), radius=0.5
    )

    assert torch.equal(ambiguity_set.worst_case_expectation(loss), loss[0])


# --- General bounds and monotonicity ----------------------------------------


def test_worst_case_expectation_is_monotonic_in_radius() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)

    small = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=0.05
    ).worst_case_expectation(loss)
    large = WassersteinAmbiguitySet(
        nominal, cost=FOUR_POINT_COST, radius=0.5
    ).worst_case_expectation(loss)

    assert large >= small - 1e-6


def test_worst_case_expectation_is_at_least_nominal_expectation() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
    )
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.1)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result >= torch.sum(nominal * loss) - 1e-12


def test_worst_case_expectation_does_not_exceed_max_loss() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64
    )
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.5)

    result = ambiguity_set.worst_case_expectation(loss)

    assert result <= loss.max() + 1e-12


# --- Lipschitz-regularization equivalence -----------------------------------


def _metric_support(num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return midpoint normal quantiles and their Euclidean cost matrix."""
    quantiles = (torch.arange(num_points, dtype=torch.float64) + 0.5) / num_points
    support = torch.special.ndtri(quantiles)
    return support, torch.abs(support.unsqueeze(-1) - support.unsqueeze(-2))


def _empirical_lipschitz_constant(
    loss: torch.Tensor, cost: torch.Tensor
) -> torch.Tensor:
    """Return the largest loss increment per unit of transport on the support."""
    increments = torch.abs(loss.unsqueeze(-1) - loss.unsqueeze(-2))
    separation = cost + torch.eye(cost.shape[-1], dtype=cost.dtype)
    return torch.max(increments / separation)


def test_small_radius_dual_equals_lipschitz_regularized_expectation() -> None:
    # On a metric support the dual is minimized at gamma = Lip(loss) for every
    # radius below the threshold at which some support point's maximizer stops
    # being itself, so the dual value is exactly the linear surrogate
    # E_nominal[loss] + radius * Lip(loss) (Wu, Li and Mao 2025). Here
    # loss(z) = |z| attains its Lipschitz constant 1 between adjacent support
    # points, and the threshold radius is 0.8.
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    cost = torch.abs(support.unsqueeze(-1) - support.unsqueeze(-2))
    loss = torch.abs(support)
    nominal = torch.full_like(support, 0.2)
    radius = 0.2
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(
        _empirical_lipschitz_constant(loss, cost),
        torch.tensor(1.0, dtype=torch.float64),
    )
    expected = torch.sum(nominal * loss) + radius * 1.0
    assert torch.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("radius", [0.01, 0.1, 1.0, 5.0])
def test_lipschitz_surrogate_upper_bounds_the_dual(radius: float) -> None:
    support, cost = _metric_support(24)
    loss = torch.nn.functional.softplus(support)
    nominal = torch.full_like(support, 1.0 / support.numel())
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    # softplus is 1-Lipschitz: its derivative is the sigmoid.
    surrogate = torch.sum(nominal * loss) + radius * 1.0
    assert result <= surrogate + 1e-6


def test_lipschitz_surrogate_gap_shrinks_as_the_sample_refines() -> None:
    # The small-radius gap against the surrogate built from the true Lipschitz
    # constant is radius * (Lip - L_n), where L_n is the loss's Lipschitz
    # modulus restricted to the sample. softplus approaches slope 1 only
    # asymptotically, so L_n increases -- and the gap shrinks -- as the sample
    # refines.
    radius = 1e-3
    normalized_gaps = []
    deficiencies = []
    for num_points in (8, 32, 128):
        support, cost = _metric_support(num_points)
        loss = torch.nn.functional.softplus(support)
        nominal = torch.full_like(support, 1.0 / num_points)
        ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

        result = ambiguity_set.worst_case_expectation(loss)

        surrogate = torch.sum(nominal * loss) + radius * 1.0
        normalized_gaps.append(((surrogate - result) / radius).item())
        deficiencies.append((1.0 - _empirical_lipschitz_constant(loss, cost)).item())

    assert normalized_gaps == sorted(normalized_gaps, reverse=True)
    for measured, predicted in zip(normalized_gaps, deficiencies, strict=True):
        assert abs(measured - predicted) < 1e-9


# --- Validation --------------------------------------------------------------


def test_mismatched_loss_shape_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])
    loss = torch.tensor([1.0, 2.0, 3.0])
    ambiguity_set = WassersteinAmbiguitySet(
        nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=0.1
    )

    with pytest.raises(ValueError):
        ambiguity_set.worst_case_expectation(loss)


def test_negative_radius_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(
            nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=-1.0
        )


def test_cost_size_mismatched_with_nominal_raises_value_error() -> None:
    nominal = torch.tensor([0.3, 0.3, 0.4])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(
            nominal, cost=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), radius=0.1
        )


def test_non_square_cost_raises_value_error() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError):
        WassersteinAmbiguitySet(nominal, cost=torch.zeros(2, 3), radius=0.1)


def test_negative_cost_raises_value_error_when_validation_is_requested() -> None:
    nominal = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError, match="cost must be nonnegative"):
        WassersteinAmbiguitySet(
            nominal,
            cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]),
            radius=0.1,
            validate=True,
        )


def test_negative_cost_is_not_inspected_by_default() -> None:
    nominal = torch.tensor([0.5, 0.5])

    WassersteinAmbiguitySet(
        nominal, cost=torch.tensor([[0.0, -1.0], [-1.0, 0.0]]), radius=0.1
    )


def test_construction_does_not_synchronize_by_default(
    host_sync_counter: Callable[[], Any],
) -> None:
    nominal = torch.tensor([0.5, 0.5])

    with host_sync_counter() as syncs:
        WassersteinAmbiguitySet(nominal, cost=TWO_POINT_COST, radius=0.1)

    assert len(syncs) == 0


def test_sinkhorn_parameters_are_passed_through_to_divergence() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(
        nominal,
        cost=cost,
        radius=0.1,
        epsilon=0.02,
        sinkhorn_max_iter=50,
        sinkhorn_tol=1e-4,
        eps=1e-10,
    )

    assert isinstance(ambiguity_set.divergence, SinkhornDivergence)
    assert ambiguity_set.divergence.epsilon == 0.02
    assert ambiguity_set.divergence.max_iter == 50
    assert ambiguity_set.divergence.tol == 1e-4
    assert ambiguity_set.divergence.eps == 1e-10


def test_cost_is_not_duplicated_as_a_separate_buffer() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.1)

    buffer_names = [name for name, _ in ambiguity_set.named_buffers()]

    assert buffer_names.count("divergence.cost") == 1
    assert "cost" not in buffer_names
    assert ambiguity_set.cost.data_ptr() == ambiguity_set.divergence.cost.data_ptr()


def test_cost_stays_aliased_with_divergence_cost_after_to() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.1)

    ambiguity_set = ambiguity_set.to(torch.float64)

    assert ambiguity_set.cost.dtype == torch.float64
    assert ambiguity_set.cost.data_ptr() == ambiguity_set.divergence.cost.data_ptr()


# --- Containment and type checks ---------------------------------------------


def test_contains_uses_sinkhorn_divergence_and_radius() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=0.05)

    assert ambiguity_set.contains(nominal)
    assert not ambiguity_set.contains(torch.tensor([0.95, 0.05]))


def test_is_an_ambiguity_set_instance() -> None:
    nominal = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert isinstance(
        WassersteinAmbiguitySet(nominal, cost=cost, radius=0.1), AmbiguitySet
    )
