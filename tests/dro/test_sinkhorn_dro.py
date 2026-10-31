r"""Tests for `SinkhornAmbiguitySet`.

The set is centred on the kernel-smoothed nominal $P^0 = \hat p^\top Q$
rather than on $\hat p$, so it is not in `tests.dro.specs.SPECS`, whose
invariants are stated about the nominal. The same invariants are checked
here about $P^0$, against independent references:

- strong and weak duality against Wang, Gao and Xie's dual evaluated in
  NumPy (`tests.dro.duality.sinkhorn_dual`);
- with a zero cost, exact agreement with `KLAmbiguitySet` of radius
  $\bar\rho/\epsilon$ around the normalized reference;
- as $\epsilon \to 0$, convergence to `WassersteinAmbiguitySet`;
- the closed-form saturation radius
  $\epsilon \sum_i \hat p_i \log(1/Q_i(A))$;
- a gradient that is a distribution inside the ball, measured by the
  Newton solve of `tests.dro.duality.entropic_transport_divergence`, which
  `contains` and its `EntropicTransportDivergence` agree with however
  large $C/\epsilon$ is (issue #69).
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st
from torch.autograd import gradcheck, gradgradcheck

from optora.core.dro_base import AmbiguitySet, TiltedAmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from tests.dro.duality import (
    entropic_transport_divergence,
    sinkhorn_dual,
    sinkhorn_dual_minimum,
    sinkhorn_kernel,
)
from tests.dro.instances import INSTANCES, Instance
from tests.dro.strategies import instances, positive_radii, radii

VALUE_TOLERANCE = 1e-9

epsilons = st.sampled_from((0.05, 0.3, 1.0, 5.0))

ASYMMETRIC = next(instance for instance in INSTANCES if instance.name == "asymmetric")


def tolerance(instance: Instance) -> float:
    """Return the absolute tolerance on a worst-case value for `instance`."""
    return VALUE_TOLERANCE * (1.0 + float(np.abs(instance.loss_array).max()))


def build(
    instance: Instance,
    epsilon: float,
    radius: float | torch.Tensor,
    reference: torch.Tensor | None = None,
) -> SinkhornAmbiguitySet:
    """Build the Sinkhorn set of `instance` with its absolute-distance cost."""
    nominal, _, cost = instance.tensors()
    return SinkhornAmbiguitySet(nominal, cost, epsilon, radius, reference=reference)


def worst_case(
    instance: Instance, epsilon: float, radius: float | torch.Tensor
) -> float:
    """Evaluate the worst case of `instance`'s Sinkhorn set."""
    _, loss, _ = instance.tensors()
    return float(build(instance, epsilon, radius).worst_case_expectation(loss))


def numpy_kernel(instance: Instance, epsilon: float) -> np.ndarray:
    """Return the Gibbs kernel of `instance` under the counting reference."""
    reference = np.ones(len(instance.loss))
    return sinkhorn_kernel(instance.cost_array, epsilon, reference)


def center_expectation(instance: Instance, epsilon: float) -> float:
    """Return the expectation under the kernel-smoothed nominal, in NumPy."""
    kernel = numpy_kernel(instance, epsilon)
    return float(instance.nominal_array @ kernel @ instance.loss_array)


# --- Duality -----------------------------------------------------------------


@given(instance=instances(), epsilon=epsilons, radius=radii)
def test_worst_case_equals_the_dual_minimum(
    instance: Instance, epsilon: float, radius: float
) -> None:
    reference = sinkhorn_dual_minimum(
        instance.nominal_array,
        numpy_kernel(instance, epsilon),
        instance.loss_array,
        radius / epsilon,
    )

    assert worst_case(instance, epsilon, radius) == pytest.approx(
        reference, abs=tolerance(instance)
    )


@given(
    instance=instances(),
    epsilon=epsilons,
    radius=radii,
    log_eta=st.floats(min_value=-8.0, max_value=8.0),
)
def test_every_dual_point_bounds_the_worst_case(
    instance: Instance, epsilon: float, radius: float, log_eta: float
) -> None:
    dual = sinkhorn_dual(
        instance.nominal_array,
        numpy_kernel(instance, epsilon),
        instance.loss_array,
        radius / epsilon,
        math.exp(log_eta),
    )

    assert dual >= worst_case(instance, epsilon, radius) - tolerance(instance)


# --- Structure ---------------------------------------------------------------


@given(instance=instances(), epsilon=epsilons, radius=radii)
def test_worst_case_lies_between_the_smoothed_nominal_and_the_maximum(
    instance: Instance, epsilon: float, radius: float
) -> None:
    value = worst_case(instance, epsilon, radius)

    assert value >= center_expectation(instance, epsilon) - tolerance(instance)
    assert value <= float(instance.loss_array.max()) + tolerance(instance)


@given(instance=instances(), epsilon=epsilons, radius=radii, other=radii)
def test_worst_case_is_nondecreasing_in_radius(
    instance: Instance, epsilon: float, radius: float, other: float
) -> None:
    smaller, larger = sorted((radius, other))

    assert worst_case(instance, epsilon, smaller) <= worst_case(
        instance, epsilon, larger
    ) + tolerance(instance)


@given(instance=instances(), epsilon=epsilons)
def test_zero_float_radius_is_the_smoothed_nominal_expectation_exactly(
    instance: Instance, epsilon: float
) -> None:
    _, loss, _ = instance.tensors()
    ambiguity_set = build(instance, epsilon, 0.0)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.equal(result, torch.sum(ambiguity_set.smoothed_nominal() * loss))
    assert float(result) == pytest.approx(
        center_expectation(instance, epsilon), abs=tolerance(instance)
    )


@given(instance=instances(), epsilon=epsilons)
def test_zero_tensor_radius_matches_the_float_shortcut(
    instance: Instance, epsilon: float
) -> None:
    value = worst_case(instance, epsilon, torch.tensor(0.0, dtype=torch.float64))

    assert value == pytest.approx(
        worst_case(instance, epsilon, 0.0), abs=tolerance(instance)
    )


@given(instance=instances(), epsilon=epsilons, radius=radii)
def test_a_zero_cost_is_the_kl_ball_around_the_reference(
    instance: Instance, epsilon: float, radius: float
) -> None:
    nominal, loss, cost = instance.tensors()
    reference = torch.linspace(1.0, 2.0, loss.numel(), dtype=torch.float64)

    sinkhorn = SinkhornAmbiguitySet(
        nominal, torch.zeros_like(cost), epsilon, radius, reference=reference
    )
    kl = KLAmbiguitySet(reference / reference.sum(), radius / epsilon)

    assert float(sinkhorn.worst_case_expectation(loss)) == pytest.approx(
        float(kl.worst_case_expectation(loss)), abs=tolerance(instance)
    )


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda instance: instance.name)
@pytest.mark.parametrize("epsilon", [0.3, 1.0])
def test_saturates_exactly_at_the_closed_form_radius(
    instance: Instance, epsilon: float
) -> None:
    # Once the budget covers moving every row's kernel onto the highest-loss
    # scenarios, eps * sum_i p_i log(1 / Q_i(A)), the worst case is the
    # maximum loss; just below it, it is strictly less.
    kernel = numpy_kernel(instance, epsilon)
    loss = instance.loss_array
    top = loss >= loss.max()
    mass = kernel[:, top].sum(axis=1)
    rows = instance.nominal_array > 0.0
    saturation = epsilon * float(
        np.sum(instance.nominal_array[rows] * -np.log(mass[rows]))
    )
    maximum = float(loss.max())

    assert worst_case(instance, epsilon, saturation * (1 + 1e-9) + 1e-15) == (
        pytest.approx(maximum, abs=4 * math.ulp(maximum) + 1e-15)
    )
    if saturation > 1e-12:
        assert worst_case(instance, epsilon, 0.9 * saturation) < maximum - 1e-9


def test_vanishing_regularization_approaches_the_wasserstein_worst_case() -> None:
    nominal, loss, cost = ASYMMETRIC.tensors()
    wasserstein = float(
        WassersteinAmbiguitySet(nominal, cost, 0.3).worst_case_expectation(loss)
    )

    gaps = [
        abs(worst_case(ASYMMETRIC, epsilon, 0.3) - wasserstein)
        for epsilon in (1.0, 0.1, 0.03, 0.01)
    ]

    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-9


@given(
    instance=instances(),
    epsilon=epsilons,
    radius=radii,
    permutation=st.permutations(range(6)),
)
def test_worst_case_is_invariant_to_permuting_the_support(
    instance: Instance, epsilon: float, radius: float, permutation: list[int]
) -> None:
    order = [index for index in permutation if index < len(instance.loss)]
    permuted = Instance(
        name="permuted",
        nominal=tuple(instance.nominal[index] for index in order),
        loss=tuple(instance.loss[index] for index in order),
        position=tuple(instance.position[index] for index in order),
    )

    assert worst_case(permuted, epsilon, radius) == pytest.approx(
        worst_case(instance, epsilon, radius), abs=tolerance(instance)
    )


def test_a_reference_restricts_where_the_worst_case_can_move_mass() -> None:
    nominal, loss, cost = ASYMMETRIC.tensors()
    reference = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    differentiable = loss.clone().requires_grad_(True)
    ambiguity_set = SinkhornAmbiguitySet(nominal, cost, 0.5, 1e3, reference=reference)

    value = ambiguity_set.worst_case_expectation(differentiable)
    value.backward()

    assert float(value.detach()) == pytest.approx(2.0, abs=1e-12)
    assert differentiable.grad is not None
    assert float(differentiable.grad[3]) == 0.0


# --- Gradients ---------------------------------------------------------------


@given(instance=instances(), epsilon=epsilons, radius=positive_radii)
@example(
    # Issue #85: coincident support points left with a vanishing share of
    # the worst case made the Newton polish of the divergence singular.
    instance=Instance(
        name="coincident_vanishing",
        nominal=(0.0, 0.5, 0.0, 0.5),
        loss=(-1.0, -1.0, 0.0, -1.0),
        position=(1.0, 3.0, 0.0, 1.0),
    ),
    epsilon=0.05,
    radius=10**-0.28125,
)
def test_gradient_is_a_distribution_inside_the_ball_attaining_the_worst_case(
    instance: Instance, epsilon: float, radius: float
) -> None:
    _, loss, _ = instance.tensors()
    ambiguity_set = build(instance, epsilon, radius)
    differentiable = loss.clone().requires_grad_(True)

    value = ambiguity_set.worst_case_expectation(differentiable)
    value.backward()

    assert differentiable.grad is not None
    candidate = differentiable.grad
    assert float(candidate.min()) >= -1e-12
    assert float(candidate.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(candidate @ loss) == pytest.approx(
        float(value.detach()), abs=tolerance(instance)
    )
    divergence = entropic_transport_divergence(
        candidate.numpy(),
        instance.nominal_array,
        numpy_kernel(instance, epsilon),
        epsilon,
    )
    assert divergence <= radius * (1.0 + 1e-6) + 1e-9
    # The divergence `contains` uses agrees with the reference up to what
    # its tol of 1e-9 on the L1 marginal violation resolves: a candidate that
    # differs from the smoothed nominal only where the kernel is ~e^-40 meets
    # that tol before the solve moves at all.
    measured = float(ambiguity_set.divergence(candidate, ambiguity_set.nominal))
    assert measured == pytest.approx(divergence, rel=1e-6, abs=1e-9)


SMOOTH = (
    ASYMMETRIC,
    Instance(
        name="generic",
        nominal=(0.05, 0.1, 0.2, 0.15, 0.25, 0.15, 0.1),
        loss=(-1.3, 0.4, 2.2, -0.7, 1.1, 3.6, 0.9),
        position=(0.0, 0.7, 1.9, 2.4, 3.1, 4.6, 5.0),
    ),
)


@pytest.mark.parametrize("instance", SMOOTH, ids=lambda instance: instance.name)
@pytest.mark.parametrize("radius", [0.0, 1e-6, 0.05, 0.3, 1e3])
@pytest.mark.parametrize("epsilon", [0.1, 1.0])
def test_first_and_second_derivatives_match_finite_differences(
    instance: Instance, radius: float, epsilon: float
) -> None:
    function = build(instance, epsilon, radius).worst_case_expectation
    _, loss, _ = instance.tensors()
    leaf = loss.requires_grad_(True)

    assert gradcheck(function, (leaf,), eps=1e-6, atol=1e-6, rtol=1e-3)
    assert gradgradcheck(function, (leaf,), eps=1e-6, atol=1e-6, rtol=1e-3)


def test_inside_the_radius_the_hessian_is_nonzero() -> None:
    _, loss, _ = SMOOTH[1].tensors()

    hessian = torch.autograd.functional.hessian(
        build(SMOOTH[1], 0.5, 0.3).worst_case_expectation, loss
    )

    assert float(torch.linalg.eigvalsh(hessian).max()) > 1e-2
    assert float(torch.linalg.eigvalsh(hessian).min()) >= -1e-12


@pytest.mark.parametrize(
    "cost_dtype", [torch.float64, torch.float32], ids=["f64_cost", "f32_cost"]
)
@pytest.mark.parametrize("partial_reference", [False, True], ids=["full", "partial"])
def test_forward_mode_transforms_of_a_set_built_inside_match_reverse_mode(
    cost_dtype: torch.dtype, partial_reference: bool
) -> None:
    # Built inside the transformed function, the set's kernel and reachable
    # mask belong to the innermost transform level, which the forward-mode
    # rule runs outside (issue #80). A float32 cost has its kernel rebuilt in
    # the float64 working dtype, and a partial reference makes the mask
    # matter.
    instance = SMOOTH[1]
    nominal, loss, cost = instance.tensors()
    reference = (
        torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        if partial_reference
        else None
    )

    def function(loss: torch.Tensor) -> torch.Tensor:
        return SinkhornAmbiguitySet(
            nominal, cost.to(cost_dtype), 0.5, 0.3, reference=reference
        ).worst_case_expectation(loss)

    expected = torch.autograd.functional.hessian(function, loss)

    torch.testing.assert_close(
        torch.func.hessian(function)(loss), expected, rtol=0.0, atol=1e-14
    )
    torch.testing.assert_close(
        torch.func.jacrev(torch.func.jacfwd(function))(loss),
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    torch.testing.assert_close(
        torch.func.vmap(torch.func.hessian(function))(torch.stack([loss, -loss]))[0],
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    assert float(torch.linalg.eigvalsh(expected).max()) > 1e-2


# --- Batching and buffers ----------------------------------------------------


def test_batched_losses_nominals_and_radii_match_looped_evaluation() -> None:
    nominal, loss, cost = ASYMMETRIC.tensors()
    nominals = torch.stack([nominal, nominal.flip(-1), torch.full_like(nominal, 0.25)])
    losses = torch.stack([loss, -loss, loss**2])
    radii_tensor = torch.tensor([0.05, 0.3, 2.0], dtype=torch.float64)

    batched = SinkhornAmbiguitySet(
        nominals, cost, 0.4, radii_tensor
    ).worst_case_expectation(losses)

    for index in range(3):
        single = SinkhornAmbiguitySet(
            nominals[index], cost, 0.4, float(radii_tensor[index])
        ).worst_case_expectation(losses[index])
        assert float(batched[index]) == pytest.approx(float(single), abs=1e-12)


def test_worst_case_never_synchronizes(host_sync_counter: Callable[[], Any]) -> None:
    nominal, loss, cost = ASYMMETRIC.tensors()
    ambiguity_set = SinkhornAmbiguitySet(
        nominal, cost, 0.4, torch.tensor([0.05, 0.3], dtype=torch.float64)
    )
    batch = torch.stack([loss, -loss]).requires_grad_(True)

    with host_sync_counter() as syncs:
        ambiguity_set.worst_case_expectation(batch).sum().backward()

    assert len(syncs) == 0


def test_cost_and_reference_live_once_in_the_divergence() -> None:
    nominal, _, cost = ASYMMETRIC.tensors()
    ambiguity_set = SinkhornAmbiguitySet(nominal, cost, 0.4, 0.1)

    names = [name for name, _ in ambiguity_set.named_buffers()]

    assert ambiguity_set.cost is ambiguity_set.divergence.cost
    assert names.count("divergence.cost") == 1
    assert "cost" not in names
    assert "log_kernel" not in ambiguity_set.state_dict()


def test_contains_the_smoothed_nominal_and_its_worst_case() -> None:
    nominal, loss, cost = ASYMMETRIC.tensors()
    ambiguity_set = SinkhornAmbiguitySet(nominal, cost, 0.4, 0.3)
    differentiable = loss.clone().requires_grad_(True)
    ambiguity_set.worst_case_expectation(differentiable).backward()
    assert differentiable.grad is not None

    assert ambiguity_set.contains(ambiguity_set.smoothed_nominal())
    assert ambiguity_set.contains(differentiable.grad * (1 - 1e-9) + 1e-9 * nominal)
    point_mass = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    assert not ambiguity_set.contains(point_mass)


def test_is_a_tilted_ambiguity_set() -> None:
    nominal, _, cost = ASYMMETRIC.tensors()
    ambiguity_set = SinkhornAmbiguitySet(nominal, cost, 0.4, 0.1)

    assert isinstance(ambiguity_set, TiltedAmbiguitySet)
    assert isinstance(ambiguity_set, AmbiguitySet)
    assert ambiguity_set.epsilon == 0.4


def test_tilt_cost_counts_every_kernel_row() -> None:
    # Each tilt evaluates an (n, n) matrix of rows, so the search must weigh
    # n times the support against its launch-bound threshold, or it widens
    # its steps on solves that are bound by arithmetic.
    nominal, loss, cost = ASYMMETRIC.tensors()
    ambiguity_set = SinkhornAmbiguitySet(nominal, cost, 0.4, 0.1)
    support = nominal.shape[-1]
    batch_shape = torch.Size((3, 2))
    gap = torch.zeros(batch_shape + (support,), dtype=loss.dtype)

    entries = ambiguity_set._tilt_entries(gap, batch_shape)

    assert entries == math.prod(batch_shape) * support**2
    assert entries == support * KLAmbiguitySet(nominal, 0.1)._tilt_entries(
        gap, batch_shape
    )


# --- Validation --------------------------------------------------------------


def test_rejects_a_cost_that_does_not_match_the_nominal() -> None:
    nominal, _, _ = ASYMMETRIC.tensors()

    with pytest.raises(ValueError, match="cost must have shape"):
        SinkhornAmbiguitySet(nominal, torch.zeros(3, 3), 0.4, 0.1)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, math.nan])
def test_rejects_a_nonpositive_epsilon(epsilon: float) -> None:
    nominal, _, cost = ASYMMETRIC.tensors()

    with pytest.raises(ValueError, match="epsilon must be positive"):
        SinkhornAmbiguitySet(nominal, cost, epsilon, 0.1)


def test_rejects_a_reference_of_the_wrong_shape() -> None:
    nominal, _, cost = ASYMMETRIC.tensors()

    with pytest.raises(ValueError, match="reference must have shape"):
        SinkhornAmbiguitySet(nominal, cost, 0.4, 0.1, reference=torch.ones(3))


def test_validate_rejects_a_negative_reference() -> None:
    nominal, _, cost = ASYMMETRIC.tensors()
    reference = torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="reference must be nonnegative"):
        SinkhornAmbiguitySet(
            nominal, cost, 0.4, 0.1, reference=reference, validate=True
        )


def test_rejects_a_negative_radius() -> None:
    nominal, _, cost = ASYMMETRIC.tensors()

    with pytest.raises(ValueError, match="radius must be nonnegative"):
        SinkhornAmbiguitySet(nominal, cost, 0.4, -0.1)
