r"""Property-based tests of the invariants every ambiguity set must satisfy.

For an ambiguity set $\mathcal{U}$ of radius $\rho$ around a nominal $p$,
the worst case $V(\rho) = \sup_{q \in \mathcal{U}} \mathbb{E}_q[\ell]$ obeys
structural identities that hold for every input, not only for hand-picked
ones. Each test here states one of them and lets Hypothesis search for a
counterexample over random nominals, losses, ground costs, and radii:

- **bounds**: $\mathbb{E}_p[\ell] \le V(\rho) \le \max \ell$ over the
  scenarios the set can reach;
- **monotonicity**: $V$ is nondecreasing in $\rho$;
- **zero radius**: $V(0) = \mathbb{E}_p[\ell]$, exactly for a float radius
  and through the full solve for a tensor one;
- **permutation invariance**: relabelling the support leaves $V$ unchanged;
- **duplicate support**: splitting one support point into two coincident
  copies leaves $V$ unchanged;
- **batching**: one batched call equals a loop of unbatched calls;
- **strong duality**: $V$ equals the minimum of an independent convex dual
  (`tests.dro.duality`), every dual point bounds $V$ from above, and the
  gradient of $V$ with respect to $\ell$ (the maximizer, by Danskin's
  theorem) is a feasible distribution attaining $V$.

The strategies in `tests.dro.strategies` are biased towards the regimes
that broke an ambiguity set before: tiny radii, ties, duplicate support
points, `n = 1`, and zero weights. The hand-picked instances of
`tests.dro.instances` are additionally replayed as explicit examples on
every run, so those regimes are exercised even when a random draw misses
them.

The sets under test and their duality references are the shared registry
`tests.dro.specs.SPECS`, which excludes `PhiAmbiguitySet`.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.dro.duality import Array
from tests.dro.instances import INSTANCES, Instance
from tests.dro.specs import Radius, SetSpec, spec_parameters
from tests.dro.strategies import (
    MAX_SUPPORT_SIZE,
    batches,
    instances,
    positive_radii,
    radii,
)
from tests.dro.tolerances import feasibility_bound

# Relative tolerance on a worst-case value, scaled by the loss magnitude: the
# 1e-9 float64 target of plan milestone 8a.
VALUE_TOLERANCE = 1e-9

# Radii replayed on every hand-picked instance: vanishing, moderate, and far
# past saturation.
REPLAYED_RADII = (1e-14, 1e-10, 0.3, 1e3)


def replay_known_regimes(
    **fixed: object,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Replay every hand-picked instance at every replayed radius as an example.

    Args:
        **fixed: Values for the test's remaining Hypothesis arguments.

    Returns:
        A decorator adding one explicit Hypothesis example per
        `(instance, radius)` pair from `INSTANCES` and `REPLAYED_RADII`.
    """

    def decorate(test: Callable[..., None]) -> Callable[..., None]:
        for instance in INSTANCES:
            for radius in REPLAYED_RADII:
                test = example(instance=instance, radius=radius, **fixed)(test)
        return test

    return decorate


def tolerance(instance: Instance) -> float:
    """Return the absolute tolerance on a worst-case value for `instance`."""
    return VALUE_TOLERANCE * (1.0 + float(np.abs(instance.loss_array).max()))


def worst_case(spec: SetSpec, instance: Instance, radius: Radius) -> float:
    """Evaluate `spec`'s worst-case expectation on `instance` at `radius`."""
    nominal, loss, cost = instance.tensors()
    return float(spec.build(nominal, cost, radius).worst_case_expectation(loss))


def reachable_maximum(spec: SetSpec, instance: Instance) -> float:
    """Return the largest loss on any scenario `spec`'s ball can put mass on."""
    loss = instance.loss_array
    if spec.off_support:
        return float(loss.max())
    return float(loss[instance.nominal_array > 0.0].max())


def maximizer(spec: SetSpec, instance: Instance, radius: float) -> Array:
    """Return the gradient of the worst case with respect to the loss."""
    nominal, loss, cost = instance.tensors()
    loss.requires_grad_(True)
    spec.build(nominal, cost, radius).worst_case_expectation(loss).backward()
    assert loss.grad is not None
    return loss.grad.numpy()


@spec_parameters
@given(instance=instances(), radius=radii)
@replay_known_regimes()
def test_worst_case_lies_between_nominal_and_reachable_maximum(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    value = worst_case(spec, instance, radius)
    nominal_expectation = float(instance.nominal_array @ instance.loss_array)

    assert value >= nominal_expectation - tolerance(instance)
    assert value <= reachable_maximum(spec, instance) + tolerance(instance)


@spec_parameters
@given(instance=instances(), radius=radii, other_radius=radii)
@replay_known_regimes(other_radius=0.5)
def test_worst_case_is_nondecreasing_in_radius(
    spec: SetSpec, instance: Instance, radius: float, other_radius: float
) -> None:
    smaller, larger = sorted((radius, other_radius))

    assert worst_case(spec, instance, smaller) <= worst_case(
        spec, instance, larger
    ) + tolerance(instance)


@spec_parameters
@given(instance=instances())
def test_zero_float_radius_returns_the_nominal_expectation_exactly(
    spec: SetSpec, instance: Instance
) -> None:
    nominal, loss, cost = instance.tensors()

    result = spec.build(nominal, cost, 0.0).worst_case_expectation(loss)

    assert torch.equal(result, torch.sum(nominal * loss))


@spec_parameters
@given(instance=instances())
def test_zero_tensor_radius_returns_the_nominal_expectation(
    spec: SetSpec, instance: Instance
) -> None:
    value = worst_case(spec, instance, torch.tensor(0.0, dtype=torch.float64))

    assert value == pytest.approx(
        float(instance.nominal_array @ instance.loss_array), abs=tolerance(instance)
    )


@spec_parameters
@given(
    instance=instances(),
    radius=radii,
    permutation=st.permutations(range(MAX_SUPPORT_SIZE)),
)
@replay_known_regimes(permutation=tuple(reversed(range(MAX_SUPPORT_SIZE))))
def test_worst_case_is_invariant_to_permuting_the_support(
    spec: SetSpec, instance: Instance, radius: float, permutation: list[int]
) -> None:
    order = [index for index in permutation if index < len(instance.loss)]
    permuted = Instance(
        name="permuted",
        nominal=tuple(instance.nominal[index] for index in order),
        loss=tuple(instance.loss[index] for index in order),
        position=tuple(instance.position[index] for index in order),
    )

    assert worst_case(spec, permuted, radius) == pytest.approx(
        worst_case(spec, instance, radius), abs=tolerance(instance)
    )


@spec_parameters
@given(
    instance=instances(max_size=MAX_SUPPORT_SIZE - 1),
    radius=radii,
    split_index=st.integers(min_value=0, max_value=MAX_SUPPORT_SIZE - 1),
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
@replay_known_regimes(split_index=0, fraction=0.5)
def test_splitting_a_support_point_leaves_the_worst_case_unchanged(
    spec: SetSpec,
    instance: Instance,
    radius: float,
    split_index: int,
    fraction: float,
) -> None:
    index = split_index % len(instance.loss)
    weight = instance.nominal[index]
    split = Instance(
        name="split",
        nominal=instance.nominal[:index]
        + (fraction * weight, (1.0 - fraction) * weight)
        + instance.nominal[index + 1 :],
        loss=instance.loss[: index + 1] + instance.loss[index:],
        position=instance.position[: index + 1] + instance.position[index:],
    )

    assert worst_case(spec, split, radius) == pytest.approx(
        worst_case(spec, instance, radius), abs=tolerance(instance)
    )


@spec_parameters
@given(batch=batches())
def test_batched_evaluation_matches_looped_evaluation(
    spec: SetSpec, batch: tuple[tuple[Instance, ...], tuple[float, ...]]
) -> None:
    members, batch_radii = batch
    tensors = [member.tensors() for member in members]
    nominal = torch.stack([nominal for nominal, _, _ in tensors])
    loss = torch.stack([loss for _, loss, _ in tensors])
    cost = tensors[0][2]
    radius = torch.tensor(batch_radii, dtype=torch.float64)

    batched = spec.build(nominal, cost, radius).worst_case_expectation(loss)

    assert batched.shape == (len(members),)
    for member, member_radius, value in zip(
        members, batch_radii, batched.tolist(), strict=True
    ):
        assert value == pytest.approx(
            worst_case(spec, member, member_radius), abs=tolerance(member)
        )


@spec_parameters
@given(instance=instances(), radius=radii)
@replay_known_regimes()
def test_worst_case_equals_the_dual_minimum(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    assert worst_case(spec, instance, radius) == pytest.approx(
        spec.dual_minimum(instance, radius), abs=tolerance(instance)
    )


@spec_parameters
@given(
    instance=instances(),
    radius=radii,
    coordinate=st.floats(min_value=-30.0, max_value=30.0),
)
@replay_known_regimes(coordinate=0.0)
def test_every_dual_point_bounds_the_worst_case(
    spec: SetSpec, instance: Instance, radius: float, coordinate: float
) -> None:
    assert spec.dual(instance, radius, coordinate) >= worst_case(
        spec, instance, radius
    ) - tolerance(instance)


@spec_parameters
@given(instance=instances(), radius=positive_radii)
@replay_known_regimes()
def test_gradient_is_a_feasible_distribution_attaining_the_worst_case(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    candidate = maximizer(spec, instance, radius)
    mass_tolerance = 1e-12
    if not spec.off_support:
        off_support = instance.nominal_array == 0.0
        assert candidate[off_support].sum() <= mass_tolerance
        candidate = np.where(off_support, 0.0, candidate)

    assert candidate.min() >= -mass_tolerance
    assert candidate.sum() == pytest.approx(1.0, abs=mass_tolerance)
    assert spec.divergence(instance, candidate) <= feasibility_bound(
        radius, candidate.size, rtol=1e-6
    )
    assert float(candidate @ instance.loss_array) == pytest.approx(
        worst_case(spec, instance, radius), abs=tolerance(instance)
    )
