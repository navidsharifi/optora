r"""Gradient checks of every ambiguity set's worst case w.r.t. nominal and radius.

The worst case $V(\ell, p, \rho) = \sup \{\langle q, \ell \rangle :
D(q \,\|\, p) \le \rho\}$ is concave in $(p, \rho)$, and by the envelope
theorem its radius derivative is the optimal multiplier $\lambda^\star \ge 0$
of the divergence constraint. Only the derivative with respect to $p$ along
directions that keep it summing to one is defined, so the checks parameterize
the nominal through a softmax or perturb it along $e_i - e_j$. This module
checks, in float64:

- **smooth points**: `torch.autograd.gradcheck` over the softmax logits of
  the nominal and a tensor radius, unbatched and batched;
- **independent references**: the radius and nominal derivatives against
  central differences of the NumPy duals in `tests.dro.duality`, which share
  no code with optora;
- **structure**: the radius derivative is nonnegative and nonincreasing in
  the radius, infinite at a zero radius where the worst case grows like
  $\sqrt{\rho}$, and zero together with the nominal derivative once the
  radius saturates the set;
- **plumbing**: no extra work without a parameter that requires grad, a set
  reused across backward passes, `torch.func.grad`, and dtypes.
"""

import dataclasses
import math
from collections.abc import Callable

import pytest
import torch
from torch.autograd import gradcheck

from optora.core.dro_base import AmbiguitySet
from optora.divergences import PhiDivergence
from optora.dro import CVaRAmbiguitySet, SinkhornAmbiguitySet
from optora.dro.phi_dro import PhiAmbiguitySet
from optora.solvers import GradientDescent
from tests.dro.instances import INSTANCES, Instance
from tests.dro.specs import SPECS, Radius, SetSpec, spec_parameters

GRADCHECK_TOLERANCES = {"eps": 1e-6, "atol": 1e-6, "rtol": 1e-3}

# Step and tolerance of the central differences taken on the NumPy duals.
REFERENCE_STEP = 1e-5
REFERENCE_TOLERANCE = 1e-5

Builder = Callable[[torch.Tensor, torch.Tensor, Radius], AmbiguitySet]

_INSTANCES = {instance.name: instance for instance in INSTANCES}

GENERIC = Instance(
    name="generic",
    nominal=(0.05, 0.1, 0.2, 0.15, 0.25, 0.15, 0.1),
    loss=(-1.3, 0.4, 2.2, -0.7, 1.1, 3.6, 0.9),
    position=(0.0, 0.7, 1.9, 2.4, 3.1, 4.6, 5.0),
)

# Instances with distinct losses, distinct support points, and positive
# nominal mass everywhere, so a softmax reaches their nominal.
INSTANCES_UNDER_TEST = (
    _INSTANCES["asymmetric"],
    _INSTANCES["wide_spread"],
    GENERIC,
)

# Radii away from the breakpoints of the total-variation and CVaR worst cases
# on the instances above, and one far past saturation.
SMOOTH_RADII = (0.12, 0.33, 1e3)


def _sinkhorn(
    nominal: torch.Tensor, cost: torch.Tensor, radius: Radius
) -> AmbiguitySet:
    """Return a `SinkhornAmbiguitySet` at a moderate entropic regularization."""
    return SinkhornAmbiguitySet(nominal, cost, 0.5, radius)


BUILDERS: tuple[tuple[str, Builder], ...] = (
    *((spec.name, spec.build) for spec in SPECS),
    ("sinkhorn", _sinkhorn),
)
TILTED = frozenset({"kl", "chi_square", "sinkhorn"})


def select_builders(names: frozenset[str] | None = None) -> pytest.MarkDecorator:
    """Parametrize over `(name, build)` for the builders in `names`, or all."""
    selected = [
        (name, build) for name, build in BUILDERS if names is None or name in names
    ]
    return pytest.mark.parametrize(
        ("name", "build"), selected, ids=[name for name, _ in selected]
    )


builder_parameters = select_builders()
instance_parameters = pytest.mark.parametrize(
    "instance",
    INSTANCES_UNDER_TEST,
    ids=[instance.name for instance in INSTANCES_UNDER_TEST],
)


def worst_case_of_parameters(
    build: Builder, instance: Instance, loss: torch.Tensor | None = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return `(logits, radius) -> worst case` on `instance`'s loss and cost."""
    _, instance_loss, cost = instance.tensors()
    loss = instance_loss if loss is None else loss

    def function(logits: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        nominal = torch.softmax(logits, dim=-1)
        return build(nominal, cost, radius).worst_case_expectation(loss)

    return function


def parameter_gradients(
    build: Builder, instance: Instance, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gradients of the worst case w.r.t. nominal and radius."""
    nominal, loss, cost = instance.tensors()
    nominal.requires_grad_(True)
    radius_tensor = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    value = build(nominal, cost, radius_tensor).worst_case_expectation(loss)
    nominal_gradient, radius_gradient = torch.autograd.grad(
        value, (nominal, radius_tensor)
    )
    return nominal_gradient, radius_gradient


def tangent(size: int, increase: int, decrease: int) -> torch.Tensor:
    """Return the float64 direction `e_increase - e_decrease`."""
    direction = torch.zeros(size, dtype=torch.float64)
    direction[increase] = 1.0
    direction[decrease] = -1.0
    return direction


@builder_parameters
@instance_parameters
@pytest.mark.parametrize("radius", SMOOTH_RADII)
def test_parameter_gradients_match_finite_differences_at_smooth_points(
    name: str, build: Builder, instance: Instance, radius: float
) -> None:
    nominal, _, _ = instance.tensors()
    logits = torch.log(nominal).requires_grad_(True)
    radius_tensor = torch.tensor(radius, dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        worst_case_of_parameters(build, instance),
        (logits, radius_tensor),
        **GRADCHECK_TOLERANCES,
    )


@builder_parameters
def test_parameter_gradients_match_finite_differences_on_a_batch(
    name: str, build: Builder
) -> None:
    nominal, loss, _ = GENERIC.tensors()
    batch = torch.stack([loss, loss.flip(-1), -loss])
    logits = torch.log(nominal).requires_grad_(True)
    radius = torch.tensor([0.12, 0.33, 0.5], dtype=torch.float64, requires_grad=True)
    function = worst_case_of_parameters(build, GENERIC, batch)

    assert function(logits, radius).shape == (3,)
    assert gradcheck(function, (logits, radius), **GRADCHECK_TOLERANCES)


@spec_parameters
@pytest.mark.parametrize(
    "instance", [_INSTANCES["asymmetric"], GENERIC], ids=["asymmetric", "generic"]
)
@pytest.mark.parametrize("radius", [0.12, 0.33])
def test_radius_gradient_matches_the_reference_dual(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    _, radius_gradient = parameter_gradients(spec.build, instance, radius)
    expected = (
        spec.dual_minimum(instance, radius + REFERENCE_STEP)
        - spec.dual_minimum(instance, radius - REFERENCE_STEP)
    ) / (2.0 * REFERENCE_STEP)

    assert float(radius_gradient) == pytest.approx(expected, abs=REFERENCE_TOLERANCE)


@spec_parameters
@pytest.mark.parametrize(
    "instance", [_INSTANCES["asymmetric"], GENERIC], ids=["asymmetric", "generic"]
)
@pytest.mark.parametrize("radius", [0.12, 0.33])
def test_nominal_gradient_matches_the_reference_dual_on_the_simplex(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    nominal_gradient, _ = parameter_gradients(spec.build, instance, radius)
    size = len(instance.nominal)

    def shifted(direction: torch.Tensor, step: float) -> float:
        nominal = instance.nominal_array + step * direction.numpy()
        perturbed = dataclasses.replace(instance, nominal=tuple(nominal))
        return spec.dual_minimum(perturbed, radius)

    for index in range(1, size):
        direction = tangent(size, index, 0)
        expected = (
            shifted(direction, REFERENCE_STEP) - shifted(direction, -REFERENCE_STEP)
        ) / (2.0 * REFERENCE_STEP)

        assert float(nominal_gradient @ direction) == pytest.approx(
            expected, abs=REFERENCE_TOLERANCE
        )


@builder_parameters
@instance_parameters
def test_radius_gradient_is_nonnegative_and_nonincreasing(
    name: str, build: Builder, instance: Instance
) -> None:
    nominal, loss, cost = instance.tensors()
    radius = torch.tensor(
        [0.01, 0.07, 0.12, 0.33, 0.7, 1.5, 4.0, 1e3],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = build(nominal, cost, radius).worst_case_expectation(loss)
    (gradient,) = torch.autograd.grad(value.sum(), radius)
    # The CVaR set bounds the likelihood ratio by exp(radius), and its worst
    # case is concave in that bound rather than in the radius itself.
    slope = gradient / torch.exp(radius.detach()) if name == "cvar" else gradient

    assert torch.all(gradient >= 0.0)
    assert torch.all(slope[1:] <= slope[:-1] + 1e-9)


@builder_parameters
@instance_parameters
def test_zero_radius_gradients_match_the_nominal_expectation(
    name: str, build: Builder, instance: Instance
) -> None:
    nominal, loss, cost = instance.tensors()
    nominal.requires_grad_(True)
    (expected,) = torch.autograd.grad(
        build(nominal, cost, 0.0).worst_case_expectation(loss), nominal
    )
    nominal_gradient, radius_gradient = parameter_gradients(build, instance, 0.0)
    size = len(instance.nominal)

    for index in range(1, size):
        direction = tangent(size, index, 0)
        torch.testing.assert_close(
            nominal_gradient @ direction, expected @ direction, rtol=0.0, atol=1e-9
        )
    if name in TILTED:
        assert radius_gradient > 1e12
    else:
        assert torch.isfinite(radius_gradient)


@select_builders(TILTED)
@instance_parameters
def test_tilted_set_gradients_vanish_once_the_radius_saturates(
    name: str, build: Builder, instance: Instance
) -> None:
    nominal_gradient, radius_gradient = parameter_gradients(build, instance, 1e3)
    size = len(instance.nominal)

    assert float(radius_gradient) == pytest.approx(0.0, abs=1e-12)
    for index in range(1, size):
        assert float(nominal_gradient @ tangent(size, index, 0)) == pytest.approx(
            0.0, abs=1e-9
        )


@builder_parameters
def test_parameter_sensitivities_are_computed_only_when_a_parameter_requires_grad(
    name: str, build: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    nominal, loss, cost = GENERIC.tensors()
    ambiguity_set = build(nominal, cost, torch.tensor(0.33, dtype=torch.float64))

    def unexpected(*_: object) -> None:
        raise AssertionError("parameter sensitivities were computed")

    monkeypatch.setattr(ambiguity_set, "_parameter_sensitivities", unexpected)
    loss.requires_grad_(True)
    ambiguity_set.worst_case_expectation(loss).backward()
    nominal.requires_grad_(True)
    with torch.no_grad():
        ambiguity_set.worst_case_expectation(loss)

    assert loss.grad is not None
    with pytest.raises(AssertionError, match="sensitivities were computed"):
        ambiguity_set.worst_case_expectation(loss)


@builder_parameters
def test_a_set_reused_across_steps_accumulates_parameter_gradients(
    name: str, build: Builder
) -> None:
    nominal, loss, cost = GENERIC.tensors()
    nominal.requires_grad_(True)
    radius = torch.tensor(0.33, dtype=torch.float64, requires_grad=True)
    ambiguity_set = (
        CVaRAmbiguitySet(nominal, torch.exp(-radius.detach()).requires_grad_(True))
        if name == "cvar"
        else build(nominal, cost, radius)
    )
    radius_parameter = ambiguity_set.alpha if name == "cvar" else radius
    assert isinstance(radius_parameter, torch.Tensor)

    ambiguity_set.worst_case_expectation(loss).backward()
    assert nominal.grad is not None and radius_parameter.grad is not None
    first_nominal, first_radius = nominal.grad.clone(), radius_parameter.grad.clone()
    ambiguity_set.worst_case_expectation(loss).backward()

    torch.testing.assert_close(nominal.grad, 2.0 * first_nominal)
    torch.testing.assert_close(radius_parameter.grad, 2.0 * first_radius)


@builder_parameters
def test_torch_func_grad_matches_autograd(name: str, build: Builder) -> None:
    nominal, loss, cost = GENERIC.tensors()
    radius = torch.tensor(0.33, dtype=torch.float64)

    def function(nominal: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return build(nominal, cost, radius).worst_case_expectation(loss)

    expected = parameter_gradients(build, GENERIC, 0.33)
    actual = torch.func.grad(function, argnums=(0, 1))(nominal, radius)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)


@builder_parameters
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_parameter_gradients_follow_the_parameter_dtype(
    name: str, build: Builder, dtype: torch.dtype
) -> None:
    nominal, loss, cost = GENERIC.tensors()
    nominal = nominal.to(dtype).requires_grad_(True)
    radius = torch.tensor(0.33, dtype=dtype, requires_grad=True)
    value = build(nominal, cost.to(dtype), radius).worst_case_expectation(
        loss.to(dtype)
    )
    nominal_gradient, radius_gradient = torch.autograd.grad(value, (nominal, radius))
    expected_nominal, expected_radius = parameter_gradients(build, GENERIC, 0.33)
    size = len(GENERIC.nominal)

    assert nominal_gradient.dtype == dtype
    assert radius_gradient.dtype == dtype
    tolerance = 5e-2 if dtype == torch.float16 else 1e-4
    assert float(radius_gradient) == pytest.approx(
        float(expected_radius), rel=tolerance
    )
    for index in range(1, size):
        direction = tangent(size, index, 0)
        assert float(nominal_gradient.double() @ direction) == pytest.approx(
            float(expected_nominal @ direction), abs=tolerance * 10
        )


# Radii at which the generic instance's reallocation drains, or its CVaR
# tail fills, a scenario exactly: kinks of the worst case in the radius.
RADIUS_KINKS = (
    pytest.param("total_variation", 0.2, id="total_variation-0.2"),
    pytest.param("total_variation", 0.4, id="total_variation-0.4"),
    pytest.param("cvar", -math.log(0.35), id="cvar-0.35"),
    pytest.param("cvar", -math.log(0.6), id="cvar-0.6"),
)


@pytest.mark.parametrize(("name", "radius"), RADIUS_KINKS)
def test_radius_gradient_at_a_kink_lies_between_the_one_sided_derivatives(
    name: str, radius: float
) -> None:
    build = dict(BUILDERS)[name]
    nominal, loss, cost = GENERIC.tensors()

    def value(radius: float) -> float:
        return float(build(nominal, cost, radius).worst_case_expectation(loss))

    _, gradient = parameter_gradients(build, GENERIC, radius)
    step = 1e-6
    right = (value(radius + step) - value(radius)) / step
    left = (value(radius) - value(radius - step)) / step

    assert left - right > 1e-2
    assert right - 1e-6 <= float(gradient) <= left + 1e-6


def test_phi_set_parameter_gradients_match_finite_differences_once_converged() -> None:
    instance = _INSTANCES["asymmetric"]

    def build(nominal: torch.Tensor, _: torch.Tensor, radius: Radius) -> AmbiguitySet:
        return PhiAmbiguitySet(
            nominal,
            divergence=PhiDivergence(
                phi=lambda ratio: ratio * torch.log(ratio) - ratio + 1
            ),
            radius=radius,
            phi_conjugate=torch.expm1,
            dual_solver=GradientDescent(step_size=0.5, max_iter=200_000, tol=1e-13),
        )

    nominal, _, _ = instance.tensors()
    logits = torch.log(nominal).requires_grad_(True)
    radius = torch.tensor(0.33, dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        worst_case_of_parameters(build, instance),
        (logits, radius),
        **GRADCHECK_TOLERANCES,
    )
