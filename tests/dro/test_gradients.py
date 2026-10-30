r"""Gradient checks of every ambiguity set's worst case with respect to the loss.

The worst case $V(\ell) = \sup_{q \in \mathcal{U}} \langle q, \ell \rangle$ is
the support function of the ambiguity set $\mathcal{U}$: convex in $\ell$,
positively homogeneous, translation equivariant
($V(\ell + c\mathbf{1}) = V(\ell) + c$), and differentiable exactly where
its maximizer $q^\star$ is unique, with $\nabla V = q^\star$ there
(Danskin's theorem). This module checks those derivatives in float64:

- **smooth points**: `torch.autograd.gradcheck` and `gradgradcheck` on
  instances without ties, at radii from zero to far past saturation,
  unbatched, batched, and over a tensor sweep of radii;
- **Hessian structure**: the Hessian is symmetric positive semidefinite and
  annihilates both $\mathbf{1}$ and $\ell$, and `torch.func` transforms
  agree with reverse-over-reverse autograd;
- **kinks**: where the maximizer is not unique, one-sided finite
  differences bracket the returned gradient, each side agrees with the
  gradient just past the kink on that side, and the gradient satisfies the
  subgradient inequality of the convex worst case.

`tests/dro/test_properties.py` separately checks, over random instances
with ties forced, that the gradient is always a feasible distribution
attaining the worst case, which is what makes it a subgradient at a kink.

`PhiAmbiguitySet` is not in `tests.dro.specs.SPECS` and is checked
separately at the end of the module: its first and second derivatives are
exact once the inner dual solve has converged, the second through the
implicit-function term built from the dual Hessian.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from optora.divergences import PhiDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import PhiAmbiguitySet
from optora.solvers import GradientDescent
from tests.dro.instances import INSTANCES, Instance
from tests.dro.specs import SPECS, Radius, SetSpec, spec_parameters

GRADCHECK_TOLERANCES = {"eps": 1e-6, "atol": 1e-6, "rtol": 1e-3}

# Step of the one-sided finite differences taken across a kink. Every kink
# below is piecewise linear in the perturbed coordinate, so the step only
# has to stay inside one linear piece and well above the rounding floor.
KINK_STEP = 1e-6
KINK_TOLERANCE = 1e-7

# The smallest jump between the one-sided derivatives that counts as a
# kink, so a kink instance that has accidentally become smooth fails loudly
# instead of passing vacuously.
KINK_MARGIN = 1e-2

GENERIC = Instance(
    name="generic",
    nominal=(0.05, 0.1, 0.2, 0.15, 0.25, 0.15, 0.1),
    loss=(-1.3, 0.4, 2.2, -0.7, 1.1, 3.6, 0.9),
    position=(0.0, 0.7, 1.9, 2.4, 3.1, 4.6, 5.0),
)

_INSTANCES = {instance.name: instance for instance in INSTANCES}

# Instances on which every set's worst case is differentiable: distinct
# losses and distinct support points.
SMOOTH_INSTANCES = (
    _INSTANCES["asymmetric"],
    _INSTANCES["zero_weight_maximum"],
    _INSTANCES["wide_spread"],
    _INSTANCES["single_point"],
    GENERIC,
)

# Zero (the float shortcut), vanishing, moderate, and far past saturation.
SMOOTH_RADII = (0.0, 1e-6, 0.05, 0.3, 1e3)

instance_parameters = pytest.mark.parametrize(
    "instance", SMOOTH_INSTANCES, ids=[instance.name for instance in SMOOTH_INSTANCES]
)
radius_parameters = pytest.mark.parametrize("radius", SMOOTH_RADII)
tilted_spec_parameters = pytest.mark.parametrize(
    "spec",
    [spec for spec in SPECS if spec.name in {"kl", "chi_square"}],
    ids=["kl", "chi_square"],
)


@dataclass(frozen=True)
class Kink:
    """A loss at which an ambiguity set's worst case is not differentiable.

    Attributes:
        name: Identifier used in the pytest parameter id.
        spec_names: Names of the `SPECS` entries for which this is a kink.
        instance: Problem instance whose loss sits on the kink.
        radius: Radius at which the instance is a kink.
        index: Support index whose loss is perturbed to break the tie.
    """

    name: str
    spec_names: frozenset[str]
    instance: Instance
    radius: float
    index: int


KINKS = (
    Kink(
        name="saturated_tied_maximum",
        spec_names=frozenset(
            {"kl", "chi_square", "total_variation", "cvar", "wasserstein"}
        ),
        instance=_INSTANCES["tied_maximum"],
        radius=1e3,
        index=2,
    ),
    # A CVaR tail of mass exp(-1) ~ 0.37 cuts through the tied maximum,
    # which carries mass 0.7, so the tie decides which scenario fills it.
    Kink(
        name="tie_straddles_the_tail",
        spec_names=frozenset({"cvar"}),
        instance=_INSTANCES["tied_maximum"],
        radius=1.0,
        index=2,
    ),
    Kink(
        name="tied_maximum",
        spec_names=frozenset({"total_variation"}),
        instance=_INSTANCES["tied_maximum"],
        radius=0.05,
        index=2,
    ),
    Kink(
        name="tied_minimum",
        spec_names=frozenset({"total_variation", "cvar"}),
        instance=_INSTANCES["duplicate_support"],
        radius=0.3,
        index=0,
    ),
    Kink(
        name="coincident_support",
        spec_names=frozenset({"wasserstein"}),
        instance=_INSTANCES["duplicate_support"],
        radius=0.3,
        index=0,
    ),
)

KINK_CASES = [
    pytest.param(spec, kink, id=f"{spec.name}-{kink.name}")
    for kink in KINKS
    for spec in SPECS
    if spec.name in kink.spec_names
]


def worst_case_function(
    spec: SetSpec, instance: Instance, radius: Radius
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return `loss -> worst_case_expectation(loss)` for `spec` on `instance`."""
    nominal, _, cost = instance.tensors()
    return spec.build(nominal, cost, radius).worst_case_expectation


def leaf_loss(instance: Instance) -> torch.Tensor:
    """Return `instance`'s loss as a float64 leaf tensor requiring grad."""
    _, loss, _ = instance.tensors()
    return loss.requires_grad_(True)


def fixed_grad_outputs(output: torch.Tensor) -> torch.Tensor:
    """Return deterministic, non-uniform cotangents for `gradgradcheck`.

    `gradgradcheck` draws random cotangents by default, which would make a
    failure depend on the global random state.
    """
    return torch.linspace(0.5, 1.5, output.numel(), dtype=output.dtype).reshape(
        output.shape
    )


def gradient(
    function: Callable[[torch.Tensor], torch.Tensor], loss: torch.Tensor
) -> torch.Tensor:
    """Return the gradient of a scalar `function` at `loss`."""
    point = loss.detach().clone().requires_grad_(True)
    (result,) = torch.autograd.grad(function(point), point)
    return result


def unit_vector(size: int, index: int) -> torch.Tensor:
    """Return the float64 standard basis vector `e_index` of length `size`."""
    vector = torch.zeros(size, dtype=torch.float64)
    vector[index] = 1.0
    return vector


@spec_parameters
@instance_parameters
@radius_parameters
def test_gradient_matches_finite_differences_at_smooth_points(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    assert gradcheck(
        worst_case_function(spec, instance, radius),
        (leaf_loss(instance),),
        **GRADCHECK_TOLERANCES,
    )


@spec_parameters
@instance_parameters
@radius_parameters
def test_second_derivative_matches_finite_differences_at_smooth_points(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    function = worst_case_function(spec, instance, radius)
    loss = leaf_loss(instance)

    assert gradgradcheck(
        function,
        (loss,),
        grad_outputs=(fixed_grad_outputs(function(loss)),),
        **GRADCHECK_TOLERANCES,
    )


@spec_parameters
def test_derivatives_match_finite_differences_over_a_radius_sweep(
    spec: SetSpec,
) -> None:
    radius = torch.tensor(SMOOTH_RADII, dtype=torch.float64)
    function = worst_case_function(spec, GENERIC, radius)
    loss = leaf_loss(GENERIC)

    assert function(loss).shape == radius.shape
    assert gradcheck(function, (loss,), **GRADCHECK_TOLERANCES)
    assert gradgradcheck(
        function,
        (loss,),
        grad_outputs=(fixed_grad_outputs(function(loss)),),
        **GRADCHECK_TOLERANCES,
    )


@spec_parameters
@pytest.mark.parametrize("radius", [0.05, 0.3])
def test_derivatives_match_finite_differences_on_a_batched_loss(
    spec: SetSpec, radius: float
) -> None:
    function = worst_case_function(spec, GENERIC, radius)
    _, loss, _ = GENERIC.tensors()
    batch = torch.stack([loss, loss.flip(-1), -loss]).requires_grad_(True)

    assert function(batch).shape == (3,)
    assert gradcheck(function, (batch,), **GRADCHECK_TOLERANCES)
    assert gradgradcheck(
        function,
        (batch,),
        grad_outputs=(fixed_grad_outputs(function(batch)),),
        **GRADCHECK_TOLERANCES,
    )


@spec_parameters
@instance_parameters
@pytest.mark.parametrize("radius", [1e-6, 0.05, 0.3, 1e3])
def test_hessian_is_psd_and_annihilates_shifts_and_rescalings(
    spec: SetSpec, instance: Instance, radius: float
) -> None:
    _, loss, _ = instance.tensors()
    hessian = torch.autograd.functional.hessian(
        worst_case_function(spec, instance, radius), loss
    )
    zeros = torch.zeros_like(loss)

    torch.testing.assert_close(hessian, hessian.mT, rtol=0.0, atol=1e-12)
    assert torch.linalg.eigvalsh(hessian).min() >= -1e-12
    torch.testing.assert_close(
        hessian @ torch.ones_like(loss), zeros, rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(hessian @ loss, zeros, rtol=0.0, atol=1e-10)


@tilted_spec_parameters
def test_tilted_sets_have_a_nonzero_hessian_inside_the_radius(spec: SetSpec) -> None:
    _, loss, _ = GENERIC.tensors()
    hessian = torch.autograd.functional.hessian(
        worst_case_function(spec, GENERIC, 0.3), loss
    )

    assert torch.linalg.eigvalsh(hessian).max() > 1e-2


@spec_parameters
def test_torch_func_transforms_match_reverse_over_reverse_autograd(
    spec: SetSpec,
) -> None:
    function = worst_case_function(spec, GENERIC, 0.3)
    _, loss, _ = GENERIC.tensors()
    expected = torch.autograd.functional.hessian(function, loss)

    torch.testing.assert_close(
        torch.func.grad(function)(loss), gradient(function, loss), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        torch.func.hessian(function)(loss), expected, rtol=0.0, atol=1e-14
    )
    torch.testing.assert_close(
        torch.func.jacrev(torch.func.jacfwd(function))(loss),
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@spec_parameters
def test_torch_func_vmap_matches_a_batched_call(spec: SetSpec) -> None:
    function = worst_case_function(spec, GENERIC, 0.3)
    _, loss, _ = GENERIC.tensors()
    batch = torch.stack([loss, loss.flip(-1), -loss])

    torch.testing.assert_close(
        torch.func.vmap(function)(batch), function(batch), rtol=0.0, atol=1e-14
    )


@tilted_spec_parameters
def test_first_order_backward_does_not_rebuild_the_maximizer(
    spec: SetSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    nominal, _, cost = GENERIC.tensors()
    ambiguity_set = spec.build(nominal, cost, 0.3)

    def unexpected(*_: object) -> torch.Tensor:
        raise AssertionError("a first-order backward rebuilt the maximizer")

    monkeypatch.setattr(ambiguity_set, "_implicit_solution", unexpected)
    loss = leaf_loss(GENERIC)
    ambiguity_set.worst_case_expectation(loss).backward()

    assert loss.grad is not None
    with pytest.raises(AssertionError, match="rebuilt the maximizer"):
        torch.autograd.grad(
            ambiguity_set.worst_case_expectation(loss), loss, create_graph=True
        )


@pytest.mark.parametrize(("spec", "kink"), KINK_CASES)
def test_one_sided_finite_differences_bracket_the_gradient_at_a_kink(
    spec: SetSpec, kink: Kink
) -> None:
    function = worst_case_function(spec, kink.instance, kink.radius)
    _, loss, _ = kink.instance.tensors()
    direction = KINK_STEP * unit_vector(loss.numel(), kink.index)
    value = function(loss)

    right = float(function(loss + direction) - value) / KINK_STEP
    left = float(value - function(loss - direction)) / KINK_STEP
    slope = float(gradient(function, loss)[kink.index])

    assert right - left > KINK_MARGIN
    assert left - KINK_TOLERANCE <= slope <= right + KINK_TOLERANCE


@pytest.mark.parametrize(("spec", "kink"), KINK_CASES)
@pytest.mark.parametrize("side", [1.0, -1.0], ids=["right", "left"])
def test_one_sided_derivative_is_the_gradient_just_past_the_kink(
    spec: SetSpec, kink: Kink, side: float
) -> None:
    function = worst_case_function(spec, kink.instance, kink.radius)
    _, loss, _ = kink.instance.tensors()
    direction = side * KINK_STEP * unit_vector(loss.numel(), kink.index)

    one_sided = float(function(loss + direction) - function(loss)) / (side * KINK_STEP)
    past_the_kink = float(gradient(function, loss + direction)[kink.index])

    assert one_sided == pytest.approx(past_the_kink, abs=KINK_TOLERANCE)


@pytest.mark.parametrize(("spec", "kink"), KINK_CASES)
@pytest.mark.parametrize("step", [1e-3, 1.0])
def test_gradient_at_a_kink_satisfies_the_subgradient_inequality(
    spec: SetSpec, kink: Kink, step: float
) -> None:
    function = worst_case_function(spec, kink.instance, kink.radius)
    _, loss, _ = kink.instance.tensors()
    slope = gradient(function, loss)
    value = float(function(loss))
    size = loss.numel()

    for index in range(size):
        for sign in (1.0, -1.0):
            direction = sign * step * unit_vector(size, index)
            assert (
                float(function(loss + direction))
                >= value + float(slope @ direction) - KINK_TOLERANCE
            )


@tilted_spec_parameters
def test_tilted_sets_split_saturated_ties_in_proportion_to_nominal(
    spec: SetSpec,
) -> None:
    instance = _INSTANCES["tied_maximum"]
    nominal, loss, _ = instance.tensors()
    tied = loss == loss.max()

    slope = gradient(worst_case_function(spec, instance, 1e3), loss)

    torch.testing.assert_close(
        slope,
        torch.where(tied, nominal / nominal[tied].sum(), 0.0),
        rtol=0.0,
        atol=1e-15,
    )


def test_chi_square_gradient_is_continuous_where_a_scenario_enters_the_active_set() -> (
    None
):
    chi_square = next(spec for spec in SPECS if spec.name == "chi_square")
    instance = _INSTANCES["asymmetric"]
    nominal, loss, cost = instance.tensors()
    entering = 1
    weight = nominal * torch.clamp(loss - loss[entering], min=0.0)
    maximizer = weight / weight.sum()
    radius = float(torch.sum((maximizer - nominal) ** 2 / nominal))
    function = worst_case_function(chi_square, instance, radius)
    step = 1e-5 * unit_vector(loss.numel(), entering)

    assert gradcheck(function, (leaf_loss(instance),), **GRADCHECK_TOLERANCES)
    at_kink = gradient(function, loss)[entering]
    right = float(gradient(function, loss + step)[entering] - at_kink) / 1e-5
    left = float(at_kink - gradient(function, loss - step)[entering]) / 1e-5
    assert abs(right - left) > KINK_MARGIN


def _kl_phi_ambiguity_set(nominal: torch.Tensor, radius: Radius) -> PhiAmbiguitySet:
    """Return a KL-generator `PhiAmbiguitySet` with a tightly converged dual."""
    return PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(
            phi=lambda ratio: ratio * torch.log(ratio) - ratio + 1
        ),
        radius=radius,
        phi_conjugate=torch.expm1,
        dual_solver=GradientDescent(step_size=0.5, max_iter=200_000, tol=1e-13),
    )


def test_phi_set_gradient_matches_finite_differences_once_converged() -> None:
    instance = _INSTANCES["asymmetric"]
    nominal, _, _ = instance.tensors()
    ambiguity_set = _kl_phi_ambiguity_set(nominal, 0.3)

    assert gradcheck(
        ambiguity_set.worst_case_expectation,
        (leaf_loss(instance),),
        **GRADCHECK_TOLERANCES,
    )


def test_phi_set_second_derivative_matches_finite_differences() -> None:
    instance = _INSTANCES["asymmetric"]
    nominal, _, _ = instance.tensors()
    function = _kl_phi_ambiguity_set(nominal, 0.3).worst_case_expectation
    loss = leaf_loss(instance)

    gradgradcheck(
        function,
        (loss,),
        grad_outputs=(fixed_grad_outputs(function(loss)),),
        **GRADCHECK_TOLERANCES,
    )


def test_phi_set_hessian_matches_the_closed_form_kl_set() -> None:
    nominal, loss, _ = GENERIC.tensors()
    phi_hessian = torch.autograd.functional.hessian(
        _kl_phi_ambiguity_set(nominal, 0.3).worst_case_expectation, loss
    )
    kl_hessian = torch.autograd.functional.hessian(
        KLAmbiguitySet(nominal, 0.3).worst_case_expectation, loss
    )

    assert torch.linalg.eigvalsh(kl_hessian).max() > 1e-2
    torch.testing.assert_close(phi_hessian, kl_hessian, rtol=0.0, atol=1e-10)


def test_phi_set_mixed_loss_radius_derivative_matches_finite_differences() -> None:
    nominal, loss, _ = GENERIC.tensors()
    radius = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    point = loss.clone().requires_grad_(True)
    (loss_gradient,) = torch.autograd.grad(
        _kl_phi_ambiguity_set(nominal, radius).worst_case_expectation(point),
        point,
        create_graph=True,
    )
    mixed = torch.stack(
        [
            torch.autograd.grad(entry, radius, retain_graph=True)[0]
            for entry in loss_gradient
        ]
    )
    step = 1e-5
    expected = (
        gradient(
            _kl_phi_ambiguity_set(nominal, 0.3 + step).worst_case_expectation, loss
        )
        - gradient(
            _kl_phi_ambiguity_set(nominal, 0.3 - step).worst_case_expectation, loss
        )
    ) / (2.0 * step)

    torch.testing.assert_close(mixed, expected, rtol=0.0, atol=1e-6)
