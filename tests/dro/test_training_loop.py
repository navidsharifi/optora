"""Tests that a DRO objective trains `nn.Module` parameters with `torch.optim`.

`AmbiguitySet.worst_case_expectation` is differentiable with respect to
anything the per-scenario loss depends on, so the composition

```python
value = ambiguity_set.worst_case_expectation(loss_fn(model(inputs)))
value.backward()
optimizer.step()
```

is an ordinary PyTorch training loop needing no Optora-specific driver. These
tests pin *why* that is correct rather than only that it runs: by the envelope
theorem the gradient of the optimal value is the gradient of the dual
objective at the (detached) dual optimum, which equals the worst-case
distribution's expectation of the per-scenario gradients. A test asserting
only "the loss decreased" would pass even if the inner solve were
differentiated through, so the gradient identities below are checked first and
directly.
"""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from optora.core.dro_base import AmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.minimax_solver import MinimaxProblem, MinimaxSolver
from optora.dro.phi_dro import ChiSquareAmbiguitySet, TotalVariationAmbiguitySet
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent

DTYPE = torch.float64
RADIUS = 0.15

NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1, 0.15], dtype=DTYPE)

FEATURES = torch.tensor(
    [
        [1.0, 0.5],
        [-0.5, 1.5],
        [2.0, -1.0],
        [0.25, 0.75],
        [-1.5, -0.5],
        [1.0, 2.0],
    ],
    dtype=DTYPE,
)

TARGETS = torch.tensor([0.0, 1.0, 2.0, 0.5, 3.0, 1.5], dtype=DTYPE)

COST = torch.cdist(FEATURES, FEATURES) ** 2

AmbiguitySetFactory = Callable[[], AmbiguitySet]


def _kl() -> AmbiguitySet:
    return KLAmbiguitySet(NOMINAL, radius=RADIUS)


def _chi_square() -> AmbiguitySet:
    return ChiSquareAmbiguitySet(NOMINAL, radius=RADIUS)


def _total_variation() -> AmbiguitySet:
    return TotalVariationAmbiguitySet(NOMINAL, radius=RADIUS)


def _wasserstein() -> AmbiguitySet:
    return WassersteinAmbiguitySet(NOMINAL, cost=COST, radius=RADIUS)


BUILDERS: dict[str, AmbiguitySetFactory] = {
    "kl": _kl,
    "chi_square": _chi_square,
    "total_variation": _total_variation,
    "wasserstein": _wasserstein,
}

# The worst case over a total-variation or Wasserstein ball is a maximum over
# finitely many vertices, so the trained objective is only piecewise smooth and
# has no vanishing gradient at its minimum. Those two converge in value
# instead; see `test_adam_converges_in_value`.
SMOOTH = ("chi_square", "kl")


@pytest.fixture(params=sorted(BUILDERS))
def name(request: pytest.FixtureRequest) -> str:
    """Name one ambiguity set of each kind."""
    return str(request.param)


@pytest.fixture
def factory(name: str) -> AmbiguitySetFactory:
    """Build a freshly constructed ambiguity set of the named kind."""
    return BUILDERS[name]


class _LinearModel(nn.Module):
    """Minimal `nn.Module` producing one prediction per support point."""

    def __init__(self, weight: float = 0.3, bias: float = -0.2) -> None:
        """Initialize the model deterministically.

        Args:
            weight: Value assigned to every entry of the linear weight.
            bias: Value assigned to the bias.
        """
        super().__init__()
        self.linear = nn.Linear(FEATURES.shape[1], 1, dtype=DTYPE)
        with torch.no_grad():
            self.linear.weight.fill_(weight)
            self.linear.bias.fill_(bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict one value per row of `features`.

        Args:
            features: Design matrix of shape `(n, d)`.

        Returns:
            Predictions of shape `(n,)`.
        """
        prediction: torch.Tensor = self.linear(features).squeeze(-1)
        return prediction


def _per_scenario_loss(model: nn.Module) -> torch.Tensor:
    """Return the squared error at each support point as an `(n,)` tensor."""
    return (model(FEATURES) - TARGETS) ** 2


def _flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors into one vector."""
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _flat_gradient(model: nn.Module) -> torch.Tensor:
    """Flatten every parameter's accumulated `.grad` into one vector."""
    return _flat(
        [
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            for parameter in model.parameters()
        ]
    )


def _flat_parameters(model: nn.Module) -> torch.Tensor:
    """Flatten every parameter's value into one detached vector."""
    return _flat([parameter.detach() for parameter in model.parameters()])


def _assign_flat_parameters(model: nn.Module, values: torch.Tensor) -> None:
    """Write a flat parameter vector back into `model`, in `parameters()` order."""
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(values[offset : offset + size].view_as(parameter))
            offset += size


def _gradient(value: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Differentiate `value` with respect to every parameter of `model`."""
    return _flat(list(torch.autograd.grad(value, list(model.parameters()))))


def _central_difference_gradient(
    factory: AmbiguitySetFactory, model: nn.Module, step: float = 1e-6
) -> torch.Tensor:
    """Differentiate the *optimal value* by central differences.

    Each perturbed evaluation builds a fresh ambiguity set so its inner dual
    is solved from scratch, making the result a property of the optimal value
    alone rather than of any warm-start state.

    Args:
        factory: Builds a fresh ambiguity set per evaluation.
        model: Module whose parameters are perturbed one at a time.
        step: Half-width of the central difference.

    Returns:
        A flat vector of finite-difference derivatives, in `parameters()`
        order.
    """
    base = _flat_parameters(model)
    derivatives = torch.zeros_like(base)
    for index in range(base.numel()):
        shifted = []
        for sign in (1.0, -1.0):
            perturbed = base.clone()
            perturbed[index] += sign * step
            _assign_flat_parameters(model, perturbed)
            shifted.append(
                factory().worst_case_expectation(_per_scenario_loss(model)).detach()
            )
        derivatives[index] = (shifted[0] - shifted[1]) / (2.0 * step)
    _assign_flat_parameters(model, base)
    return derivatives


# --- The gradient is the envelope gradient -----------------------------------


def test_gradient_with_respect_to_loss_is_the_worst_case_distribution(
    factory: AmbiguitySetFactory,
) -> None:
    # The envelope theorem says the derivative of the optimal value with
    # respect to each per-scenario loss is the worst-case distribution's mass
    # on that scenario. Checking that directly is the sharpest available test
    # of the composition: an extra gradient path through the dual variable
    # would break the identity, and a dual left short of its optimum would
    # open a duality gap in the equality below.
    ambiguity_set = factory()
    loss = TARGETS.clone().requires_grad_(True)

    value = ambiguity_set.worst_case_expectation(loss)
    (worst_case_distribution,) = torch.autograd.grad(value, loss)

    assert torch.all(worst_case_distribution >= 0.0)
    assert torch.allclose(
        worst_case_distribution.sum(), torch.ones((), dtype=DTYPE), atol=1e-10
    )
    # Strong duality: the dual optimum equals the primal worst-case
    # expectation taken under that distribution.
    assert torch.allclose(
        value, torch.sum(worst_case_distribution * TARGETS), atol=1e-9
    )
    # The worst case sits on the boundary of the ambiguity set, so it is a
    # genuinely different distribution from the nominal one.
    assert not torch.allclose(worst_case_distribution, NOMINAL, atol=1e-3)


def test_module_gradient_equals_the_worst_case_weighted_parameter_gradient() -> None:
    # Spelled out on module parameters, the envelope theorem says
    # grad_theta V = sum_i q*_i grad_theta loss_i with q* held fixed. Compare
    # against that surrogate, and against the nominal-weighted gradient the
    # non-robust objective would produce, so the assertion cannot pass by the
    # inner solve collapsing onto the nominal distribution.
    model = _LinearModel()
    ambiguity_set = _kl()

    gradient = _gradient(
        ambiguity_set.worst_case_expectation(_per_scenario_loss(model)), model
    )

    detached_loss = _per_scenario_loss(model).detach().requires_grad_(True)
    (weights,) = torch.autograd.grad(
        ambiguity_set.worst_case_expectation(detached_loss), detached_loss
    )
    envelope = _gradient(torch.sum(weights * _per_scenario_loss(model)), model)
    nominal_gradient = _gradient(torch.sum(NOMINAL * _per_scenario_loss(model)), model)

    assert torch.allclose(gradient, envelope, atol=1e-8)
    assert not torch.allclose(gradient, nominal_gradient, atol=1e-3)


def test_module_gradient_matches_finite_differences_of_the_optimal_value(
    factory: AmbiguitySetFactory,
) -> None:
    # Finite differences of the optimal value know nothing about how the inner
    # problem is solved, so agreeing with them rules out a gradient that leaks
    # the inner solve's iteration path.
    model = _LinearModel()

    factory().worst_case_expectation(_per_scenario_loss(model)).backward()
    gradient = _flat_gradient(model)

    expected = _central_difference_gradient(factory, model)

    assert torch.allclose(gradient, expected, atol=1e-6)


def test_gradient_is_unchanged_by_a_repeat_solve() -> None:
    # The inner solve carries no state between calls, so repeating an
    # identical evaluation must reproduce the same gradient exactly.
    model = _LinearModel()
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=RADIUS)

    def gradient() -> torch.Tensor:
        return _gradient(
            ambiguity_set.worst_case_expectation(_per_scenario_loss(model)), model
        )

    first = gradient()
    second = gradient()

    assert torch.equal(first, second)


def test_zero_radius_gradient_is_the_nominal_expectation_gradient() -> None:
    # The degenerate ambiguity set contains only the nominal distribution, so
    # the training-loop gradient must reduce exactly to the ordinary
    # nominal-weighted empirical gradient.
    model = _LinearModel()

    robust = _gradient(
        KLAmbiguitySet(NOMINAL, radius=0.0).worst_case_expectation(
            _per_scenario_loss(model)
        ),
        model,
    )
    empirical = _gradient(torch.sum(NOMINAL * _per_scenario_loss(model)), model)

    assert torch.allclose(robust, empirical)


# --- The inner solve stays detached ------------------------------------------


def test_inner_solve_returns_a_detached_worst_case_distribution() -> None:
    # The envelope argument is only valid if the worst-case distribution
    # multiplying the loss carries no autograd history back to the module:
    # its gradient must be that distribution, not a path through the
    # bisection that produced it.
    model = _LinearModel()
    ambiguity_set = KLAmbiguitySet(NOMINAL, radius=RADIUS)

    leaf = _per_scenario_loss(model).detach().requires_grad_(True)
    (distribution,) = torch.autograd.grad(
        ambiguity_set.worst_case_expectation(leaf), leaf
    )
    value = _gradient(
        ambiguity_set.worst_case_expectation(_per_scenario_loss(model)), model
    )
    reweighted = _gradient(torch.sum(distribution * _per_scenario_loss(model)), model)

    assert torch.allclose(value, reweighted, atol=1e-12)


def test_inner_solve_leaves_module_gradients_untouched() -> None:
    # The inner solve runs on a tensor that reaches back into the module.
    # It must never accumulate into the module's `.grad` buffers.
    model = _LinearModel()

    value = _kl().worst_case_expectation(_per_scenario_loss(model))

    assert all(parameter.grad is None for parameter in model.parameters())

    value.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_repeated_backward_calls_accumulate_exactly_once_per_step() -> None:
    # A second evaluation-and-backward must add exactly one more gradient, so
    # `optimizer.zero_grad()` is the only thing standing between steps.
    model = _LinearModel()
    ambiguity_set = _kl()

    ambiguity_set.worst_case_expectation(_per_scenario_loss(model)).backward()
    once = _flat_gradient(model).clone()
    ambiguity_set.worst_case_expectation(_per_scenario_loss(model)).backward()
    twice = _flat_gradient(model)

    assert torch.allclose(twice, 2.0 * once, atol=1e-9)


# --- The loop itself ---------------------------------------------------------


def _train(
    ambiguity_set: AmbiguitySet,
    model: nn.Module,
    steps: int = 120,
    learning_rate: float = 0.1,
) -> list[torch.Tensor]:
    """Run a standard Adam loop over a DRO objective.

    Args:
        ambiguity_set: Ambiguity set providing the worst-case objective.
        model: Module whose parameters are optimized.
        steps: Number of optimizer steps.
        learning_rate: Adam learning rate.

    Returns:
        The worst-case expected loss recorded before each step.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    values = []
    for _ in range(steps):
        optimizer.zero_grad()
        value = ambiguity_set.worst_case_expectation(_per_scenario_loss(model))
        value.backward()
        optimizer.step()
        values.append(value.detach().clone())
    return values


def test_adam_converges_in_value(factory: AmbiguitySetFactory) -> None:
    # Every parameter must move, the objective must fall substantially, the
    # trajectory must then flatten rather than keep drifting, and no nearby
    # parameter vector may do meaningfully better. The last check is what
    # makes this more than "the loss decreased", and it stays meaningful for
    # the piecewise smooth total-variation and Wasserstein objectives, whose
    # minima are kinks with no vanishing gradient.
    model = _LinearModel()
    initial = _flat_parameters(model).clone()

    values = torch.stack(_train(factory(), model))
    descent = values[0] - values.min()

    assert torch.all(torch.abs(_flat_parameters(model) - initial) > 0.0)
    assert descent > 1.0
    assert values[-20:].max() - values[-20:].min() < 0.01 * descent

    optimum = _flat_parameters(model).clone()
    final = factory().worst_case_expectation(_per_scenario_loss(model))
    generator = torch.Generator().manual_seed(0)
    for _ in range(25):
        _assign_flat_parameters(
            model,
            optimum
            + 0.01 * torch.randn(optimum.shape, dtype=DTYPE, generator=generator),
        )
        candidate = factory().worst_case_expectation(_per_scenario_loss(model))
        assert candidate > final - 0.01 * descent
    _assign_flat_parameters(model, optimum)


def test_adam_reaches_a_stationary_point_on_a_smooth_objective(name: str) -> None:
    # Where the robust objective is differentiable, convergence in value must
    # also be convergence in gradient. A gradient merely pointing roughly
    # downhill could still drive the value down, so this pins the direction
    # too.
    if name not in SMOOTH:
        pytest.skip(f"the {name} worst case is only piecewise smooth at its minimum")
    model = _LinearModel()

    _train(BUILDERS[name](), model, steps=400, learning_rate=0.03)
    BUILDERS[name]().worst_case_expectation(_per_scenario_loss(model)).backward()

    assert torch.linalg.vector_norm(_flat_gradient(model)) < 1e-5


def test_adam_reaches_the_same_optimum_as_the_minimax_solver() -> None:
    # The training-loop pattern and `MinimaxSolver` minimize the same
    # objective, so a scalar decision variable must land in the same place
    # whichever drives the outer iteration.
    def loss_fn(decision: torch.Tensor) -> torch.Tensor:
        return (decision - TARGETS) ** 2

    solved = MinimaxSolver(
        solver=GradientDescent(
            step_size=0.02, max_iter=5000, tol=1e-11, check_interval=1
        )
    ).solve(
        MinimaxProblem(
            ambiguity_set=_kl(),
            loss_fn=loss_fn,
            initial_point=torch.zeros((), dtype=DTYPE),
        )
    )

    decision = nn.Parameter(torch.zeros((), dtype=DTYPE))
    ambiguity_set = _kl()
    optimizer = torch.optim.Adam([decision], lr=0.05)
    for _ in range(1000):
        optimizer.zero_grad()
        ambiguity_set.worst_case_expectation(loss_fn(decision)).backward()
        optimizer.step()

    assert solved.converged
    assert torch.allclose(decision.detach(), solved.point, atol=1e-6)


def test_robust_training_trades_nominal_risk_for_worst_case_risk() -> None:
    # Training against a positive radius must produce a genuinely different
    # model from empirical risk minimization: better in the worst case, worse
    # on the nominal distribution. This behavioural signature is what a test
    # asserting only "the loss decreased" would miss entirely.
    robust_model = _LinearModel()
    empirical_model = _LinearModel()

    _train(_kl(), robust_model)
    _train(KLAmbiguitySet(NOMINAL, radius=0.0), empirical_model)

    def nominal_risk(model: nn.Module) -> torch.Tensor:
        return torch.sum(NOMINAL * _per_scenario_loss(model)).detach()

    def worst_case_risk(model: nn.Module) -> torch.Tensor:
        value = _kl().worst_case_expectation(_per_scenario_loss(model))
        return value.detach()

    assert not torch.allclose(
        _flat_parameters(robust_model), _flat_parameters(empirical_model), atol=1e-3
    )
    assert worst_case_risk(robust_model) < worst_case_risk(empirical_model)
    assert nominal_risk(robust_model) > nominal_risk(empirical_model)


def test_training_loop_preserves_dtype_and_keeps_parameters_leaves() -> None:
    # A training loop that quietly promoted dtype, or replaced parameters with
    # non-leaf tensors, would break `torch.optim` in ways the value checks
    # above cannot see.
    model = _LinearModel()

    values = _train(_kl(), model, steps=5)

    assert all(value.dtype is DTYPE for value in values)
    assert all(
        parameter.dtype is DTYPE and parameter.is_leaf
        for parameter in model.parameters()
    )
