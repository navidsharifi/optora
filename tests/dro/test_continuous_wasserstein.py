"""Tests for the continuous Wasserstein adversary of Sinha et al. (2018).

The transport map is checked against the exact maximizer of linear and
quadratic losses, against an independent PyTorch transcription of the
authors' `wrm` loop, and against `GradientDescent` on the same strongly
concave inner problem.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from optora.core.solver_base import MinimizationProblem
from optora.dro.continuous_wasserstein import squared_distance, transport_map
from optora.solvers.gradient_descent import GradientDescent

Loss = Callable[[torch.Tensor], torch.Tensor]

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device is available"
        ),
    ),
]


def _network(seed: int = 0, dtype: torch.dtype = torch.float64) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 8), nn.ELU(), nn.Linear(8, 2)).to(dtype)


def _cross_entropy(network: nn.Module, labels: torch.Tensor) -> Loss:
    def loss(points: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(network(points), labels, reduction="none")

    return loss


def _inputs(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0), dtype=dtype)


def _authors_wrm(
    loss: Loss, inputs: torch.Tensor, eps: float, steps: int
) -> torch.Tensor:
    """The loop of `attacks_tf.wrm` in the authors' code, written in PyTorch."""
    x = inputs.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad((eps * loss(x)).sum(), x)
    x_adv = (inputs + grad).detach()
    for t in range(steps):
        x_adv.requires_grad_(True)
        (grad,) = torch.autograd.grad((eps * loss(x_adv)).sum(), x_adv)
        grad2 = x_adv - inputs
        x_adv = (x_adv + (grad - grad2) / math.sqrt(t + 2)).detach()
    return x_adv


def test_squared_distance_sums_every_dimension_but_the_first() -> None:
    points = torch.arange(12.0).reshape(2, 3, 2)

    distance = squared_distance(points, torch.zeros(2, 3, 2))

    assert distance.tolist() == [55.0, 451.0]


def test_squared_distance_of_scalar_examples_is_the_squared_difference() -> None:
    distance = squared_distance(torch.tensor([1.0, -2.0]), torch.tensor([0.0, 1.0]))

    assert distance.tolist() == [1.0, 9.0]


def test_squared_distance_of_an_empty_batch_is_empty() -> None:
    assert squared_distance(torch.zeros(0, 3), torch.zeros(0, 3)).shape == (0,)


def test_squared_distance_promotes_dtypes_and_keeps_half_precision(
    half_dtype: torch.dtype,
) -> None:
    points = torch.full((2, 300), 1.0, dtype=half_dtype)
    inputs = torch.zeros(2, 300, dtype=half_dtype)

    assert squared_distance(points, inputs).dtype == half_dtype
    assert squared_distance(points, inputs).tolist() == [300.0, 300.0]
    assert squared_distance(points, inputs.double()).dtype == torch.float64


@pytest.mark.parametrize(
    ("points", "inputs", "message"),
    [
        (torch.zeros(2, 3), torch.zeros(3, 2), "same shape"),
        (torch.zeros(2, 3), torch.zeros(2, 1), "same shape"),
        (torch.tensor(1.0), torch.tensor(0.0), "scalar"),
    ],
)
def test_squared_distance_rejects_invalid_shapes(
    points: torch.Tensor, inputs: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        squared_distance(points, inputs)


@pytest.mark.parametrize("penalty", [0.1, 1.0, 7.5])
def test_a_linear_loss_is_solved_exactly_by_the_first_step(
    penalty: float, dtype: torch.dtype
) -> None:
    slope = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]], dtype=dtype)
    inputs = _inputs(2, 3, dtype=dtype)

    points = transport_map(lambda z: (z * slope).sum(1), inputs, penalty, num_steps=5)

    assert points.dtype == dtype
    torch.testing.assert_close(points, inputs + slope / (2 * penalty))


def test_a_quadratic_loss_converges_to_its_maximizer() -> None:
    # The loss 0.5 z^T A z has the maximizer (2 gamma I - A)^{-1} 2 gamma z0,
    # which is unique because the largest eigenvalue of A is below 2 gamma.
    generator = torch.Generator().manual_seed(0)
    basis, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch.float64)
    )
    eigenvalues = torch.tensor([1.5, 0.5, -1.0, 0.0], dtype=torch.float64)
    curvature = basis @ torch.diag(eigenvalues) @ basis.T
    inputs = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    penalty = 2.0

    points = transport_map(
        lambda z: 0.5 * ((z @ curvature) * z).sum(1), inputs, penalty, num_steps=400
    )

    system = 2 * penalty * torch.eye(4, dtype=torch.float64) - curvature
    exact = torch.linalg.solve(system, 2 * penalty * inputs.T).T
    torch.testing.assert_close(points, exact, atol=1e-8, rtol=0)


@pytest.mark.parametrize("num_steps", [1, 2, 16])
def test_matches_the_authors_wrm_loop(num_steps: int) -> None:
    network = _network()
    loss = _cross_entropy(network, torch.tensor([0, 1, 1, 0, 1]))
    inputs = _inputs(5, 3)

    points = transport_map(loss, inputs, penalty=1 / 2.6, num_steps=num_steps)

    torch.testing.assert_close(points, _authors_wrm(loss, inputs, 1.3, num_steps - 1))


def test_reaches_the_maximizer_found_by_gradient_descent() -> None:
    network = _network(seed=1)
    loss = _cross_entropy(network, torch.tensor([1, 0, 1, 1]))
    inputs = _inputs(4, 3)
    penalty = 3.0

    def objective(points: torch.Tensor) -> torch.Tensor:
        return -(loss(points) - penalty * squared_distance(points, inputs)).sum()

    solver = GradientDescent(step_size=0.05, max_iter=20_000, tol=1e-12)
    oracle = solver.solve(MinimizationProblem(objective, inputs.clone()))

    assert oracle.converged
    torch.testing.assert_close(
        transport_map(loss, inputs, penalty, num_steps=200),
        oracle.point,
        atol=1e-9,
        rtol=0,
    )
    torch.testing.assert_close(
        transport_map(loss, inputs, penalty, num_steps=16),
        oracle.point,
        atol=1e-3,
        rtol=0,
    )


def test_the_transport_cost_falls_as_the_penalty_grows() -> None:
    network = _network(seed=3)
    loss = _cross_entropy(network, torch.tensor([0, 1, 0, 1, 1, 0]))
    inputs = _inputs(6, 3)
    penalties = torch.logspace(-0.5, 2, 6, dtype=torch.float64)

    cost = torch.stack(
        [
            squared_distance(transport_map(loss, inputs, float(gamma)), inputs).mean()
            for gamma in penalties
        ]
    )

    assert bool(torch.all(cost[1:] < cost[:-1]))


def test_a_penalty_per_example_equals_separate_solves() -> None:
    network = _network(seed=2)
    labels = torch.tensor([0, 1, 0])
    inputs = _inputs(3, 3)
    penalties = torch.tensor([0.5, 2.0, 10.0], dtype=torch.float64)

    joint = transport_map(_cross_entropy(network, labels), inputs, penalties)

    for index in range(3):
        alone = transport_map(
            _cross_entropy(network, labels[index : index + 1]),
            inputs[index : index + 1],
            float(penalties[index]),
        )
        torch.testing.assert_close(joint[index : index + 1], alone)


@pytest.mark.parametrize(
    "penalty",
    [torch.tensor(0.7), torch.tensor([0.7]), torch.full((4,), 0.7)],
    ids=["scalar", "one", "per-example"],
)
def test_a_constant_tensor_penalty_equals_the_float(penalty: torch.Tensor) -> None:
    network = _network()
    loss = _cross_entropy(network, torch.tensor([0, 1, 1, 0]))
    inputs = _inputs(4, 3)

    torch.testing.assert_close(
        transport_map(loss, inputs, penalty), transport_map(loss, inputs, 0.7)
    )


@pytest.mark.parametrize("penalty", [math.inf, torch.tensor([math.inf, 1.0])])
def test_an_infinite_penalty_leaves_the_inputs_unchanged(
    penalty: float | torch.Tensor,
) -> None:
    loss = _cross_entropy(_network(), torch.tensor([0, 1]))
    inputs = _inputs(2, 3)

    points = transport_map(loss, inputs, penalty)

    torch.testing.assert_close(points[0], inputs[0], atol=0, rtol=0)


def test_an_empty_batch_returns_an_empty_batch() -> None:
    loss = _cross_entropy(_network(), torch.zeros(0, dtype=torch.long))
    inputs = torch.zeros(0, 3, dtype=torch.float64)

    assert transport_map(loss, inputs, 1.0).shape == (0, 3)


def test_images_keep_their_shape_and_dtype() -> None:
    torch.manual_seed(0)
    network = nn.Sequential(
        nn.Conv2d(1, 2, 3), nn.ELU(), nn.Flatten(), nn.LazyLinear(3)
    )
    inputs = torch.rand(4, 1, 6, 6, generator=torch.Generator().manual_seed(0))
    loss = _cross_entropy(network, torch.tensor([0, 1, 2, 0]))

    points = transport_map(loss, inputs, 0.5)

    assert points.shape == inputs.shape and points.dtype == torch.float32
    assert not points.requires_grad


def test_scalar_examples_are_transported() -> None:
    inputs = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)

    points = transport_map(torch.sin, inputs, 2.0, num_steps=400)

    # The maximizer of sin(z) - 2 (z - z0)^2 solves cos(z) = 4 (z - z0).
    torch.testing.assert_close(torch.cos(points), 4 * (points - inputs))


def test_half_precision_is_transported_in_float32(half_dtype: torch.dtype) -> None:
    slope = torch.tensor([[1.0, -2.0, 0.5]])
    inputs = torch.tensor([[0.5, 0.25, -1.0]])
    seen: list[torch.dtype] = []

    def loss(points: torch.Tensor) -> torch.Tensor:
        seen.append(points.dtype)
        return (points * slope.to(points.dtype)).sum(1)

    points = transport_map(loss, inputs.to(half_dtype), 0.3)

    assert points.dtype == half_dtype and set(seen) == {half_dtype}
    torch.testing.assert_close(points, (inputs + slope / 0.6).to(half_dtype))


def test_leaves_every_gradient_untouched_and_works_under_no_grad() -> None:
    network = _network()
    loss = _cross_entropy(network, torch.tensor([0, 1]))
    inputs = _inputs(2, 3).requires_grad_()
    penalty = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    with torch.no_grad():
        points = transport_map(loss, inputs, penalty)

    assert all(parameter.grad is None for parameter in network.parameters())
    assert inputs.grad is None and penalty.grad is None
    assert not points.requires_grad
    assert not torch.equal(points, inputs.detach())


def test_a_training_step_descends_the_loss_at_the_transported_points() -> None:
    # The gradient of the surrogate is the gradient of the loss at the fixed
    # map (Danskin's theorem), so it equals the gradient taken by hand at the
    # same points.
    network = _network()
    labels = torch.tensor([0, 1, 1, 0])
    loss = _cross_entropy(network, labels)
    inputs = _inputs(4, 3)

    points = transport_map(loss, inputs, 0.8)
    loss(points).mean().backward()

    expected = torch.autograd.grad(
        _cross_entropy(network, labels)(points.clone()).mean(),
        list(network.parameters()),
    )
    for parameter, gradient in zip(network.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_vmap_over_penalties_equals_separate_solves() -> None:
    network = _network()
    loss = _cross_entropy(network, torch.tensor([0, 1, 1]))
    inputs = _inputs(3, 3)
    penalties = torch.tensor([0.3, 1.0, 4.0], dtype=torch.float64)

    mapped = torch.func.vmap(lambda gamma: transport_map(loss, inputs, gamma))(
        penalties
    )

    for index, gamma in enumerate(penalties):
        torch.testing.assert_close(
            mapped[index], transport_map(loss, inputs, float(gamma))
        )


def test_torch_func_grad_treats_the_map_as_a_constant() -> None:
    network = _network()
    labels = torch.tensor([0, 1, 1, 0])
    inputs = _inputs(4, 3)
    parameters = {name: value.detach() for name, value in network.named_parameters()}

    def robust_loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
        def loss(points: torch.Tensor) -> torch.Tensor:
            logits = torch.func.functional_call(network, params, (points,))
            return nn.functional.cross_entropy(logits, labels, reduction="none")

        return loss(transport_map(loss, inputs, 0.8)).mean()

    functional = torch.func.grad(robust_loss)(parameters)

    robust_loss(dict(network.named_parameters())).backward()
    for name, parameter in network.named_parameters():
        torch.testing.assert_close(functional[name], parameter.grad)


def test_issues_no_host_synchronization(host_sync_counter: Callable[[], Any]) -> None:
    network = _network(dtype=torch.float32)
    loss = _cross_entropy(network, torch.tensor([0, 1, 1, 0]))
    inputs = _inputs(4, 3, dtype=torch.float32)
    penalty = torch.full((4,), 0.5)

    with host_sync_counter() as syncs:
        transport_map(loss, inputs, penalty)
        squared_distance(inputs, inputs)

    assert len(syncs) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_runs_on_the_device_of_the_inputs(device: str) -> None:
    labels = torch.tensor([0, 1, 1, 0, 1])
    inputs = _inputs(5, 3)
    penalty = torch.tensor([0.5, 0.7, 1.0, 2.0, 0.9], dtype=torch.float64)

    reference = transport_map(_cross_entropy(_network(), labels), inputs, penalty)
    points = transport_map(
        _cross_entropy(_network().to(device), labels.to(device)),
        inputs.to(device),
        penalty,
    )

    assert points.device.type == device
    torch.testing.assert_close(points.cpu(), reference)


@pytest.mark.parametrize(
    ("penalty", "num_steps", "message"),
    [
        (0.0, 16, "penalty must be positive"),
        (-1.0, 16, "penalty must be positive"),
        (math.nan, 16, "penalty must be positive"),
        (1.0, 0, "num_steps"),
        (torch.ones(2, 2), 16, r"penalty must have shape"),
        (torch.ones(3), 16, r"penalty must have shape \(\), \(1,\) or \(2,\)"),
    ],
)
def test_rejects_invalid_arguments(
    penalty: float | torch.Tensor, num_steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        transport_map(lambda z: z.sum(1), torch.zeros(2, 2), penalty, num_steps)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.tensor(1.0), "scalar"),
        (torch.zeros(2, 3, dtype=torch.long), "floating point"),
    ],
)
def test_rejects_invalid_inputs(inputs: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        transport_map(lambda z: z.sum(-1), inputs, 1.0)


@pytest.mark.parametrize(
    "loss",
    [lambda z: z.sum(), lambda z: z, lambda z: z.sum(1, keepdim=True)],
    ids=["mean-reduced", "unreduced", "keepdim"],
)
def test_rejects_a_loss_without_one_value_per_example(loss: Loss) -> None:
    with pytest.raises(ValueError, match="one value per example"):
        transport_map(loss, torch.zeros(2, 3), 1.0)


def test_rejects_a_loss_that_ignores_the_points() -> None:
    weight = torch.ones(2, requires_grad=True)

    with pytest.raises(ValueError, match="does not depend on the points"):
        transport_map(lambda z: weight * 1.0, torch.zeros(2, 3), 1.0)
