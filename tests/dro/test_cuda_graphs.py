"""CUDA-graph capture of the exact worst cases and of the steps built on them.

A minibatch training step on a small model is bound by kernel launches rather
than by arithmetic. Every exact set's forward and backward pass runs a fixed
number of kernels and never reads a value back to the host, so the whole step
can be captured once with `torch.cuda.graph` and replayed, which removes the
launch cost (see the "Launch-bound minibatch steps" section of
`docs/training.md`). These tests pin that property. Each captures a forward
and backward pass, replays it on new data, and compares the result with an
eager evaluation of the same data. The ascent of `transport_map` also has a
fixed trip count and is pinned the same way.

The general `PhiAmbiguitySet` with a `dual_solver`, `MMDAmbiguitySet` and
`MinimaxSolver` read a convergence flag on the host by design, so they are
not capturable and are not tested here.
"""

import copy
import functools
from collections.abc import Callable

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.continuous_wasserstein import transport_map
from optora.dro.group_dro import OnlineGroupDRO, group_losses
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.minibatch import MultilevelMonteCarloEstimator, StochasticPhiDual
from optora.dro.phi_dro import ChiSquareAmbiguitySet, chi_square_conjugate
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from tests.dro.specs import SetSpec, spec_parameters

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device is available"
)

DEVICE = torch.device("cuda")
BATCH = 4
SUPPORT = 32
RADIUS = 0.2
WARMUP_STEPS = 3

Outputs = tuple[torch.Tensor, ...]


def _capture(function: Callable[[], Outputs]) -> tuple[torch.cuda.CUDAGraph, Outputs]:
    """Warm `function` up on a side stream, then capture it in a CUDA graph.

    Warming up on a side stream is the capture recipe of the PyTorch CUDA
    semantics notes. It lets lazily initialized state (cuBLAS handles,
    autograd buffers) be created outside the capture.

    Args:
        function: Callable that reads only static tensors and returns a tuple
            of tensors.

    Returns:
        The captured graph and the static output tensors that every replay
        overwrites.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_STEPS):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    return graph, outputs


def _replay_matches_eager(
    function: Callable[[], Outputs], inputs: torch.Tensor, new_inputs: torch.Tensor
) -> None:
    """Capture `function`, replay it on `new_inputs`, and compare with eager mode.

    Args:
        function: Callable reading the static tensor `inputs`.
        inputs: Static input tensor the graph reads.
        new_inputs: Data copied into `inputs` before the replay.
    """
    graph, captured = _capture(function)
    with torch.no_grad():
        inputs.copy_(new_inputs)
    graph.replay()
    expected = function()
    for replayed, eager in zip(captured, expected, strict=True):
        torch.testing.assert_close(replayed, eager)


def _line_cost(dtype: torch.dtype) -> torch.Tensor:
    points = torch.linspace(0.0, 1.0, SUPPORT, dtype=dtype, device=DEVICE)
    return torch.abs(points[:, None] - points[None, :])


def _nominal(generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.rand(SUPPORT, generator=generator, dtype=dtype) + 0.5
    return (weights / weights.sum()).to(DEVICE)


def _loss(generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(BATCH, SUPPORT, generator=generator, dtype=dtype).to(DEVICE)


def _worst_case_and_gradients(
    build: Callable[[], AmbiguitySet], loss: torch.Tensor, radius: torch.Tensor | None
) -> Callable[[], Outputs]:
    """Return a step computing the worst case and its reverse-mode gradients.

    The set is built inside the step, since a constructor reads only
    metadata. A tensor radius is then differentiated through the
    construction on every call, as CVaR's map to the level `exp(-radius)`
    requires.
    """

    def step() -> Outputs:
        value = build().worst_case_expectation(loss)
        inputs = (loss,) if radius is None else (loss, radius)
        gradients = torch.autograd.grad(value.sum(), inputs)
        return (value.detach(), *gradients)

    return step


@spec_parameters
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tensor_radius", [False, True])
def test_exact_set_forward_and_backward_replay_eagerly(
    spec: SetSpec, dtype: torch.dtype, tensor_radius: bool
) -> None:
    generator = torch.Generator().manual_seed(0)
    radius = (
        torch.full((BATCH,), RADIUS, dtype=dtype, device=DEVICE, requires_grad=True)
        if tensor_radius
        else None
    )
    nominal, cost = _nominal(generator, dtype), _line_cost(dtype)
    loss = _loss(generator, dtype).requires_grad_()

    _replay_matches_eager(
        _worst_case_and_gradients(
            lambda: spec.build(nominal, cost, RADIUS if radius is None else radius),
            loss,
            radius,
        ),
        loss,
        _loss(generator, dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sinkhorn_forward_and_backward_replay_eagerly(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    ambiguity_set = SinkhornAmbiguitySet(
        _nominal(generator, dtype), _line_cost(dtype), epsilon=0.1, radius=RADIUS
    )
    loss = _loss(generator, dtype).requires_grad_()

    _replay_matches_eager(
        _worst_case_and_gradients(lambda: ambiguity_set, loss, None),
        loss,
        _loss(generator, dtype),
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_multilevel_estimator_replays_eagerly_at_a_fixed_level(level: int) -> None:
    # The level, and with it the batch shape, is drawn on the host before the
    # minibatch is formed, so a training loop captures one graph per level.
    base_batch_size = 8
    estimator = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=RADIUS),
        base_batch_size=base_batch_size,
        max_level=3,
        device=DEVICE,
    )
    generator = torch.Generator().manual_seed(0)
    size = base_batch_size * 2**level
    loss = torch.randn(BATCH, size, generator=generator).to(DEVICE).requires_grad_()

    def step() -> Outputs:
        value = estimator(loss)
        (gradient,) = torch.autograd.grad(value.sum(), loss)
        return value.detach(), gradient

    _replay_matches_eager(
        step, loss, torch.randn(BATCH, size, generator=generator).to(DEVICE)
    )


def test_stochastic_phi_dual_replays_eagerly() -> None:
    dual = StochasticPhiDual(RADIUS, chi_square_conjugate).to(DEVICE)
    generator = torch.Generator().manual_seed(0)
    loss = _loss(generator, torch.float32).requires_grad_()

    def step() -> Outputs:
        value = dual(loss)
        gradients = torch.autograd.grad(value.sum(), (loss, dual.log_eta, dual.lam))
        return (value.detach(), *gradients)

    _replay_matches_eager(step, loss, _loss(generator, torch.float32))


def test_online_group_dro_step_replays_eagerly() -> None:
    num_groups = 4
    generator = torch.Generator().manual_seed(0)
    groups = torch.randint(num_groups, (SUPPORT,), generator=generator).to(DEVICE)
    adversary = OnlineGroupDRO(
        torch.full((BATCH, num_groups), 1.0 / num_groups, device=DEVICE),
        step_size=0.01,
    )
    loss = _loss(generator, torch.float32).requires_grad_()
    new_loss = _loss(generator, torch.float32)
    # The weights are state that every step updates, so the eager reference
    # starts from the weights the warm-up and capture left behind.
    graph, (replayed_gradient,) = _capture(
        lambda: _group_dro_step(adversary, loss, groups, num_groups)
    )
    reference = copy.deepcopy(adversary)
    with torch.no_grad():
        loss.copy_(new_loss)
    graph.replay()
    (eager_gradient,) = _group_dro_step(reference, loss, groups, num_groups)

    torch.testing.assert_close(replayed_gradient, eager_gradient)
    torch.testing.assert_close(adversary.weights, reference.weights)


def test_continuous_wasserstein_transport_map_replays_eagerly() -> None:
    # The ascent has a fixed trip count and differentiates only the inputs,
    # so its forward and backward passes capture without touching the
    # parameters' gradients.
    features, classes = 8, 3
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Linear(features, 16), torch.nn.ELU(), torch.nn.Linear(16, classes)
    ).to(DEVICE)
    labels = torch.randint(classes, (SUPPORT,), generator=generator).to(DEVICE)
    inputs = torch.randn(SUPPORT, features, generator=generator).to(DEVICE)
    new_inputs = torch.randn(SUPPORT, features, generator=generator).to(DEVICE)

    def loss(points: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            network(points), labels, reduction="none"
        )

    _replay_matches_eager(
        lambda: (transport_map(loss, inputs, 0.5),), inputs, new_inputs
    )
    assert all(parameter.grad is None for parameter in network.parameters())


def _group_dro_step(
    adversary: OnlineGroupDRO,
    loss: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> Outputs:
    group_loss = group_losses(loss, groups, num_groups)
    adversary.step(group_loss)
    (gradient,) = torch.autograd.grad(adversary(group_loss).sum(), loss)
    return (gradient,)


def test_captured_training_step_matches_eager_training() -> None:
    # The forward and backward pass, model and chi-square worst case, is
    # captured and the optimizer update runs eagerly on the gradients the
    # replay writes. Gradients are set to None before capture, so each
    # replay overwrites them rather than accumulating (PyTorch CUDA
    # semantics notes, whole-network capture).
    features, classes, steps = 8, 3, 5
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(steps, SUPPORT, features, generator=generator).to(DEVICE)
    targets = torch.randint(classes, (steps, SUPPORT), generator=generator).to(DEVICE)
    ambiguity_set = ChiSquareAmbiguitySet(
        torch.full((SUPPORT,), 1.0 / SUPPORT, device=DEVICE), radius=2.0
    )
    model = torch.nn.Linear(features, classes).to(DEVICE)
    eager_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    eager_optimizer = torch.optim.SGD(
        eager_model.parameters(), lr=0.1, momentum=0.9, nesterov=True
    )
    static_data, static_targets = data[0].clone(), targets[0].clone()

    def forward_backward() -> Outputs:
        per_sample = torch.nn.functional.cross_entropy(
            model(static_data), static_targets, reduction="none"
        )
        value = ambiguity_set.worst_case_expectation(per_sample)
        value.backward()
        return (value.detach(),)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_STEPS):
            model.zero_grad(set_to_none=True)
            forward_backward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    model.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        (captured_value,) = forward_backward()

    for step in range(steps):
        static_data.copy_(data[step])
        static_targets.copy_(targets[step])
        graph.replay()
        optimizer.step()
        eager_optimizer.zero_grad()
        per_sample = torch.nn.functional.cross_entropy(
            eager_model(data[step]), targets[step], reduction="none"
        )
        eager_value = ambiguity_set.worst_case_expectation(per_sample)
        eager_value.backward()
        eager_optimizer.step()
        torch.testing.assert_close(captured_value, eager_value.detach())

    for graphed, eager in zip(
        model.parameters(), eager_model.parameters(), strict=True
    ):
        torch.testing.assert_close(graphed, eager)
