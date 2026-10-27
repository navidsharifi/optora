"""Performance guard for the ambiguity sets and the minimax solve in `optora.dro`.

Workloads are batched float32 losses, the shape a training step produces.
Every exact set runs a trip count fixed by the dtype, so its forward and
backward passes must issue zero host synchronizations, and so must the
minibatch estimators built on them; the general `PhiAmbiguitySet` dual and the
nested minimax solve pay one per convergence checkpoint.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch

from optora.divergences.f_divergence import PhiDivergence
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.minibatch import MultilevelMonteCarloEstimator, StochasticPhiDual
from optora.dro.minimax_solver import MinimaxProblem, MinimaxSolver
from optora.dro.phi_dro import PhiAmbiguitySet, chi_square_conjugate
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from optora.solvers.gradient_descent import GradientDescent
from tests.dro.specs import SetSpec, spec_parameters

BATCH = 16
SUPPORT = 256
RADIUS = 0.1


def _nominal(generator: torch.Generator) -> torch.Tensor:
    weights = torch.rand(SUPPORT, generator=generator) + 0.5
    return weights / weights.sum()


def _line_cost() -> torch.Tensor:
    points = torch.linspace(0.0, 1.0, SUPPORT)
    return torch.abs(points[:, None] - points[None, :])


def _batched_loss(generator: torch.Generator) -> torch.Tensor:
    return torch.randn(BATCH, SUPPORT, generator=generator)


@spec_parameters
def test_worst_case_expectation(
    spec: SetSpec, guarded_benchmark: Callable[..., Any]
) -> None:
    generator = torch.Generator().manual_seed(0)
    ambiguity_set = spec.build(_nominal(generator), _line_cost(), RADIUS)
    loss = _batched_loss(generator)

    value = guarded_benchmark(
        lambda: ambiguity_set.worst_case_expectation(loss), host_syncs=0
    )

    assert value.shape == (BATCH,)


@spec_parameters
def test_worst_case_expectation_gradient(
    spec: SetSpec, guarded_benchmark: Callable[..., Any]
) -> None:
    generator = torch.Generator().manual_seed(0)
    ambiguity_set = spec.build(_nominal(generator), _line_cost(), RADIUS)
    loss = _batched_loss(generator).requires_grad_()

    def gradient() -> torch.Tensor:
        value = ambiguity_set.worst_case_expectation(loss).sum()
        (grad,) = torch.autograd.grad(value, loss)
        return grad

    grad = guarded_benchmark(gradient, host_syncs=0)

    assert grad.shape == loss.shape


def test_sinkhorn_worst_case_expectation_and_gradient(
    guarded_benchmark: Callable[..., Any],
) -> None:
    # A tilt bisection over a batch of (n, n) kernel tilts: no host reads.
    generator = torch.Generator().manual_seed(0)
    ambiguity_set = SinkhornAmbiguitySet(
        _nominal(generator), _line_cost(), epsilon=0.05, radius=RADIUS
    )
    loss = _batched_loss(generator).requires_grad_()

    def gradient() -> torch.Tensor:
        value = ambiguity_set.worst_case_expectation(loss).sum()
        (grad,) = torch.autograd.grad(value, loss)
        return grad

    grad = guarded_benchmark(gradient, host_syncs=0)

    assert grad.shape == loss.shape


def test_phi_worst_case_expectation(guarded_benchmark: Callable[..., Any]) -> None:
    # Run to `max_iter` so the checkpoint count, and with it the
    # synchronization budget, does not depend on when the dual converges.
    max_iter, check_interval = 200, 50
    generator = torch.Generator().manual_seed(0)
    ambiguity_set = PhiAmbiguitySet(
        _nominal(generator),
        divergence=PhiDivergence(phi=lambda t: t * torch.log(t) - t + 1.0),
        radius=RADIUS,
        phi_conjugate=lambda s: torch.exp(s) - 1.0,
        dual_solver=GradientDescent(
            step_size=0.1, max_iter=max_iter, tol=1e-30, check_interval=check_interval
        ),
    )
    loss = _batched_loss(generator)

    value = guarded_benchmark(
        lambda: ambiguity_set.worst_case_expectation(loss),
        host_syncs=max_iter // check_interval,
    )

    assert value.shape == (BATCH,)


def test_minimax_solve(guarded_benchmark: Callable[..., Any]) -> None:
    # A least-squares decision under a KL ambiguity set. The outer solver
    # never reaches a checkpoint, so the whole nested solve, every inner
    # tilt bisection included, must stay on the device.
    features = 8
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(SUPPORT, features, generator=generator)
    targets = torch.randn(SUPPORT, generator=generator)
    problem = MinimaxProblem(
        ambiguity_set=KLAmbiguitySet(_nominal(generator), radius=RADIUS),
        loss_fn=lambda x: (design @ x - targets) ** 2,
        initial_point=torch.zeros(features),
    )
    solver = MinimaxSolver(
        solver=GradientDescent(
            step_size=0.05, max_iter=20, tol=1e-30, check_interval=1000
        )
    )

    result = guarded_benchmark(lambda: solver.solve(problem), host_syncs=0)

    assert result.point.shape == (features,)


def test_multilevel_estimate_gradient(guarded_benchmark: Callable[..., Any]) -> None:
    # The largest level: three plug-in solves on 32 * 2**3 = SUPPORT samples.
    estimator = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=RADIUS),
        base_batch_size=SUPPORT // 8,
        max_level=3,
    )
    loss = _batched_loss(torch.Generator().manual_seed(0)).requires_grad_()

    def gradient() -> torch.Tensor:
        (grad,) = torch.autograd.grad(estimator(loss).sum(), loss)
        return grad

    grad = guarded_benchmark(gradient, host_syncs=0)

    assert grad.shape == loss.shape


def test_stochastic_phi_dual_gradient(guarded_benchmark: Callable[..., Any]) -> None:
    dual = StochasticPhiDual(RADIUS, chi_square_conjugate)
    loss = _batched_loss(torch.Generator().manual_seed(0)).requires_grad_()

    def gradient() -> torch.Tensor:
        (grad,) = torch.autograd.grad(dual(loss).sum(), loss)
        return grad

    grad = guarded_benchmark(gradient, host_syncs=0)

    assert grad.shape == loss.shape
