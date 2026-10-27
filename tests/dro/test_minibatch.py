"""Tests for the minibatch estimators in `optora.dro.minibatch`.

`MultilevelMonteCarloEstimator` is checked against its definition, composed
here from independently built plug-in sets, and against the claim it exists
for: on a population whose plug-in worst case has an exact expectation, its
mean matches the plug-in at the largest batch size, not the base one.
`StochasticPhiDual` is checked for exact minibatch unbiasedness, weak and
strong duality against the exact KL and chi-square sets, and joint
minimization with a decision variable.
"""

import functools
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.minibatch import MultilevelMonteCarloEstimator, StochasticPhiDual
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    TotalVariationAmbiguitySet,
    chi_square_conjugate,
    cvar_conjugate,
)

DTYPE = torch.float64
RADIUS = 0.3

AmbiguitySetFactory = Callable[[torch.Tensor], AmbiguitySet]
Conjugate = Callable[[torch.Tensor], torch.Tensor]

FACTORIES: dict[str, AmbiguitySetFactory] = {
    "kl": functools.partial(KLAmbiguitySet, radius=RADIUS),
    "chi_square": functools.partial(ChiSquareAmbiguitySet, radius=RADIUS),
    "total_variation": functools.partial(TotalVariationAmbiguitySet, radius=RADIUS),
    "cvar": functools.partial(CVaRAmbiguitySet, alpha=0.3),
}

EXACT_PHI_SETS: dict[str, tuple[Conjugate, Callable[..., AmbiguitySet]]] = {
    "kl": (torch.expm1, KLAmbiguitySet),
    "chi_square": (chi_square_conjugate, ChiSquareAmbiguitySet),
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> AmbiguitySetFactory:
    """Build one plug-in ambiguity set of each nominal-only kind."""
    return FACTORIES[str(request.param)]


@pytest.fixture(params=sorted(EXACT_PHI_SETS))
def exact_phi(
    request: pytest.FixtureRequest,
) -> tuple[Conjugate, Callable[..., AmbiguitySet]]:
    """Pair a phi conjugate with the exact ambiguity set of the same ball."""
    return EXACT_PHI_SETS[str(request.param)]


def _uniform(size: int) -> torch.Tensor:
    return torch.full((size,), 1.0 / size, dtype=DTYPE)


def _plug_in(factory: AmbiguitySetFactory, loss: torch.Tensor) -> torch.Tensor:
    """Plug-in worst case on a freshly built set over `loss`'s samples."""
    return factory(_uniform(loss.shape[-1])).worst_case_expectation(loss)


def _probability(level: int, max_level: int) -> float:
    """Probability of `level` under the truncated geometric law `2 ** -j`."""
    return 2.0**-level / sum(2.0**-j for j in range(1, max_level + 1))


def _reference_estimate(
    factory: AmbiguitySetFactory, loss: torch.Tensor, base: int, max_level: int
) -> torch.Tensor:
    """Compose the multilevel estimate from its definition."""
    size = loss.shape[-1]
    level = round(math.log2(size // base))
    halves = (
        _plug_in(factory, loss[..., : size // 2]),
        _plug_in(factory, loss[..., size // 2 :]),
    )
    correction = _plug_in(factory, loss) - 0.5 * (halves[0] + halves[1])
    return _plug_in(factory, loss[..., :base]) + correction / _probability(
        level, max_level
    )


def _estimator(
    factory: AmbiguitySetFactory, base: int = 3, max_level: int = 3
) -> MultilevelMonteCarloEstimator:
    return MultilevelMonteCarloEstimator(factory, base, max_level, dtype=DTYPE)


def _minimize(
    parameters: list[torch.Tensor], objective: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """Minimize `objective` over `parameters` to float64 precision with L-BFGS."""
    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=1000,
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = objective()
        value.backward()
        return value

    optimizer.step(closure)
    return objective().detach()


class TestMultilevelConstruction:
    """Validation, level sizes, and the batch-size law."""

    @pytest.mark.parametrize(("base", "max_level"), [(0, 2), (-1, 2), (2, -1)], ids=str)
    def test_rejects_invalid_sizes(self, base: int, max_level: int) -> None:
        with pytest.raises(ValueError, match="must be"):
            MultilevelMonteCarloEstimator(FACTORIES["kl"], base, max_level)

    def test_builds_one_uniform_set_per_level(self) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 3, 2)

        sizes = [
            len(ambiguity_set.nominal) for ambiguity_set in estimator.ambiguity_sets
        ]
        assert sizes == [3, 6, 12]
        for ambiguity_set in estimator.ambiguity_sets:
            assert torch.allclose(ambiguity_set.nominal.sum(), torch.tensor(1.0))

    @pytest.mark.parametrize("max_level", [1, 2, 5])
    def test_expected_batch_size_matches_closed_form(self, max_level: int) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 4, max_level)

        expected = 4 * max_level / (1.0 - 2.0**-max_level)
        assert estimator.expected_batch_size == pytest.approx(expected)

    def test_level_zero_always_draws_the_base_size(self) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 5, 0)

        assert estimator.expected_batch_size == 5.0
        assert {estimator.sample_batch_size() for _ in range(20)} == {5}

    def test_sampled_sizes_follow_the_geometric_law(self) -> None:
        max_level, draws = 4, 20000
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 2, max_level)
        generator = torch.Generator().manual_seed(0)

        sizes = [estimator.sample_batch_size(generator) for _ in range(draws)]

        for level in range(1, max_level + 1):
            frequency = sizes.count(2 * 2**level) / draws
            probability = _probability(level, max_level)
            standard_error = math.sqrt(probability * (1 - probability) / draws)
            assert abs(frequency - probability) < 5 * standard_error
        assert set(sizes) <= {2 * 2**level for level in range(1, max_level + 1)}

    def test_generator_makes_draws_reproducible(self) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 2, 4)

        draws = [
            [estimator.sample_batch_size(generator) for _ in range(50)]
            for generator in (
                torch.Generator().manual_seed(7),
                torch.Generator().manual_seed(7),
            )
        ]

        assert draws[0] == draws[1]
        assert len(set(draws[0])) > 1


class TestMultilevelEstimate:
    """Value, gradient, shapes, and precision of a single estimate."""

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_matches_the_definition_at_every_level(
        self, factory: AmbiguitySetFactory, level: int
    ) -> None:
        generator = torch.Generator().manual_seed(level)
        loss = torch.randn(3 * 2**level, generator=generator, dtype=DTYPE)

        result = _estimator(factory)(loss)

        expected = _reference_estimate(factory, loss, base=3, max_level=3)
        assert torch.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_level_zero_is_the_plain_plug_in(
        self, factory: AmbiguitySetFactory
    ) -> None:
        loss = torch.tensor([0.5, -1.0, 2.0], dtype=DTYPE)

        result = _estimator(factory, base=3, max_level=0)(loss)

        assert torch.equal(result, _plug_in(factory, loss))

    def test_single_sample_base_level(self, factory: AmbiguitySetFactory) -> None:
        loss = torch.tensor([1.0, 4.0], dtype=DTYPE)

        result = _estimator(factory, base=1, max_level=1)(loss)

        expected = 1.0 + (_plug_in(factory, loss) - 2.5) / 1.0
        assert torch.allclose(result, expected)

    def test_gradient_is_the_combination_of_level_maximizers(
        self, factory: AmbiguitySetFactory
    ) -> None:
        generator = torch.Generator().manual_seed(0)
        loss = torch.randn(12, generator=generator, dtype=DTYPE).requires_grad_()

        _estimator(factory)(loss).backward()

        reference = loss.detach().clone().requires_grad_()
        _reference_estimate(factory, reference, base=3, max_level=3).backward()
        assert loss.grad is not None and reference.grad is not None
        assert torch.allclose(loss.grad, reference.grad, atol=1e-10)

    def test_gradient_carries_unit_mass(self, factory: AmbiguitySetFactory) -> None:
        # Each level's maximizer is a distribution, so the correction's
        # gradient has zero mass and the base level's has one.
        loss = torch.linspace(-2.0, 3.0, 24, dtype=DTYPE).requires_grad_()

        _estimator(factory)(loss).backward()

        assert loss.grad is not None
        assert loss.grad.sum().item() == pytest.approx(1.0, abs=1e-10)

    def test_batched_loss_matches_rows(self, factory: AmbiguitySetFactory) -> None:
        generator = torch.Generator().manual_seed(1)
        loss = torch.randn(4, 1, 12, generator=generator, dtype=DTYPE)
        estimator = _estimator(factory)

        result = estimator(loss)

        expected = torch.stack([estimator(row) for row in loss.reshape(4, 12)])
        assert result.shape == (4, 1)
        assert torch.allclose(result.reshape(4), expected, atol=1e-10)

    @pytest.mark.parametrize("loss_shape", [(12,), (5, 1, 12)], ids=str)
    def test_tensor_radius_sweeps_in_one_call(
        self, loss_shape: tuple[int, ...]
    ) -> None:
        radii = [0.05, 0.3, 1.0]
        generator = torch.Generator().manual_seed(2)
        loss = torch.randn(loss_shape, generator=generator, dtype=DTYPE)
        sweep = _estimator(
            functools.partial(KLAmbiguitySet, radius=torch.tensor(radii, dtype=DTYPE))
        )

        result = sweep(loss)

        expected = torch.stack(
            [
                _estimator(functools.partial(KLAmbiguitySet, radius=radius))(loss)
                for radius in radii
            ],
            dim=-1,
        )
        assert result.shape == torch.broadcast_shapes(loss_shape[:-1], (3,))
        assert torch.allclose(result, expected.reshape(result.shape), atol=1e-10)

    @pytest.mark.parametrize("size", [0, 3, 5, 7, 48])
    def test_rejects_sizes_it_cannot_sample(self, size: int) -> None:
        with pytest.raises(ValueError, match=r"m in \[6, 12, 24\]"):
            _estimator(FACTORIES["kl"])(torch.zeros(size, dtype=DTYPE))

    def test_rejects_a_scalar_loss(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            _estimator(FACTORIES["kl"])(torch.tensor(1.0, dtype=DTYPE))

    def test_dtype_follows_the_module_and_the_loss(self, dtype: torch.dtype) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 3, 2, dtype=dtype)
        loss = torch.linspace(0.0, 1.0, 6, dtype=torch.float32)

        result = estimator(loss)

        assert result.dtype == torch.promote_types(dtype, torch.float32)
        for ambiguity_set in estimator.ambiguity_sets:
            assert ambiguity_set.nominal.dtype == dtype

    def test_half_precision_loss_returns_float32(self, half_dtype: torch.dtype) -> None:
        estimator = MultilevelMonteCarloEstimator(FACTORIES["kl"], 3, 2)
        loss = torch.linspace(0.0, 1.0, 12).to(half_dtype)

        result = estimator(loss)

        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_forward_and_backward_never_synchronize(
        self, host_sync_counter: Callable[[], Any]
    ) -> None:
        estimator = _estimator(FACTORIES["kl"])
        loss = torch.linspace(-1.0, 1.0, 24, dtype=DTYPE).requires_grad_()

        with host_sync_counter() as syncs:
            estimator(loss).backward()

        assert len(syncs) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_matches_cpu(self) -> None:
        estimator = _estimator(FACTORIES["kl"])
        loss = torch.linspace(-1.0, 1.0, 24, dtype=DTYPE)

        result = estimator.to("cuda")(loss.to("cuda"))

        assert result.device.type == "cuda"
        assert torch.allclose(result.cpu(), _estimator(FACTORIES["kl"])(loss))


class TestMultilevelBias:
    """The estimator's mean is the plug-in mean at the largest batch size.

    Losses are Bernoulli, so a plug-in estimate on `m` samples depends only
    on the number of ones `k`, and equals the worst case over the ball
    around the two-point distribution `(1 - k/m, k/m)`. Its exact
    expectation is a binomial sum, a reference independent of the estimator.
    """

    SUCCESS = 0.15
    BASE = 2
    MAX_LEVEL = 4
    REPLICATES = 40000

    def _exact_plug_in_mean(self, size: int) -> float:
        support = torch.tensor([0.0, 1.0], dtype=DTYPE)
        total = 0.0
        for ones in range(size + 1):
            nominal = torch.tensor([1 - ones / size, ones / size], dtype=DTYPE)
            value = KLAmbiguitySet(nominal, radius=RADIUS).worst_case_expectation(
                support
            )
            weight = (
                math.comb(size, ones)
                * self.SUCCESS**ones
                * (1 - self.SUCCESS) ** (size - ones)
            )
            total += weight * float(value)
        return total

    def test_mean_matches_the_largest_level_not_the_base(self) -> None:
        estimator = _estimator(FACTORIES["kl"], self.BASE, self.MAX_LEVEL)
        generator = torch.Generator().manual_seed(0)
        sizes = torch.tensor(
            [estimator.sample_batch_size(generator) for _ in range(self.REPLICATES)]
        )

        estimates = []
        for size in sizes.unique().tolist():
            count = int((sizes == size).sum())
            losses = torch.bernoulli(
                torch.full((count, size), self.SUCCESS, dtype=DTYPE),
                generator=generator,
            )
            estimates.append(estimator(losses))
        sample = torch.cat(estimates)
        mean = float(sample.mean())
        standard_error = float(sample.std()) / math.sqrt(self.REPLICATES)

        target = self._exact_plug_in_mean(self.BASE * 2**self.MAX_LEVEL)
        base_bias = abs(self._exact_plug_in_mean(self.BASE) - target)
        assert abs(mean - target) < 4 * standard_error
        assert base_bias > 10 * standard_error


class TestStochasticPhiDual:
    """Parameters, unbiasedness, and duality of the stochastic phi dual."""

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            StochasticPhiDual(-0.1, torch.expm1)
        with pytest.raises(ValueError, match="radius"):
            StochasticPhiDual(float("nan"), torch.expm1)
        with pytest.raises(ValueError, match="finite"):
            StochasticPhiDual(0.1, torch.expm1, initial_log_eta=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            StochasticPhiDual(0.1, torch.expm1, initial_lam=float("nan"))
        with pytest.raises(ValueError, match="scalar"):
            StochasticPhiDual(0.1, torch.expm1)(torch.tensor(1.0))

    def test_dual_variables_are_registered_parameters(self) -> None:
        dual = StochasticPhiDual(0.1, torch.expm1, initial_log_eta=0.5, initial_lam=2.0)

        parameters = dict(dual.named_parameters())

        assert set(parameters) == {"log_eta", "lam"}
        assert parameters["log_eta"].shape == ()
        assert parameters["log_eta"].item() == 0.5
        assert parameters["lam"].item() == 2.0

    def test_tensor_radius_gives_each_entry_its_own_dual_variables(self) -> None:
        radius = torch.tensor([0.1, 0.5], dtype=DTYPE)

        dual = StochasticPhiDual(radius, torch.expm1)

        assert dual.log_eta.shape == dual.lam.shape == (2,)
        assert dual.log_eta.dtype == dual.lam.dtype == DTYPE
        assert "radius" in dict(dual.named_buffers())

    def test_matches_the_dual_formula(self) -> None:
        loss = torch.tensor([0.5, 1.0, 3.0, -2.0], dtype=DTYPE)
        dual = StochasticPhiDual(0.2, chi_square_conjugate, 0.3, 0.7).to(DTYPE)

        result = dual(loss)

        eta = math.exp(0.3)
        expected = (
            0.2 * eta + 0.7 + eta * chi_square_conjugate((loss - 0.7) / eta).mean()
        )
        assert torch.allclose(result, expected)

    def test_minibatch_mean_is_exactly_the_full_sample_dual(
        self, exact_phi: tuple[Conjugate, Callable[..., AmbiguitySet]]
    ) -> None:
        conjugate, _ = exact_phi
        generator = torch.Generator().manual_seed(0)
        loss = torch.randn(64, generator=generator, dtype=DTYPE)
        dual = StochasticPhiDual(RADIUS, conjugate, 0.2, 0.1).to(DTYPE)

        full = dual(loss)
        (full_grad,) = torch.autograd.grad(full, dual.log_eta)
        minibatches = dual(loss.reshape(8, 8)).mean()
        (minibatch_grad,) = torch.autograd.grad(minibatches, dual.log_eta)

        assert torch.allclose(minibatches, full, rtol=1e-12)
        assert torch.allclose(minibatch_grad, full_grad, rtol=1e-12)

    @pytest.mark.parametrize(("log_eta", "lam"), [(-1.0, 0.0), (0.0, 1.5), (1.0, -1.0)])
    def test_weak_duality_bounds_the_worst_case(
        self,
        exact_phi: tuple[Conjugate, Callable[..., AmbiguitySet]],
        log_eta: float,
        lam: float,
    ) -> None:
        conjugate, exact_set = exact_phi
        loss = torch.linspace(-1.0, 2.0, 16, dtype=DTYPE)
        dual = StochasticPhiDual(RADIUS, conjugate, log_eta, lam).to(DTYPE)

        exact = exact_set(_uniform(16), radius=RADIUS).worst_case_expectation(loss)

        assert dual(loss) >= exact - 1e-12

    @pytest.mark.parametrize("radius", [0.05, 0.3, 1.0])
    def test_strong_duality_with_the_exact_set(
        self,
        exact_phi: tuple[Conjugate, Callable[..., AmbiguitySet]],
        radius: float,
    ) -> None:
        conjugate, exact_set = exact_phi
        generator = torch.Generator().manual_seed(3)
        loss = torch.randn(50, generator=generator, dtype=DTYPE)
        dual = StochasticPhiDual(radius, conjugate, initial_lam=float(loss.mean()))
        dual = dual.to(DTYPE)

        value = _minimize(list(dual.parameters()), lambda: dual(loss))

        exact = exact_set(_uniform(50), radius=radius).worst_case_expectation(loss)
        assert torch.allclose(value, exact, rtol=1e-7)

    def test_joint_minimization_matches_the_robust_decision(self) -> None:
        # Robust location estimate: the joint minimum over the decision and
        # the dual variables equals the minimum of the exact worst case.
        generator = torch.Generator().manual_seed(4)
        samples = torch.randn(40, generator=generator, dtype=DTYPE) ** 2
        exact_set = ChiSquareAmbiguitySet(_uniform(40), radius=RADIUS)
        decision = torch.zeros((), dtype=DTYPE, requires_grad=True)
        robust = _minimize(
            [decision],
            lambda: exact_set.worst_case_expectation((samples - decision) ** 2),
        )

        joint_decision = torch.zeros((), dtype=DTYPE, requires_grad=True)
        dual = StochasticPhiDual(RADIUS, chi_square_conjugate, initial_lam=1.0)
        dual = dual.to(DTYPE)
        joint = _minimize(
            [joint_decision, *dual.parameters()],
            lambda: dual((samples - joint_decision) ** 2),
        )

        assert torch.allclose(joint, robust, rtol=1e-6)
        assert torch.allclose(joint_decision.detach(), decision.detach(), atol=1e-5)

    @pytest.mark.parametrize("loss_shape", [(10,), (4, 1, 10)], ids=str)
    def test_tensor_radius_matches_scalar_duals(
        self, loss_shape: tuple[int, ...]
    ) -> None:
        radii = [0.05, 0.3, 1.0]
        loss = torch.randn(loss_shape, generator=torch.Generator().manual_seed(5))
        loss = loss.to(DTYPE)
        sweep = StochasticPhiDual(
            torch.tensor(radii, dtype=DTYPE), torch.expm1, 0.1, 0.2
        )

        result = sweep(loss)

        expected = torch.stack(
            [
                StochasticPhiDual(radius, torch.expm1, 0.1, 0.2).to(DTYPE)(loss)
                for radius in radii
            ],
            dim=-1,
        )
        assert result.shape == torch.broadcast_shapes(loss_shape[:-1], (3,))
        assert torch.allclose(result, expected.reshape(result.shape))

    def test_dtype_follows_the_module_and_the_loss(self, dtype: torch.dtype) -> None:
        dual = StochasticPhiDual(0.1, torch.expm1).to(dtype)
        loss = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)

        assert dual(loss).dtype == torch.promote_types(dtype, torch.float32)

    def test_half_precision_loss_returns_float32(self, half_dtype: torch.dtype) -> None:
        dual = StochasticPhiDual(0.1, torch.expm1)
        loss = torch.linspace(0.0, 1.0, 5).to(half_dtype)

        result = dual(loss)

        assert result.dtype == torch.float32
        assert torch.allclose(result, dual(loss.float()))

    @pytest.mark.parametrize("alpha", [1.0, 0.5, 0.3, 0.05])
    def test_cvar_conjugate_at_radius_zero_is_the_rockafellar_uryasev_dual(
        self, alpha: float
    ) -> None:
        # The dual is piecewise linear in lam and minimized at the upper
        # alpha-quantile, where it equals CVaR exactly; any other lam, and
        # any eta, only bounds it from above.
        generator = torch.Generator().manual_seed(6)
        loss = torch.randn(40, generator=generator, dtype=DTYPE)
        conjugate = functools.partial(cvar_conjugate, alpha=alpha)
        exact = CVaRAmbiguitySet(_uniform(40), alpha=alpha).worst_case_expectation(loss)
        quantile = torch.sort(loss, descending=True).values[math.ceil(alpha * 40) - 1]

        for log_eta in (-2.0, 0.0, 3.0):
            at_quantile = StochasticPhiDual(0.0, conjugate, log_eta, float(quantile))
            assert torch.allclose(at_quantile.to(DTYPE)(loss), exact, rtol=1e-12)
            for lam in (-3.0, 0.0, 0.7, 3.0):
                dual = StochasticPhiDual(0.0, conjugate, log_eta, lam).to(DTYPE)
                assert dual(loss) >= exact - 1e-12

    def test_cvar_conjugate_broadcasts_a_tensor_level(self) -> None:
        scaled_shift = torch.tensor([[-1.0, 0.5, 2.0]], dtype=DTYPE)
        alpha = torch.tensor([[0.5], [0.25]], dtype=DTYPE)

        result = cvar_conjugate(scaled_shift, alpha)

        expected = torch.tensor([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]], dtype=DTYPE)
        assert torch.equal(result, expected)

    def test_forward_and_backward_never_synchronize(
        self, host_sync_counter: Callable[[], Any]
    ) -> None:
        dual = StochasticPhiDual(0.1, chi_square_conjugate)
        loss = torch.linspace(-1.0, 1.0, 32).requires_grad_()

        with host_sync_counter() as syncs:
            dual(loss).backward()

        assert len(syncs) == 0
