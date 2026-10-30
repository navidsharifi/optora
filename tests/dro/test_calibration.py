"""Tests for the calibrated radii in `optora.dro.calibration`.

Quantiles are checked against `scipy.stats`, which shares no code with the
standard-library normal quantile the helpers use. The statistical claim, that
a calibrated ball is an asymptotically exact confidence region, is checked by
Monte Carlo coverage of the true mean on Gaussian data, where the limit law
is already accurate at a few hundred observations.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from scipy import stats

from optora.core.dro_base import AmbiguitySet
from optora.dro.calibration import (
    chi_square_radius,
    kl_radius,
    phi_divergence_radius,
    wasserstein_radius,
)
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import ChiSquareAmbiguitySet

DTYPE = torch.float64

RadiusFunction = Callable[..., float]

CALIBRATED_SETS: dict[str, tuple[RadiusFunction, type[AmbiguitySet]]] = {
    "kl": (kl_radius, KLAmbiguitySet),
    "chi_square": (chi_square_radius, ChiSquareAmbiguitySet),
}


@pytest.fixture(params=sorted(CALIBRATED_SETS))
def calibrated_set(
    request: pytest.FixtureRequest,
) -> tuple[RadiusFunction, type[AmbiguitySet]]:
    """Pair a phi-divergence radius helper with the set it calibrates."""
    return CALIBRATED_SETS[str(request.param)]


def _uniform(size: int) -> torch.Tensor:
    return torch.full((size,), 1.0 / size, dtype=DTYPE)


def _gaussian(replications: int, size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(replications, size, generator=generator, dtype=DTYPE)


def _assert_coverage(covered: torch.Tensor, confidence: float) -> None:
    """Assert a Monte Carlo coverage is within four standard errors of target."""
    replications = covered.numel()
    standard_error = math.sqrt(confidence * (1.0 - confidence) / replications)
    coverage = float(covered.to(DTYPE).mean())
    assert abs(coverage - confidence) < 4.0 * standard_error


class TestPhiDivergenceRadius:
    """Closed-form radii for smooth phi-divergence balls."""

    @pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.99])
    @pytest.mark.parametrize("num_samples", [1, 37, 1000])
    def test_two_sided_radius_is_the_chi_square_quantile(
        self, num_samples: int, confidence: float
    ) -> None:
        quantile = stats.chi2.ppf(confidence, df=1)

        assert chi_square_radius(num_samples, confidence) == pytest.approx(
            quantile / num_samples, rel=1e-12
        )
        assert kl_radius(num_samples, confidence) == pytest.approx(
            quantile / (2 * num_samples), rel=1e-12
        )

    @pytest.mark.parametrize("confidence", [0.6, 0.9, 0.95, 0.99])
    def test_one_sided_radius_is_the_squared_normal_quantile(
        self, confidence: float
    ) -> None:
        quantile = stats.norm.ppf(confidence) ** 2

        result = chi_square_radius(20, confidence, one_sided=True)

        assert result == pytest.approx(quantile / 20, rel=1e-12)
        assert result == pytest.approx(chi_square_radius(20, 2 * confidence - 1))

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0, 7.0])
    def test_radius_scales_with_half_the_curvature(self, curvature: float) -> None:
        result = phi_divergence_radius(50, 0.9, curvature)

        assert result == pytest.approx(0.5 * curvature * chi_square_radius(50, 0.9))

    def test_named_helpers_use_their_generator_curvature(self) -> None:
        assert kl_radius(12, 0.8) == phi_divergence_radius(12, 0.8, 1.0)
        assert chi_square_radius(12, 0.8, True) == phi_divergence_radius(
            12, 0.8, 2.0, True
        )

    def test_radius_grows_with_confidence_and_shrinks_with_samples(self) -> None:
        levels = [0.1, 0.5, 0.9, 0.99, 0.999999]
        radii = [chi_square_radius(10, level) for level in levels]

        assert radii == sorted(radii)
        assert chi_square_radius(40, 0.9) == pytest.approx(
            chi_square_radius(10, 0.9) / 4
        )

    def test_returns_a_python_float(self) -> None:
        assert type(kl_radius(5, 0.9)) is float

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_rejects_two_sided_confidence_outside_the_unit_interval(
        self, confidence: float
    ) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
            chi_square_radius(10, confidence)

    @pytest.mark.parametrize("confidence", [0.0, 0.49, 1.0, float("nan")])
    def test_rejects_one_sided_confidence_below_one_half(
        self, confidence: float
    ) -> None:
        with pytest.raises(ValueError, match=r"one-sided confidence"):
            kl_radius(10, confidence, one_sided=True)

    def test_one_sided_one_half_is_a_zero_radius(self) -> None:
        assert kl_radius(10, 0.5, one_sided=True) == 0.0

    @pytest.mark.parametrize("num_samples", [0, -3])
    def test_rejects_an_empty_sample(self, num_samples: int) -> None:
        with pytest.raises(ValueError, match="num_samples"):
            kl_radius(num_samples, 0.9)

    @pytest.mark.parametrize(
        "curvature", [0.0, -1.0, float("inf"), float("nan")], ids=str
    )
    def test_rejects_a_degenerate_curvature(self, curvature: float) -> None:
        with pytest.raises(ValueError, match="curvature"):
            phi_divergence_radius(10, 0.9, curvature)

    def test_chi_square_radius_reproduces_the_normal_interval(self) -> None:
        num_samples, confidence = 64, 0.95
        observations = _gaussian(1, num_samples, seed=0)[0]
        ambiguity_set = ChiSquareAmbiguitySet(
            _uniform(num_samples), radius=chi_square_radius(num_samples, confidence)
        )

        result = ambiguity_set.worst_case_expectation(observations)

        half_width = stats.norm.ppf(0.975) * torch.std(observations, correction=0)
        expected = observations.mean() + half_width / math.sqrt(num_samples)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_kl_radius_reproduces_the_normal_interval_to_leading_order(self) -> None:
        # The KL worst case is mean + sqrt(2 * radius * variance) + O(radius),
        # so the calibrated radius matches the normal interval up to O(1/n).
        num_samples, confidence = 4000, 0.95
        observations = _gaussian(1, num_samples, seed=1)[0]
        ambiguity_set = KLAmbiguitySet(
            _uniform(num_samples), radius=kl_radius(num_samples, confidence)
        )

        excess = (
            ambiguity_set.worst_case_expectation(observations) - observations.mean()
        )

        half_width = stats.norm.ppf(0.975) * torch.std(observations, correction=0)
        expected = half_width / math.sqrt(num_samples)
        assert float(excess) == pytest.approx(float(expected), rel=0.05)

    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_two_sided_interval_covers_the_true_mean(
        self,
        calibrated_set: tuple[RadiusFunction, type[AmbiguitySet]],
        confidence: float,
    ) -> None:
        radius_function, set_class = calibrated_set
        num_samples = 200
        samples = _gaussian(4000, num_samples, seed=2)
        ambiguity_set = set_class(
            _uniform(num_samples), radius=radius_function(num_samples, confidence)
        )

        upper = ambiguity_set.worst_case_expectation(samples)
        lower = -ambiguity_set.worst_case_expectation(-samples)

        _assert_coverage((lower <= 0.0) & (upper >= 0.0), confidence)

    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_one_sided_upper_bound_covers_the_true_mean(
        self,
        calibrated_set: tuple[RadiusFunction, type[AmbiguitySet]],
        confidence: float,
    ) -> None:
        radius_function, set_class = calibrated_set
        num_samples = 200
        samples = _gaussian(4000, num_samples, seed=3)
        radius = radius_function(num_samples, confidence, one_sided=True)
        ambiguity_set = set_class(_uniform(num_samples), radius=radius)

        upper = ambiguity_set.worst_case_expectation(samples)

        _assert_coverage(upper >= 0.0, confidence)


def _vector_mean_coverage(
    radius_function: RadiusFunction,
    set_class: type[AmbiguitySet],
    confidence: float,
    degrees_of_freedom: int,
    dimension: int,
) -> torch.Tensor:
    """Return, per replication, whether the profile region covers a zero mean.

    With $\\delta = \\mu_0 - \\bar x$ and the plug-in covariance
    $\\hat\\Sigma$, the direction $u = \\hat\\Sigma^{-1} \\delta$ is the one
    along which the region is most likely to miss $\\mu_0$: to first order the
    worst case of $u^\\top x$ exceeds $u^\\top \\mu_0$ exactly when
    $n\\, \\delta^\\top \\hat\\Sigma^{-1} \\delta$ is below the quantile the
    radius was calibrated with. So the region covers $\\mu_0 = 0$ iff that
    one worst case is nonnegative.
    """
    num_samples, replications = 200, 4000
    generator = torch.Generator().manual_seed(degrees_of_freedom + 10 * dimension)
    samples = torch.randn(
        replications, num_samples, dimension, generator=generator, dtype=DTYPE
    )
    mean = samples.mean(dim=-2)
    centered = samples - mean[:, None, :]
    covariance = centered.mT @ centered / num_samples
    direction = torch.linalg.solve(covariance, -mean)
    loss = (samples @ direction[..., None]).squeeze(-1)
    radius = radius_function(
        num_samples, confidence, degrees_of_freedom=degrees_of_freedom
    )

    upper = set_class(_uniform(num_samples), radius=radius).worst_case_expectation(loss)
    return upper >= 0.0


class TestDegreesOfFreedom:
    """Radii calibrated with the chi-square quantile of a d-dimensional parameter."""

    @pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.99])
    @pytest.mark.parametrize("degrees_of_freedom", [1, 2, 3, 5, 10, 100])
    def test_radius_is_the_chi_square_quantile_with_d_degrees_of_freedom(
        self, degrees_of_freedom: int, confidence: float
    ) -> None:
        quantile = stats.chi2.ppf(confidence, df=degrees_of_freedom)

        assert chi_square_radius(
            37, confidence, degrees_of_freedom=degrees_of_freedom
        ) == pytest.approx(quantile / 37, rel=1e-12)
        assert kl_radius(
            37, confidence, degrees_of_freedom=degrees_of_freedom
        ) == pytest.approx(quantile / 74, rel=1e-12)

    @pytest.mark.parametrize(
        ("degrees_of_freedom", "confidence"),
        [
            *(
                (dimension, level)
                for dimension in (2, 3, 4, 7, 50, 1000, 100_000)
                for level in (1e-12, 0.01, 0.4999, 0.5, 0.95, 1.0 - 1e-14)
            ),
            # SciPy's own lower-tail quantile is only reliable this deep in
            # the tail for a small number of degrees of freedom.
            *((dimension, 1e-300) for dimension in (2, 3, 4, 7, 50)),
        ],
    )
    def test_quantile_keeps_its_precision_in_both_tails(
        self, degrees_of_freedom: int, confidence: float
    ) -> None:
        if confidence < 0.5:
            reference = stats.chi2.ppf(confidence, df=degrees_of_freedom)
        else:
            reference = stats.chi2.isf(1.0 - confidence, df=degrees_of_freedom)

        result = chi_square_radius(1, confidence, degrees_of_freedom=degrees_of_freedom)

        assert result == pytest.approx(reference, rel=1e-12)

    @pytest.mark.parametrize("confidence", [0.1, 0.9, 0.999999])
    def test_two_degrees_of_freedom_is_the_exponential_quantile(
        self, confidence: float
    ) -> None:
        result = chi_square_radius(1, confidence, degrees_of_freedom=2)

        assert result == pytest.approx(-2.0 * math.log1p(-confidence), rel=1e-13)

    def test_one_degree_of_freedom_is_the_default(self) -> None:
        assert chi_square_radius(12, 0.9, degrees_of_freedom=1) == chi_square_radius(
            12, 0.9
        )
        assert kl_radius(12, 0.9, True, 1) == kl_radius(12, 0.9, True)

    def test_radius_grows_with_the_dimension(self) -> None:
        radii = [
            phi_divergence_radius(50, 0.95, 2.0, degrees_of_freedom=dimension)
            for dimension in range(1, 30)
        ]

        assert radii == sorted(radii)
        assert len(set(radii)) == len(radii)

    def test_returns_a_python_float(self) -> None:
        assert type(kl_radius(5, 0.9, degrees_of_freedom=4)) is float

    @pytest.mark.parametrize("degrees_of_freedom", [0, -2, 1.5, True, "3"], ids=repr)
    def test_rejects_a_non_positive_integer(self, degrees_of_freedom: Any) -> None:
        with pytest.raises(ValueError, match="degrees_of_freedom"):
            chi_square_radius(10, 0.9, degrees_of_freedom=degrees_of_freedom)

    def test_rejects_a_one_sided_vector_calibration(self) -> None:
        with pytest.raises(ValueError, match="one_sided"):
            kl_radius(10, 0.9, one_sided=True, degrees_of_freedom=2)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
    def test_rejects_confidence_outside_the_unit_interval(
        self, confidence: float
    ) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
            chi_square_radius(10, confidence, degrees_of_freedom=3)

    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_region_covers_a_vector_mean(
        self,
        calibrated_set: tuple[RadiusFunction, type[AmbiguitySet]],
        confidence: float,
    ) -> None:
        radius_function, set_class = calibrated_set

        covered = _vector_mean_coverage(
            radius_function, set_class, confidence, degrees_of_freedom=3, dimension=3
        )

        _assert_coverage(covered, confidence)

    def test_scalar_calibration_undercovers_a_vector_mean(self) -> None:
        # With d = 1 the region covers a 3-dimensional mean with probability
        # P(chi2_3 <= chi2_{1, 0.95}) ~ 0.72, far below the 0.95 asked for.
        covered = _vector_mean_coverage(
            chi_square_radius,
            ChiSquareAmbiguitySet,
            0.95,
            degrees_of_freedom=1,
            dimension=3,
        )

        expected = stats.chi2.cdf(stats.chi2.ppf(0.95, df=1), df=3)
        assert float(covered.to(DTYPE).mean()) == pytest.approx(expected, abs=0.03)


class TestWassersteinRadius:
    """Plug-in robust Wasserstein profile radii for the quadratic cost."""

    def test_matches_the_plug_in_formula(self) -> None:
        generator = torch.Generator().manual_seed(4)
        loss = torch.randn(30, generator=generator, dtype=DTYPE)
        gradient = torch.randn(30, 3, generator=generator, dtype=DTYPE)

        result = wasserstein_radius(loss, gradient, 0.9)

        energy = (gradient**2).sum(-1).mean()
        expected = stats.chi2.ppf(0.9, df=1) * loss.var(correction=0) / (30 * energy)
        assert torch.allclose(result, expected, rtol=1e-12)

    def test_one_sided_uses_the_one_sided_quantile(self) -> None:
        loss = torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE)
        gradient = torch.ones(3, 1, dtype=DTYPE)

        one_sided = wasserstein_radius(loss, gradient, 0.95, one_sided=True)

        assert torch.allclose(one_sided, wasserstein_radius(loss, gradient, 0.9))

    def test_mean_estimation_reproduces_the_normal_interval(self) -> None:
        # For the identity loss every gradient is one, the transport that
        # moves the mean most shifts every point equally, and the worst case
        # is mean + sqrt(radius): the normal interval's upper end.
        observations = _gaussian(1, 50, seed=5)[0]

        radius = wasserstein_radius(observations, torch.ones(50, 1, dtype=DTYPE), 0.95)

        half_width = stats.norm.ppf(0.975) * torch.std(observations, correction=0)
        assert torch.allclose(torch.sqrt(radius), half_width / math.sqrt(50))

    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_profile_region_covers_the_true_risk(self, confidence: float) -> None:
        # For a linear loss w @ x and cost |x - y|^2, the profile of a risk
        # value r is exactly (r - mean loss)^2 / |w|^2, so the calibrated
        # ball contains the true risk 0 exactly when the profile at 0 is
        # within the radius.
        replications, num_samples, features = 4000, 200, 3
        generator = torch.Generator().manual_seed(6)
        inputs = torch.randn(
            replications, num_samples, features, generator=generator, dtype=DTYPE
        )
        weight = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
        loss = inputs @ weight
        gradient = weight.expand_as(inputs)

        radius = wasserstein_radius(loss, gradient, confidence)

        profile = loss.mean(-1) ** 2 / (weight**2).sum()
        _assert_coverage(profile <= radius, confidence)

    def test_batched_samples_match_one_at_a_time(self) -> None:
        generator = torch.Generator().manual_seed(7)
        loss = torch.randn(2, 3, 10, generator=generator, dtype=DTYPE)
        gradient = torch.randn(2, 3, 10, 4, generator=generator, dtype=DTYPE)

        result = wasserstein_radius(loss, gradient, 0.9)

        expected = torch.stack(
            [
                wasserstein_radius(row, row_gradient, 0.9)
                for row, row_gradient in zip(
                    loss.reshape(6, 10), gradient.reshape(6, 10, 4), strict=True
                )
            ]
        )
        assert result.shape == (2, 3)
        assert torch.allclose(result.reshape(6), expected)

    def test_constant_loss_gives_a_zero_radius(self) -> None:
        loss = torch.full((5,), 2.0, dtype=DTYPE)

        for gradient in (torch.zeros(5, 2, dtype=DTYPE), torch.ones(5, 2, dtype=DTYPE)):
            assert wasserstein_radius(loss, gradient, 0.9) == 0.0

    def test_vanishing_gradients_give_an_infinite_radius(self) -> None:
        loss = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)

        result = wasserstein_radius(loss, torch.zeros(3, 2, dtype=DTYPE), 0.9)

        assert torch.isinf(result)

    def test_follows_the_dtype_of_the_loss(self, dtype: torch.dtype) -> None:
        loss = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)

        result = wasserstein_radius(loss, torch.ones(3, 1, dtype=DTYPE), 0.9)

        assert result.dtype == dtype

    @pytest.mark.parametrize(
        ("loss_shape", "gradient_shape"),
        [((), (1,)), ((5,), (5,)), ((5,), (4, 2)), ((2, 5), (5, 2))],
        ids=str,
    )
    def test_rejects_mismatched_shapes(
        self, loss_shape: tuple[int, ...], gradient_shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match="input_gradient"):
            wasserstein_radius(
                torch.zeros(loss_shape), torch.zeros(gradient_shape), 0.9
            )

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            wasserstein_radius(torch.zeros(3), torch.zeros(3, 1), 1.0)

    def test_never_synchronizes(self, host_sync_counter: Callable[[], Any]) -> None:
        loss = torch.linspace(0.0, 1.0, 8)
        gradient = torch.ones(8, 2)

        with host_sync_counter() as syncs:
            wasserstein_radius(loss, gradient, 0.9)

        assert len(syncs) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_stays_on_the_device_of_the_loss(self) -> None:
        loss = torch.linspace(0.0, 1.0, 8, device="cuda")

        result = wasserstein_radius(loss, torch.ones(8, 2, device="cuda"), 0.9)

        assert result.device.type == "cuda"
