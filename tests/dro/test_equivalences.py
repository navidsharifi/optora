r"""Regularization equivalences and divergence inequalities as test oracles.

A worst case $V(\rho) = \sup_{q \in \mathcal{U}_\rho} \mathbb{E}_q[\ell]$ is
pinned here against mathematics that no ambiguity set's implementation
uses: regularized objectives it must equal or approach, and inequalities
between different sets that follow from comparing their divergences. Each
oracle is independent of the solve it checks, so a wrong maximizer, a wrong
radius convention, or a wrong dual fails it.

- **Variance regularization.** The chi-square worst case is exactly
  $\mathbb{E}_p[\ell] + \sqrt{\rho\,\mathrm{Var}_p(\ell)}$ until the
  maximizer truncates a scenario, and is bounded by it everywhere. The KL
  worst case follows the cumulant expansion
  $\mathbb{E}_p[\ell] + \sigma\sqrt{2\rho} + \kappa_3\rho/(3\sigma^2)
  + c_3\,\rho^{3/2} + O(\rho^2)$ (Lam 2016 for the first two terms).
- **Lipschitz and gradient-norm regularization.** A type-1 Wasserstein ball
  on a metric support adds $\rho$ times the loss's Lipschitz modulus on the
  sample (Wu, Li and Mao 2025); a type-2 ball, with squared-distance cost
  and budget $\delta$, adds $\sqrt{\delta}\,\lVert \ell' \rVert_{L^2(p)}$
  to first order (Bartl et al. 2021; Gao, Chen and Kleywegt 2024), the
  expansion behind `optora.dro.wasserstein_radius`.
- **Total variation and CVaR.** Spending a total-variation budget $r \le 1$
  moves mass $r$ onto the maximum loss and keeps the upper $(1-r)$-tail of
  the nominal, so
  $V_{\mathrm{TV}}(r) = r \max \ell + (1-r)\,\mathrm{CVaR}_{1-r}(\ell)$.
- **Nesting.** Renyi divergences increase with their order, so
  $D_{\mathrm{KL}} \le \log(1 + \chi^2) \le D_\infty$ (van Erven and
  Harremoes 2014), and Pinsker's inequality bounds total variation by
  $\sqrt{D_{\mathrm{KL}}/2}$. The balls nest accordingly, and so do their
  worst cases:
  $V_{\mathrm{CVaR}}(e^{-\rho}) \le V_{\chi^2}(e^{\rho} - 1)
  \le V_{\mathrm{KL}}(\rho) \le V_{\mathrm{TV}}(\sqrt{\rho/2})$.
"""

import math

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    TotalVariationAmbiguitySet,
)
from optora.dro.wasserstein_dro import WassersteinAmbiguitySet
from tests.dro.instances import Instance
from tests.dro.strategies import instances, radii

# Relative tolerance on a worst-case value, scaled by the loss magnitude, as
# in `tests/dro/test_properties.py`.
VALUE_TOLERANCE = 1e-9

# Beyond this radius `exp(radius) - 1` overflows float64; the chi-square
# ball it names then contains every distribution on the nominal's support.
_LARGEST_FINITE_EXPONENT = 700.0


def tolerance(instance: Instance) -> float:
    """Return the absolute tolerance on a worst-case value for `instance`."""
    return VALUE_TOLERANCE * (1.0 + float(np.abs(instance.loss_array).max()))


def positive_support_moments(instance: Instance) -> tuple[float, float, float]:
    """Return the nominal mean, variance, and minimum loss on the nominal's support."""
    nominal, loss = instance.nominal_array, instance.loss_array
    support = nominal > 0.0
    mean = float(nominal @ loss)
    variance = float(nominal @ (loss - mean) ** 2)
    return mean, variance, float(loss[support].min())


# --- Variance regularization -------------------------------------------------


def test_chi_square_matches_mean_plus_sqrt_radius_variance_in_interior_regime() -> None:
    nominal = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    radius = 0.05
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    mean = torch.sum(nominal * loss)
    variance = torch.sum(nominal * (loss - mean) ** 2)
    expected = mean + torch.sqrt(torch.tensor(radius, dtype=torch.float64) * variance)
    assert torch.allclose(result, expected, atol=1e-12)


@given(instance=instances(), fraction=st.floats(min_value=0.0, max_value=1.0))
def test_chi_square_is_variance_regularized_until_a_scenario_is_truncated(
    instance: Instance, fraction: float
) -> None:
    # The maximizer p_i (1 + (loss_i - mean) sqrt(radius / variance)) stays
    # nonnegative exactly while radius <= variance / (mean - min loss)^2, and
    # spends the whole budget, so the worst case is the variance-regularized
    # mean on that whole interval.
    mean, variance, minimum = positive_support_moments(instance)
    if variance == 0.0:
        return
    radius = fraction * variance / (mean - minimum) ** 2
    nominal, loss, _ = instance.tensors()

    value = ChiSquareAmbiguitySet(nominal, radius).worst_case_expectation(loss)

    expected = mean + math.sqrt(radius * variance)
    assert float(value) == pytest.approx(expected, abs=tolerance(instance))


@given(instance=instances(), radius=radii)
def test_variance_regularization_bounds_the_chi_square_worst_case(
    instance: Instance, radius: float
) -> None:
    # Cauchy-Schwarz: E_q[loss] - mean = E_p[(q/p - 1)(loss - mean)] is at
    # most sqrt(chi^2(q || p) * Var_p(loss)) for every q in the ball.
    mean, variance, _ = positive_support_moments(instance)
    nominal, loss, _ = instance.tensors()

    value = ChiSquareAmbiguitySet(nominal, radius).worst_case_expectation(loss)

    assert float(value) <= mean + math.sqrt(radius * variance) + tolerance(instance)


def test_chi_square_falls_strictly_below_the_variance_bound_once_truncated() -> None:
    nominal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    loss = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    mean = float(nominal @ loss)
    variance = float(nominal @ (loss - mean) ** 2)
    threshold = variance / mean**2
    radii_list = [1.01 * threshold, 1.5 * threshold, 3.0 * threshold]

    values = ChiSquareAmbiguitySet(
        nominal, torch.tensor(radii_list, dtype=torch.float64)
    ).worst_case_expectation(loss)

    for radius, value in zip(radii_list, values.tolist(), strict=True):
        assert value < mean + math.sqrt(radius * variance) - 1e-6


KL_INSTANCES = (
    Instance(
        name="skewed",
        nominal=(0.5, 0.2, 0.2, 0.05, 0.05),
        loss=(0.0, 1.0, 2.0, 3.0, 10.0),
        position=(0.0, 1.0, 2.0, 3.0, 4.0),
    ),
    Instance(
        name="asymmetric",
        nominal=(0.1, 0.2, 0.3, 0.4),
        loss=(0.0, 1.0, 2.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="uniform",
        nominal=(1.0 / 7.0,) * 7,
        loss=(-1.3, 0.4, 2.2, -0.7, 1.1, 3.6, 0.9),
        position=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    ),
)


def kl_expansion(
    instance: Instance, radius: float
) -> tuple[float, float, float, float]:
    r"""Return the terms of the small-radius expansion of the KL worst case.

    The worst case is the exponential tilt $q_t \propto p\,e^{t \ell}$ with
    value $K'(t)$ and divergence $t K'(t) - K(t)$, for the cumulant
    generating function $K$ of $\ell$ under $p$. Substituting
    $K(t) = \mu t + \kappa_2 t^2/2 + \kappa_3 t^3/6 + \kappa_4 t^4/24 +
    O(t^5)$, solving the divergence for $t$ as a series in $\sqrt{2\rho}$,
    and substituting back gives

    $$
    V(\rho) = \mu + \sigma\sqrt{2\rho} + \frac{\kappa_3}{3\sigma^2}\,\rho
        + 2\sqrt{2}\left(\frac{\kappa_4}{24\sigma^3}
        - \frac{\kappa_3^2}{18\sigma^5}\right)\rho^{3/2} + O(\rho^2).
    $$

    Args:
        instance: Problem instance with a nondegenerate loss.
        radius: KL radius $\rho$.

    Returns:
        The mean and the $\rho^{1/2}$, $\rho$, and $\rho^{3/2}$ terms.
    """
    nominal, loss = instance.nominal_array, instance.loss_array
    mean = float(nominal @ loss)
    centered = loss - mean
    variance = float(nominal @ centered**2)
    third = float(nominal @ centered**3)
    fourth = float(nominal @ centered**4) - 3.0 * variance**2
    deviation = math.sqrt(variance)
    return (
        mean,
        deviation * math.sqrt(2.0 * radius),
        third / (3.0 * variance) * radius,
        2.0
        * math.sqrt(2.0)
        * (fourth / (24.0 * deviation**3) - third**2 / (18.0 * deviation**5))
        * radius**1.5,
    )


@pytest.mark.parametrize("instance", KL_INSTANCES, ids=lambda instance: instance.name)
@pytest.mark.parametrize("radius", [1e-2, 1e-3, 1e-4, 1e-5])
def test_kl_follows_its_cumulant_expansion_to_third_order(
    instance: Instance, radius: float
) -> None:
    # The remainder after the rho^(3/2) term is O(rho^2) with a coefficient
    # below 0.14 on these unit-scale instances; each earlier truncation
    # leaves a remainder that is larger by at least a factor of sqrt(rho).
    nominal, loss, _ = instance.tensors()
    mean, first, second, third = kl_expansion(instance, radius)

    value = float(KLAmbiguitySet(nominal, radius).worst_case_expectation(loss))

    remainder = value - (mean + first + second + third)
    assert abs(remainder) <= radius**2
    assert abs(value - (mean + first + second)) > 10.0 * abs(remainder)


# --- Lipschitz and gradient-norm regularization -----------------------------


def _metric_support(num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return midpoint normal quantiles and their Euclidean cost matrix."""
    quantiles = (torch.arange(num_points, dtype=torch.float64) + 0.5) / num_points
    support = torch.special.ndtri(quantiles)
    return support, torch.abs(support.unsqueeze(-1) - support.unsqueeze(-2))


def _empirical_lipschitz_constant(
    loss: torch.Tensor, cost: torch.Tensor
) -> torch.Tensor:
    """Return the largest loss increment per unit of transport on the support."""
    increments = torch.abs(loss.unsqueeze(-1) - loss.unsqueeze(-2))
    separation = cost + torch.eye(cost.shape[-1], dtype=cost.dtype)
    return torch.max(increments / separation)


def test_small_radius_dual_equals_lipschitz_regularized_expectation() -> None:
    # On a metric support the dual is minimized at gamma = Lip(loss) for every
    # radius below the threshold at which some support point's maximizer stops
    # being itself, so the dual value is exactly the linear surrogate
    # E_nominal[loss] + radius * Lip(loss) (Wu, Li and Mao 2025). Here
    # loss(z) = |z| attains its Lipschitz constant 1 between adjacent support
    # points, and the threshold radius is 0.8.
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    cost = torch.abs(support.unsqueeze(-1) - support.unsqueeze(-2))
    loss = torch.abs(support)
    nominal = torch.full_like(support, 0.2)
    radius = 0.2
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.allclose(
        _empirical_lipschitz_constant(loss, cost),
        torch.tensor(1.0, dtype=torch.float64),
    )
    expected = torch.sum(nominal * loss) + radius * 1.0
    assert torch.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("radius", [0.01, 0.1, 1.0, 5.0])
def test_lipschitz_surrogate_upper_bounds_the_dual(radius: float) -> None:
    support, cost = _metric_support(24)
    loss = torch.nn.functional.softplus(support)
    nominal = torch.full_like(support, 1.0 / support.numel())
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

    result = ambiguity_set.worst_case_expectation(loss)

    # softplus is 1-Lipschitz: its derivative is the sigmoid.
    surrogate = torch.sum(nominal * loss) + radius * 1.0
    assert result <= surrogate + 1e-6


def test_lipschitz_surrogate_gap_shrinks_as_the_sample_refines() -> None:
    # The small-radius gap against the surrogate built from the true Lipschitz
    # constant is radius * (Lip - L_n), where L_n is the loss's Lipschitz
    # modulus restricted to the sample. softplus approaches slope 1 only
    # asymptotically, so L_n increases -- and the gap shrinks -- as the sample
    # refines.
    radius = 1e-3
    normalized_gaps = []
    deficiencies = []
    for num_points in (8, 32, 128):
        support, cost = _metric_support(num_points)
        loss = torch.nn.functional.softplus(support)
        nominal = torch.full_like(support, 1.0 / num_points)
        ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=radius)

        result = ambiguity_set.worst_case_expectation(loss)

        surrogate = torch.sum(nominal * loss) + radius * 1.0
        normalized_gaps.append(((surrogate - result) / radius).item())
        deficiencies.append((1.0 - _empirical_lipschitz_constant(loss, cost)).item())

    assert normalized_gaps == sorted(normalized_gaps, reverse=True)
    for measured, predicted in zip(normalized_gaps, deficiencies, strict=True):
        assert abs(measured - predicted) < 1e-9


# A fine grid on [-2, 2] carries the candidate distributions; the nominal
# sits on seven interior grid points, far enough inside that no budget below
# lets mass leave the grid.
GRID_SPACING = 0.004
GRID = torch.arange(-2.0, 2.0 + GRID_SPACING / 2, GRID_SPACING, dtype=torch.float64)
NOMINAL_INDEX = torch.tensor([150, 300, 420, 500, 610, 700, 830])
NOMINAL_WEIGHT = torch.tensor(
    [0.1, 0.2, 0.1, 0.15, 0.2, 0.15, 0.1], dtype=torch.float64
)

# sin(x) + x^2 / 4, whose second derivative 1/2 - sin(x) is bounded by 3/2.
CURVATURE_BOUND = 1.5


def _smooth_loss(points: torch.Tensor) -> torch.Tensor:
    return torch.sin(points) + 0.25 * points**2


def _smooth_loss_derivative(points: torch.Tensor) -> torch.Tensor:
    return torch.cos(points) + 0.5 * points


def _type_two_problem(
    budget: float,
) -> tuple[float, float, float, float]:
    r"""Solve the squared-distance Wasserstein worst case and its two bounds.

    Returns:
        The worst case, the nominal mean, the gradient norm
        $\lVert \ell' \rVert_{L^2(p)}$, and the value of an explicit
        feasible plan: every nominal point moved along $\ell'$ by
        $\sqrt{\delta}\,\ell'(x)/\lVert \ell' \rVert_{L^2(p)}$, rounded to
        the grid towards its start so it never spends more than $\delta$.
    """
    nominal = torch.zeros_like(GRID)
    nominal[NOMINAL_INDEX] = NOMINAL_WEIGHT
    cost = (GRID.unsqueeze(-1) - GRID.unsqueeze(-2)) ** 2
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost=cost, radius=budget)
    value = float(ambiguity_set.worst_case_expectation(_smooth_loss(GRID)))

    points = GRID[NOMINAL_INDEX]
    slope = _smooth_loss_derivative(points)
    mean = float(NOMINAL_WEIGHT @ _smooth_loss(points))
    gradient_norm = float(torch.sqrt(NOMINAL_WEIGHT @ slope**2))
    steps = math.sqrt(budget) * slope / gradient_norm / GRID_SPACING
    rounded = torch.where(steps >= 0.0, torch.floor(steps), torch.ceil(steps))
    target = NOMINAL_INDEX + rounded.long()
    assert float(NOMINAL_WEIGHT @ (GRID[target] - points) ** 2) <= budget
    feasible = float(NOMINAL_WEIGHT @ _smooth_loss(GRID[target]))
    return value, mean, gradient_norm, feasible


@pytest.mark.parametrize("budget", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
def test_type_two_wasserstein_is_sandwiched_by_gradient_norm_regularization(
    budget: float,
) -> None:
    # Upper bound, for every plan with E|y - x|^2 <= budget: a second-order
    # Taylor bound and Cauchy-Schwarz give
    # E[loss(y)] <= mean + sqrt(budget) * ||loss'|| + curvature * budget / 2.
    # Lower bound: the worst case is at least any feasible plan's value.
    value, mean, gradient_norm, feasible = _type_two_problem(budget)

    upper = mean + math.sqrt(budget) * gradient_norm + CURVATURE_BOUND * budget / 2
    assert feasible <= value + 1e-12
    assert value <= upper + 1e-12


@pytest.mark.parametrize("budget", [1e-2, 1e-3])
def test_type_two_wasserstein_slope_is_the_gradient_norm(budget: float) -> None:
    # (V - mean) / sqrt(budget) -> ||loss'||_{L^2(p)} with an O(sqrt(budget))
    # curvature error, as long as the optimal moves, of size sqrt(budget),
    # stay well above the grid spacing.
    value, mean, gradient_norm, _ = _type_two_problem(budget)

    slope = (value - mean) / math.sqrt(budget)
    assert abs(slope - gradient_norm) <= CURVATURE_BOUND * math.sqrt(budget) / 2


# --- Total variation and CVaR ------------------------------------------------


@given(instance=instances(), radius=radii)
def test_total_variation_is_a_mixture_of_the_maximum_and_cvar(
    instance: Instance, radius: float
) -> None:
    nominal, loss, _ = instance.tensors()
    moved = min(radius, 1.0)

    total_variation = TotalVariationAmbiguitySet(nominal, radius)
    cvar = CVaRAmbiguitySet(nominal, alpha=1.0 - moved)

    expected = moved * float(loss.max()) + (1.0 - moved) * float(
        cvar.worst_case_expectation(loss)
    )
    assert float(total_variation.worst_case_expectation(loss)) == pytest.approx(
        expected, abs=tolerance(instance)
    )


# --- Nesting of the divergence balls -----------------------------------------


def _renyi_two_radius(radius: float) -> float:
    """Return the chi-square radius `exp(radius) - 1` of a Renyi-2 radius."""
    return math.expm1(radius) if radius < _LARGEST_FINITE_EXPONENT else math.inf


@given(instance=instances(), radius=radii)
def test_worst_cases_nest_like_their_divergences(
    instance: Instance, radius: float
) -> None:
    nominal, loss, _ = instance.tensors()

    cvar = CVaRAmbiguitySet(nominal, alpha=math.exp(-radius))
    chi_square = ChiSquareAmbiguitySet(nominal, _renyi_two_radius(radius))
    kl = KLAmbiguitySet(nominal, radius)
    total_variation = TotalVariationAmbiguitySet(nominal, math.sqrt(radius / 2.0))

    chain = [
        float(ambiguity_set.worst_case_expectation(loss))
        for ambiguity_set in (cvar, chi_square, kl, total_variation)
    ]
    for smaller, larger in zip(chain, chain[1:], strict=False):
        assert smaller <= larger + tolerance(instance)
