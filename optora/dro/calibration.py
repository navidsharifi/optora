r"""Ambiguity radii calibrated as confidence levels from finite-sample theory.

An ambiguity set centered on the empirical distribution $\hat P_n$ of $n$
i.i.d. observations is a confidence region for the data-generating
distribution once its radius is a quantile of the right limit law. The robust
value $\sup_{q \in \mathcal{U}} \mathbb{E}_q[\ell]$ is then an asymptotically
exact upper confidence bound on the true risk $\mathbb{E}_{P_0}[\ell]$, and
$[-\sup_{q} \mathbb{E}_q[-\ell], \sup_{q} \mathbb{E}_q[\ell]]$ an
asymptotically exact confidence interval for it.

- **Phi-divergence balls** (Duchi, Glynn, and Namkoong 2021; Duchi and
  Namkoong 2021). For a generator smooth at one, $n$ times the profile
  divergence of the true risk converges to $\frac{\varphi''(1)}{2}
  \chi^2_1$, so the radius is

  $$
  \rho_n = \frac{\varphi''(1)}{2} \cdot \frac{\chi^2_{1,\,1-\alpha}}{n}.
  $$

- **Wasserstein balls with quadratic cost** (Blanchet, Kang, and Murthy
  2019). For the cost $c(x, y) = \lVert x - y \rVert_2^2$ on the input
  space, $n$ times the robust Wasserstein profile of the true risk converges
  to $\frac{\mathrm{Var}(\ell)}{\mathbb{E} \lVert \nabla_x \ell
  \rVert_2^2} \chi^2_1$, so the radius is the plug-in

  $$
  \delta_n = \frac{\widehat{\mathrm{Var}}(\ell)}
      {\overline{\lVert \nabla_x \ell \rVert_2^2}} \cdot
      \frac{\chi^2_{1,\,1-\alpha}}{n}.
  $$

Both radii shrink like $1/n$, which is what makes the robust value a
variance-sized correction to the empirical risk rather than a
dimension-dependent one.

A two-sided `confidence` $1 - \alpha$ uses $\chi^2_{1,\,1-\alpha} =
z_{1-\alpha/2}^2$; a one-sided upper bound at level $1 - \alpha$ uses
$z_{1-\alpha}^2$, since each side of the interval fails with probability
$\alpha / 2$.

A phi-divergence ball can instead be calibrated to cover a $d$-dimensional
parameter simultaneously, such as every component of a vector risk or the
minimizer of a risk over $d$ parameters, which replaces $\chi^2_1$ by
$\chi^2_d$ (Duchi, Glynn, and Namkoong 2021). Every quantile is computed
on the host in double precision, from the standard library's normal
quantile for one degree of freedom and from the closed-form $\chi^2_d$
survival function otherwise, so no helper here touches a device.
"""

import math
from statistics import NormalDist

import torch

_KL_CURVATURE = 1.0
_CHI_SQUARE_CURVATURE = 2.0

_QUANTILE_MAX_ITERATIONS = 200
_QUANTILE_RELATIVE_TOLERANCE = 4.0 * 2.0**-52


def _chi_square_survival(value: float, degrees_of_freedom: int) -> float:
    r"""Return the upper tail $\Pr(\chi^2_d > x)$ of an integer-order chi-square.

    With $y = x / 2$ the tail is the regularized upper incomplete gamma
    function $Q(d / 2, y)$, which at a half-integer order is a finite sum
    of positive terms (Abramowitz and Stegun 1964, 26.4.4 and 26.4.5):

    $$
    Q(k, y) = \sum_{j=0}^{k-1} \frac{e^{-y} y^j}{j!}, \qquad
    Q\big(k + \tfrac{1}{2}, y\big) = \operatorname{erfc}(\sqrt{y})
    + \sum_{j=0}^{k-1} \frac{e^{-y} y^{j + 1/2}}{\Gamma(j + \tfrac{3}{2})}.
    $$

    No term cancels another, so the tail keeps its relative precision far
    into it, and each term is formed in log space, so neither $e^{-y}$ nor
    $y^j$ under- or overflows on its own.

    Args:
        value: Point $x > 0$ at which the tail is evaluated.
        degrees_of_freedom: Number of degrees of freedom $d \ge 1$.

    Returns:
        The tail probability, in $[0, 1]$.
    """
    half = 0.5 * value
    offset = 0.5 * (degrees_of_freedom % 2)
    tail = math.erfc(math.sqrt(half)) if offset else 0.0
    log_half = math.log(half)
    for index in range(degrees_of_freedom // 2):
        order = index + offset
        tail += math.exp(order * log_half - half - math.lgamma(order + 1.0))
    return min(tail, 1.0)


def _chi_square_cdf(value: float, degrees_of_freedom: int) -> float:
    r"""Return the lower tail $\Pr(\chi^2_d \le x)$, accurate where it is small.

    With $y = x / 2$ and $a = d / 2$ the lower tail is the regularized lower
    incomplete gamma function, summed from its series of positive terms
    (Abramowitz and Stegun 1964, 6.5.29),

    $$
    P(a, y) = \frac{e^{-y} y^a}{\Gamma(a + 1)}
    \sum_{k=0}^{\infty} \frac{y^k}{(a + 1) \cdots (a + k)},
    $$

    which keeps the relative precision that $1 - Q(a, y)$ loses once the
    lower tail is small. The series converges geometrically for $y < a + 1$
    (Press et al. 2007, section 6.2); above that the lower tail is at least
    about one half, so it is formed as $1 - Q(a, y)$ without loss.

    Args:
        value: Point $x > 0$ at which the tail is evaluated.
        degrees_of_freedom: Number of degrees of freedom $d \ge 1$.

    Returns:
        The lower-tail probability, in $[0, 1]$.
    """
    half = 0.5 * value
    shape = 0.5 * degrees_of_freedom
    if half >= shape + 1.0:
        return 1.0 - _chi_square_survival(value, degrees_of_freedom)
    term = total = 1.0
    index = 0
    while term > 0.25 * _QUANTILE_RELATIVE_TOLERANCE * total:
        index += 1
        term *= half / (shape + index)
        total += term
    prefactor = math.exp(shape * math.log(half) - half - math.lgamma(shape + 1.0))
    return min(prefactor * total, 1.0)


def _chi_square_density(value: float, degrees_of_freedom: int) -> float:
    r"""Return the density of $\chi^2_d$ at `value > 0`."""
    shape = 0.5 * degrees_of_freedom
    return math.exp(
        (shape - 1.0) * math.log(value)
        - 0.5 * value
        - shape * math.log(2.0)
        - math.lgamma(shape)
    )


def _chi_square_inverse_cdf(confidence: float, degrees_of_freedom: int) -> float:
    r"""Return the quantile $\chi^2_{d,\,c}$, the $x$ with $\Pr(\chi^2_d \le x) = c$.

    The equation is solved on whichever tail is the smaller, the lower one
    $P(x) = c$ below the median and the upper one $Q(x) = 1 - c$ above it,
    so the residual keeps its relative precision at any level. The residual
    is taken between logarithms, $\log P(x) - \log c$ or $\log(1 - c) -
    \log Q(x)$, both increasing in $x$ and close to linear along the power
    law and exponential decay of the two tails, where Newton steps on the
    probabilities themselves would only converge linearly. Below the median
    the bound $P(a, y) \le y^a / \Gamma(a + 1)$ gives a positive lower end
    of the bracket; the upper end is found by doubling from the
    Wilson-Hilferty approximation (Wilson and Hilferty 1931). Each Newton
    step that would leave the bracket is replaced by a bisection step,
    geometric while the bracket spans more than a factor of two.

    Args:
        confidence: Level $c \in (0, 1)$.
        degrees_of_freedom: Number of degrees of freedom $d \ge 2$.

    Returns:
        The quantile, to a few ulps.
    """
    shape = 0.5 * degrees_of_freedom
    lower_tail = confidence < 0.5

    def log_residual(value: float) -> tuple[float, float]:
        if lower_tail:
            tail = _chi_square_cdf(value, degrees_of_freedom)
            excess = math.log(tail) - math.log(confidence) if tail > 0.0 else -math.inf
        else:
            tail = _chi_square_survival(value, degrees_of_freedom)
            excess = (
                math.log1p(-confidence) - math.log(tail) if tail > 0.0 else math.inf
            )
        return excess, tail

    lower = 0.0
    if lower_tail:
        log_bound = (math.log(confidence) + math.lgamma(shape + 1.0)) / shape
        lower = 2.0 * math.exp(log_bound)
    scale = 2.0 / (9.0 * degrees_of_freedom)
    normal_quantile = NormalDist().inv_cdf(confidence)
    cube_root = max(1.0 - scale + normal_quantile * math.sqrt(scale), 0.1)
    root = upper = max(degrees_of_freedom * cube_root**3, lower)
    while log_residual(upper)[0] < 0.0:
        lower, upper = upper, 2.0 * upper
    for _ in range(_QUANTILE_MAX_ITERATIONS):
        excess, tail = log_residual(root)
        if excess < 0.0:
            lower = root
        else:
            upper = root
        density = _chi_square_density(root, degrees_of_freedom)
        step = excess * tail / density if density > 0.0 else math.nan
        candidate = root - step
        if not lower < candidate < upper:
            wide = 0.0 < 2.0 * lower < upper
            midpoint = math.sqrt(lower) * math.sqrt(upper)
            candidate = midpoint if wide else 0.5 * (lower + upper)
        tolerance = _QUANTILE_RELATIVE_TOLERANCE * candidate
        if abs(candidate - root) <= tolerance or upper - lower <= tolerance:
            return candidate
        root = candidate
    return root


def _chi_square_quantile(
    confidence: float, one_sided: bool, degrees_of_freedom: int = 1
) -> float:
    r"""Return the chi-square quantile a calibrated radius scales with.

    Args:
        confidence: Coverage level, in $(0, 1)$ when two-sided and in
            $[1/2, 1)$ when one-sided.
        one_sided: Whether the coverage is that of the upper bound alone,
            which is only defined for one degree of freedom.
        degrees_of_freedom: Dimension $d \ge 1$ of the parameter the region
            covers.

    Returns:
        $z_{(1 + c)/2}^2 = \chi^2_{1,\,c}$ for a two-sided level $c$ and
        $d = 1$, $z_c^2$ for a one-sided one, and $\chi^2_{d,\,c}$ for
        $d \ge 2$.

    Raises:
        ValueError: If `confidence` is outside its range,
            `degrees_of_freedom` is not a positive integer, or `one_sided`
            is set with more than one degree of freedom.
    """
    if (
        isinstance(degrees_of_freedom, bool)
        or not isinstance(degrees_of_freedom, int)
        or degrees_of_freedom < 1
    ):
        raise ValueError(
            "degrees_of_freedom must be a positive integer, "
            f"got {degrees_of_freedom!r}."
        )
    if one_sided and degrees_of_freedom > 1:
        raise ValueError(
            "one_sided is only defined for one degree of freedom, "
            f"got degrees_of_freedom={degrees_of_freedom}."
        )
    if one_sided and not 0.5 <= confidence < 1.0:
        raise ValueError(
            f"one-sided confidence must lie in [0.5, 1), got {confidence}."
        )
    if not one_sided and not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}.")
    if degrees_of_freedom > 1:
        return _chi_square_inverse_cdf(confidence, degrees_of_freedom)
    level = confidence if one_sided else 0.5 * (1.0 + confidence)
    return NormalDist().inv_cdf(level) ** 2


def _validate_num_samples(num_samples: int) -> None:
    """Raise if `num_samples` is not a positive sample size."""
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least one, got {num_samples}.")


def phi_divergence_radius(
    num_samples: int,
    confidence: float,
    curvature: float,
    one_sided: bool = False,
    degrees_of_freedom: int = 1,
) -> float:
    r"""Return the phi-divergence radius that makes the ball a confidence region.

    Sets $\rho_n = \frac{\varphi''(1)}{2} \chi^2_{d,\,1-\alpha} / n$
    (Duchi, Glynn, and Namkoong 2021). With the default $d = 1$ the ball is
    a confidence region for a scalar risk; with $d > 1$ it covers a
    $d$-dimensional parameter simultaneously, at the price of a larger
    radius. The divergence is the one optora
    computes, $D_\varphi(q \,\|\, p) = \sum_i p_i \varphi(q_i / p_i)$, so
    the curvature is that of its generator: $1$ for KL
    ($\varphi(t) = t \log t - t + 1$) and $2$ for chi-square
    ($\varphi(t) = (t - 1)^2$). Papers that normalize $\varphi''(1) = 2$ or
    $\varphi(t) = \frac{1}{2}(t - 1)^2$ state radii that differ from these
    by a constant factor. Total variation is not smooth at one and has no
    such calibration.

    Args:
        num_samples: Number $n$ of i.i.d. observations the empirical nominal
            is built from, at least one.
        confidence: Target coverage $1 - \alpha$ of the interval, or of the
            upper bound alone when `one_sided` is set.
        curvature: Second derivative $\varphi''(1)$ of the generator at one,
            positive and finite.
        one_sided: Whether `confidence` is the coverage of the upper
            confidence bound $\sup_q \mathbb{E}_q[\ell]$ alone rather than of
            the two-sided interval. Only defined for one degree of freedom.
        degrees_of_freedom: Dimension $d \ge 1$ of the parameter the
            confidence region covers.

    Returns:
        The calibrated radius, a Python float.

    Raises:
        ValueError: If `num_samples` is less than one, `confidence` is out of
            range, `curvature` is not positive and finite,
            `degrees_of_freedom` is not a positive integer, or `one_sided`
            is set with more than one degree of freedom.
    """
    _validate_num_samples(num_samples)
    if not (math.isfinite(curvature) and curvature > 0.0):
        raise ValueError(f"curvature must be positive and finite, got {curvature}.")
    quantile = _chi_square_quantile(confidence, one_sided, degrees_of_freedom)
    return 0.5 * curvature * quantile / num_samples


def kl_radius(
    num_samples: int,
    confidence: float,
    one_sided: bool = False,
    degrees_of_freedom: int = 1,
) -> float:
    r"""Return the calibrated radius of a `KLAmbiguitySet`.

    The radius is $\chi^2_{d,\,1-\alpha} / (2n)$.

    Args:
        num_samples: Number of i.i.d. observations the empirical nominal is
            built from.
        confidence: Target coverage of the interval, or of the upper bound
            alone when `one_sided` is set.
        one_sided: Whether `confidence` is the coverage of the upper bound
            alone.
        degrees_of_freedom: Dimension of the parameter the confidence region
            covers.

    Returns:
        `phi_divergence_radius` with the KL curvature $\varphi''(1) = 1$.

    Raises:
        ValueError: If `num_samples`, `confidence`, or `degrees_of_freedom`
            is out of range.
    """
    return phi_divergence_radius(
        num_samples, confidence, _KL_CURVATURE, one_sided, degrees_of_freedom
    )


def chi_square_radius(
    num_samples: int,
    confidence: float,
    one_sided: bool = False,
    degrees_of_freedom: int = 1,
) -> float:
    r"""Return the calibrated radius of a `ChiSquareAmbiguitySet`.

    The radius is $\chi^2_{d,\,1-\alpha} / n$. In the interior regime the
    chi-square worst case is $\bar\ell + \sqrt{\rho\, s_n^2}$, so at $d = 1$
    this radius reproduces the normal interval
    $\bar\ell \pm z_{1-\alpha/2}\, s_n / \sqrt{n}$ exactly.

    Args:
        num_samples: Number of i.i.d. observations the empirical nominal is
            built from.
        confidence: Target coverage of the interval, or of the upper bound
            alone when `one_sided` is set.
        one_sided: Whether `confidence` is the coverage of the upper bound
            alone.
        degrees_of_freedom: Dimension of the parameter the confidence region
            covers.

    Returns:
        `phi_divergence_radius` with the chi-square curvature
        $\varphi''(1) = 2$.

    Raises:
        ValueError: If `num_samples`, `confidence`, or `degrees_of_freedom`
            is out of range.
    """
    return phi_divergence_radius(
        num_samples, confidence, _CHI_SQUARE_CURVATURE, one_sided, degrees_of_freedom
    )


def wasserstein_radius(
    loss: torch.Tensor,
    input_gradient: torch.Tensor,
    confidence: float,
    one_sided: bool = False,
) -> torch.Tensor:
    r"""Return the robust-Wasserstein-profile radius for the quadratic transport cost.

    Sets $\delta_n = \chi^2_{1,\,1-\alpha}\, \widehat{\mathrm{Var}}(\ell) /
    (n\, \overline{\lVert \nabla_x \ell \rVert_2^2})$, the plug-in quantile
    of the limit law of the robust Wasserstein profile of the true risk
    (Blanchet, Kang, and Murthy 2019), with the plug-in (biased) variance.
    To first order the worst case over that ball is $\bar\ell +
    \sqrt{\delta_n\, \overline{\lVert \nabla_x \ell \rVert_2^2}} = \bar\ell
    + z_{1-\alpha/2}\, s_n / \sqrt{n}$, the same normal interval the
    calibrated phi-divergence balls give (Gao, Chen, and Kleywegt 2024).

    The calibration is for transport over the whole input space with cost
    $\lVert x - y \rVert_2^2$. A `WassersteinAmbiguitySet` whose support is
    the observations alone can only move mass among them, so its worst case
    under this radius is no larger, and its coverage no higher, than the
    calibrated one.

    Args:
        loss: Per-observation losses of shape `(..., n)`. Leading dimensions
            are independent samples, calibrated in one call.
        input_gradient: Gradient of each observation's loss with respect to
            that observation's input, of shape `(..., n, d)`.
        confidence: Target coverage of the interval, or of the upper bound
            alone when `one_sided` is set.
        one_sided: Whether `confidence` is the coverage of the upper bound
            alone.

    Returns:
        The calibrated radius of shape `(...)`, in the dtype and on the
        device of `loss`. It is `0` where the losses are constant and `inf`
        where they vary but every input gradient vanishes, since no
        transport of the inputs then moves the loss.

    Raises:
        ValueError: If `confidence` is out of range, `loss` is
            zero-dimensional, or `input_gradient`'s shape is not `loss`'s
            followed by a feature dimension.
    """
    quantile = _chi_square_quantile(confidence, one_sided)
    if loss.ndim == 0 or input_gradient.shape[:-1] != loss.shape:
        raise ValueError(
            "loss must have shape (..., n) and input_gradient shape (..., n, d), "
            f"got {tuple(loss.shape)} and {tuple(input_gradient.shape)}."
        )
    num_samples = loss.shape[-1]
    variance = torch.var(loss, dim=-1, correction=0)
    energy = torch.mean(torch.sum(input_gradient**2, dim=-1), dim=-1).to(loss.dtype)
    radius = quantile * variance / (num_samples * energy)
    return torch.where(variance > 0.0, radius, torch.zeros_like(radius))
