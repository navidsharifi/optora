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
$\alpha / 2$. The quantile is computed in double precision from the
standard library's normal quantile, so no helper here touches a device.
"""

import math
from statistics import NormalDist

import torch

_KL_CURVATURE = 1.0
_CHI_SQUARE_CURVATURE = 2.0


def _chi_square_quantile(confidence: float, one_sided: bool) -> float:
    r"""Return the squared normal quantile a calibrated radius scales with.

    Args:
        confidence: Coverage level, in $(0, 1)$ when two-sided and in
            $[1/2, 1)$ when one-sided.
        one_sided: Whether the coverage is that of the upper bound alone.

    Returns:
        $z_{(1 + c)/2}^2 = \chi^2_{1,\,c}$ for a two-sided level $c$, or
        $z_c^2$ for a one-sided one.

    Raises:
        ValueError: If `confidence` is outside its range.
    """
    if one_sided and not 0.5 <= confidence < 1.0:
        raise ValueError(
            f"one-sided confidence must lie in [0.5, 1), got {confidence}."
        )
    if not one_sided and not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}.")
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
) -> float:
    r"""Return the phi-divergence radius that makes the ball a confidence region.

    Sets $\rho_n = \frac{\varphi''(1)}{2} \chi^2_{1,\,1-\alpha} / n$
    (Duchi, Glynn, and Namkoong 2021). The divergence is the one optora
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
            the two-sided interval.

    Returns:
        The calibrated radius, a Python float.

    Raises:
        ValueError: If `num_samples` is less than one, `confidence` is out of
            range, or `curvature` is not positive and finite.
    """
    _validate_num_samples(num_samples)
    if not (math.isfinite(curvature) and curvature > 0.0):
        raise ValueError(f"curvature must be positive and finite, got {curvature}.")
    return 0.5 * curvature * _chi_square_quantile(confidence, one_sided) / num_samples


def kl_radius(num_samples: int, confidence: float, one_sided: bool = False) -> float:
    r"""Return the calibrated radius of a `KLAmbiguitySet`.

    The radius is $\chi^2_{1,\,1-\alpha} / (2n)$.

    Args:
        num_samples: Number of i.i.d. observations the empirical nominal is
            built from.
        confidence: Target coverage of the interval, or of the upper bound
            alone when `one_sided` is set.
        one_sided: Whether `confidence` is the coverage of the upper bound
            alone.

    Returns:
        `phi_divergence_radius` with the KL curvature $\varphi''(1) = 1$.

    Raises:
        ValueError: If `num_samples` or `confidence` is out of range.
    """
    return phi_divergence_radius(num_samples, confidence, _KL_CURVATURE, one_sided)


def chi_square_radius(
    num_samples: int, confidence: float, one_sided: bool = False
) -> float:
    r"""Return the calibrated radius of a `ChiSquareAmbiguitySet`.

    The radius is $\chi^2_{1,\,1-\alpha} / n$. In the interior regime the
    chi-square worst case is $\bar\ell + \sqrt{\rho\, s_n^2}$, so this
    radius reproduces the normal interval
    $\bar\ell \pm z_{1-\alpha/2}\, s_n / \sqrt{n}$ exactly.

    Args:
        num_samples: Number of i.i.d. observations the empirical nominal is
            built from.
        confidence: Target coverage of the interval, or of the upper bound
            alone when `one_sided` is set.
        one_sided: Whether `confidence` is the coverage of the upper bound
            alone.

    Returns:
        `phi_divergence_radius` with the chi-square curvature
        $\varphi''(1) = 2$.

    Raises:
        ValueError: If `num_samples` or `confidence` is out of range.
    """
    return phi_divergence_radius(
        num_samples, confidence, _CHI_SQUARE_CURVATURE, one_sided
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
