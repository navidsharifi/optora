# Wasserstein-DRO and Lipschitz regularization

Introductions to Wasserstein-DRO often state that it is equivalent to
regularization. This page checks the equivalence numerically, because the
statement is easy to misread. It does not say that the robust value is
approximately a penalty, and it does not allow an arbitrary norm as the
penalty. It concerns the Lipschitz modulus of the loss, measured in the
same ground metric that defines the transport cost.

For a type-1 ball of radius $\rho$ around a nominal $\hat{P}$ and a loss
$\ell$ that is Lipschitz with respect to the ground metric, the
Kantorovich-Rubinstein duality bounds the worst case by
$\mathbb{E}_{\hat{P}}[\ell] + \rho \cdot \mathrm{Lip}(\ell)$. The bound
holds with equality when the space is unbounded and the loss attains its
Lipschitz modulus as a slope at infinity, as the softplus loss used below
does. In the strong dual of Gao and Kleywegt (2022), the inner supremum
$\sup_z \{\ell(z) - \gamma \lvert z - \hat{z} \rvert\}$ is then infinite
for $\gamma < \mathrm{Lip}(\ell)$ and equal to $\ell(\hat{z})$ for
$\gamma \ge \mathrm{Lip}(\ell)$, so the dual minimum is attained at
$\gamma = \mathrm{Lip}(\ell)$ and gives

$$
\sup_{q \,:\, W(q, \hat{P}) \le \rho} \mathbb{E}_q[\ell]
\;=\;
\mathbb{E}_{\hat{P}}[\ell] \;+\; \rho \cdot \mathrm{Lip}(\ell).
$$

Wu, Li and Mao[^wlm] prove exact equivalences of this kind in a much
broader setting than the one used here, for general Wasserstein balls
and general decision criteria. This page checks the finite-support
special case that Optora solves.

## What the discrete dual computes

[`WassersteinAmbiguitySet`](../api/dro/wasserstein_dro.md) has no notion
of a Lipschitz constant. It solves the exact discrete dual

$$
\inf_{\gamma \ge 0} \;
\gamma \rho
+ \sum_i \hat{p}_i \max_j \big( \ell_j - \gamma \, c_{ij} \big),
$$

and the regularization identity follows from it. Since $c_{ii} = 0$, once
$\gamma$ is large enough that no transport is profitable, the inner
maximum is attained at $j = i$ for every $i$. The objective then reduces
to $\mathbb{E}_{\hat{P}}[\ell] + \gamma\rho$, which is strictly increasing
in $\gamma$. The smallest $\gamma$ at which this happens is

$$
L_n = \max_{i \ne j} \frac{|\ell_i - \ell_j|}{c_{ij}},
$$

which is the Lipschitz modulus of the loss *restricted to the sample*.
The minimizer therefore never lies above $L_n$, and for every radius
below a positive threshold it lies exactly *at* $L_n$, which gives

$$
\sup_{q \,:\, W(q, \hat{P}) \le \rho} \mathbb{E}_q[\ell]
= \mathbb{E}_{\hat{P}}[\ell] + \rho \, L_n .
$$

This is the equivalence on a finite support. Robustifying the empirical
risk over a small Wasserstein ball is the same as penalizing the
empirical Lipschitz modulus, with $\rho$ as the regularization weight.
It differs from the population statement above only by the gap between
$L_n$ and $\mathrm{Lip}(\ell)$.

## The two ways the gap closes

If the surrogate is built with the *true* constant, the gap splits into a
term that vanishes as the sample refines and a term that vanishes as the
ball shrinks,

$$
\underbrace{\mathbb{E}_{\hat{P}}[\ell] + \rho \, \mathrm{Lip}(\ell)}_{\text{surrogate}}
\;-\;
\underbrace{\sup_{q} \mathbb{E}_q[\ell]}_{\text{exact dual}}
\;=\;
\rho \big( \mathrm{Lip}(\ell) - L_n \big)
\;+\;
\underbrace{\text{concavity correction}}_{\to\, 0 \text{ as } \rho \to 0}.
$$

| Limit | What closes the gap | What the script plots |
| --- | --- | --- |
| $\rho \to 0$ | the exact value is concave in $\rho$ with tangent $\mathbb{E}_{\hat{P}}[\ell] + \rho L_n$ at the origin, and it equals this tangent below a threshold radius, so the correction vanishes | left panel, where the normalized gap flattens onto $\mathrm{Lip}(\ell) - L_n$ |
| $n \to \infty$ | a larger sample contains steeper secants, so $L_n \uparrow \mathrm{Lip}(\ell)$ | right panel, where that plateau decreases toward zero |

The script divides the gap by $\rho$ throughout. Without the division the
raw gap would shrink with $\rho$ for any loss, and the small-radius panel
would show nothing specific to the Lipschitz modulus.

## Why `softplus`

The loss is $\ell(z) = \log(1 + e^z)$ on a deterministic quantile
discretization of $\mathcal{N}(0, 1)$, with $c_{ij} = |z_i - z_j|$.

It is exactly $1$-Lipschitz, because its derivative is the sigmoid, whose
supremum is $1$. The sigmoid is strictly below $1$ everywhere, so the
slope $1$ is approached only as $z \to \infty$ and never attained. Every
secant slope is therefore below $1$, so $L_n < 1$ at every finite sample
size and the sample-size axis has a nonzero gap to show.

!!! note "A piecewise-linear loss would hide the effect"

    Take $\ell(z) = |z|$ instead. Any two support points on the same side
    of the origin attain the full slope, so $L_n = \mathrm{Lip}$ as soon
    as the sample has two such points, and the small-radius gap is zero
    to machine precision. That case is better suited to a unit test than to
    a plot, and it is tested in
    `tests/dro/test_equivalences.py::test_small_radius_dual_equals_lipschitz_regularized_expectation`.
    The same file covers the type-2 (squared-distance) counterpart, where
    the radius acts as a gradient-norm penalty instead.

## Reading the output

The printed table gives $L_n$, the measured normalized gap and the
predicted deficiency $1 - L_n$ side by side. They agree to every printed
digit at every sample size, and the script asserts agreement within
$5 \times 10^{-3}$. The figure shows the result in two ways.

- **Left panel.** The normalized gap against the radius, one curve per
  sample size, each with a dotted line at its own $1 - L_n$. Each curve
  lies on its dotted line below a threshold radius and rises above it at
  larger radii, where the dual minimizer $\gamma^\star$ moves below $L_n$
  and the worst case grows more slowly than linearly in $\rho$. The
  threshold shrinks as $n$ grows, because it is the transport cost of the
  first profitable move, which falls as the support becomes denser.
- **Right panel.** The small-radius plateau against the sample size, on
  log-log axes, with the closed-form $1 - L_n$ overlaid. The two lines
  coincide, which shows that the exact dual value equals the regularized
  objective $\mathbb{E}_{\hat{P}}[\ell] + \rho L_n$ at this radius, up to
  floating-point rounding.

!!! warning "The dual is piecewise linear, so Optora bisects it"

    The objective above is a nonnegative combination of maxima of affine
    functions of $\gamma$, so it is convex and piecewise linear, and its
    minimizer generically lies at a kink. A fixed-step gradient method
    cannot make the gradient norm small there. It oscillates across the
    kink and never passes a gradient-norm stopping test. `WassersteinAmbiguitySet`
    therefore bisects the sign change of the derivative, which is
    monotone in $\gamma$, in a fixed number of steps set by the dtype.
    The worst case is then read off the primal transport plan, so the
    value is exact to working precision and the measured gap matches
    $1 - L_n$ up to rounding.

## The dual's kink at $L_n$

The equivalence is a statement about the minimizer of the dual, and the
animation follows that minimizer as the radius shrinks over four and a
half decades. Subtracting $\mathbb{E}_{\hat{P}}[\ell]$ and dividing by
$\rho$ gives the curve in the left panel a fixed shape. To the right of
$L_n$, where no transport is profitable, it is the line $\gamma$. To the
left it rises steeply, by a term that scales as $1/\rho$. For a large
radius this rise is gentle and the minimizer lies below $L_n$. As the
radius shrinks the rise steepens and the minimizer is fixed at the kink.
The middle panel shows the same behavior from the primal side. By the
envelope theorem $\mathrm{d}V/\mathrm{d}\rho$ equals $\gamma^\star$, and
the animation takes it from autograd. It settles at $L_n$ for every
sample size, below the true constant. The right panel is the normalized
gap of the static figure, with markers moving along it.

<figure class="figure" markdown="span">
  ![The normalized Wasserstein dual in gamma, the derivative of the worst case in the radius, and the normalized gap, as the radius shrinks](../assets/figures/lipschitz-dual-light.gif#only-light){ loading=lazy }
  ![The normalized Wasserstein dual in gamma, the derivative of the worst case in the radius, and the normalized gap, as the radius shrinks](../assets/figures/lipschitz-dual-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the normalized dual objective at
  $n = 64$, with its minimizer, the sample modulus $L_n$ dotted and the
  true Lipschitz constant dashed. The middle panel shows
  $\mathrm{d}V/\mathrm{d}\rho$ for three sample sizes, each settling on
  its own $L_n$. The right panel shows the normalized gap, which flattens
  at $\mathrm{Lip} - L_n$.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Gap between the Lipschitz surrogate and the exact Wasserstein worst case, per unit radius, against the radius and the sample size](../assets/figures/05_wasserstein_lipschitz_equivalence-light.png#only-light){ loading=lazy }
  ![Gap between the Lipschitz surrogate and the exact Wasserstein worst case, per unit radius, against the radius and the sample size](../assets/figures/05_wasserstein_lipschitz_equivalence-dark.png#only-dark){ loading=lazy }
  <figcaption>In the left panel, the excess of the surrogate over the exact value per unit radius flattens at the Lipschitz deficiency $\mathrm{Lip} - L_n$ as the radius shrinks. The right panel shows that this deficiency vanishes as the sample refines.</figcaption>
</figure>

## Source

```py title="tutorials/05_wasserstein_lipschitz_equivalence.py"
--8<-- "tutorials/05_wasserstein_lipschitz_equivalence.py"
```

[^wlm]:
    Qinyu Wu, Jonathan Yu-Meng Li and Tiantian Mao, "On Generalization and
    Regularization via Wasserstein Distributionally Robust Optimization",
    *Management Science* (2025).
    [arXiv:2212.05716](https://arxiv.org/abs/2212.05716).
