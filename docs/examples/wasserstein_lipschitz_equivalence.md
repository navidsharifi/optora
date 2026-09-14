# Wasserstein-DRO and Lipschitz regularization

Every introduction to Wasserstein-DRO eventually says some version of "and
this is equivalent to regularization". I wanted to watch that equivalence
happen on real numbers instead of taking it on trust, because the statement
is easy to state and surprisingly easy to misread: it is *not* that the
robust value is approximately a penalty, and it is *not* that the penalty
is whatever norm your favourite paper happens to use. It is a statement
about the Lipschitz modulus of the loss, measured in the same ground metric
that defines the transport cost.

The clean version, for a type-1 ball of radius $\rho$ around a nominal
$\hat{P}$ and a loss $\ell$ that is Lipschitz with respect to the ground
metric:

$$
\sup_{q \,:\, W(q, \hat{P}) \le \rho} \mathbb{E}_q[\ell]
\;=\;
\mathbb{E}_{\hat{P}}[\ell] \;+\; \rho \cdot \mathrm{Lip}(\ell)
$$

Wu, Li and Mao[^wlm] prove this in a much broader setting than the one used
here — general Wasserstein balls, general decision criteria, and an
equivalence that is *exact* rather than an upper bound. This page checks
the humble finite-support special case that Optora actually solves.

## What the discrete dual is really doing

[`WassersteinAmbiguitySet`](../api/dro/wasserstein_dro.md) does not know
anything about Lipschitz constants. It solves the exact discrete dual

$$
\inf_{\gamma \ge 0} \;
\gamma \rho
+ \sum_i \hat{p}_i \max_j \big( \ell_j - \gamma \, c_{ij} \big)
$$

and the regularization identity falls out of it. Since $c_{ii} = 0$, once
$\gamma$ is large enough that no point can profitably be transported
anywhere, the inner maximum is attained at $j = i$ for every $i$, and the
whole objective collapses to $\mathbb{E}_{\hat{P}}[\ell] + \gamma\rho$ —
strictly increasing in $\gamma$. The smallest $\gamma$ at which that
happens is

$$
L_n = \max_{i \ne j} \frac{|\ell_i - \ell_j|}{c_{ij}}
$$

which is exactly the Lipschitz modulus of the loss *restricted to the
sample*. So the minimizer never sits above $L_n$, and for every radius
below a positive threshold it sits exactly *at* $L_n$, giving

$$
\sup_{q \,:\, W(q, \hat{P}) \le \rho} \mathbb{E}_q[\ell]
= \mathbb{E}_{\hat{P}}[\ell] + \rho \, L_n .
$$

That is the whole equivalence: robustifying the empirical risk over a small
Wasserstein ball *is* penalizing the empirical Lipschitz modulus, with
$\rho$ as the regularization weight. The only thing separating it from the
population statement above is the gap between $L_n$ and
$\mathrm{Lip}(\ell)$.

## The two ways the gap closes

Writing the surrogate with the *true* constant, the gap decomposes into a
term that vanishes as the sample refines and a term that vanishes as the
ball shrinks:

$$
\underbrace{\mathbb{E}_{\hat{P}}[\ell] + \rho \, \mathrm{Lip}(\ell)}_{\text{surrogate}}
\;-\;
\underbrace{\sup_{q} \mathbb{E}_q[\ell]}_{\text{exact dual}}
\;=\;
\rho \big( \mathrm{Lip}(\ell) - L_n \big)
\;+\;
\underbrace{\text{concavity correction}}_{\to\, 0 \text{ as } \rho \to 0}
$$

| Limit | What closes the gap | What the script plots |
| --- | --- | --- |
| $\rho \to 0$ | the exact value is concave in $\rho$ and the surrogate is its tangent at the origin, so the correction dies | left panel: the normalized gap flattens onto $\mathrm{Lip}(\ell) - L_n$ |
| $n \to \infty$ | the sample sees steeper and steeper secants, so $L_n \uparrow \mathrm{Lip}(\ell)$ | right panel: that plateau falls toward zero |

The script divides the gap by $\rho$ throughout. Dividing is the honest
choice: the raw gap shrinks with $\rho$ no matter what the loss looks like,
which would make the small-radius panel a tautology rather than a test.

## Why `softplus`

The loss is $\ell(z) = \log(1 + e^z)$ on a deterministic quantile
discretization of $\mathcal{N}(0, 1)$, with $c_{ij} = |z_i - z_j|$.

It is exactly $1$-Lipschitz — its derivative is the sigmoid, whose supremum
is $1$ — but that slope is only ever *approached*, never attained on a
bounded set. So $L_n < 1$ strictly at every finite sample size, and the
sample-size axis has something to show.

!!! note "A piecewise-linear loss would hide the effect"

    Take $\ell(z) = |z|$ instead. Any two support points on the same side
    of the origin already realize the full slope, so $L_n = \mathrm{Lip}$
    at essentially any sample size and the small-radius gap is zero to
    machine precision. That is a much better *unit test* than a plot — and
    it is one, in
    `tests/dro/test_wasserstein_dro.py::test_small_radius_dual_equals_lipschitz_regularized_expectation`.

## Reading the output

The printed table gives $L_n$, the measured normalized gap, and the
predicted deficiency $1 - L_n$ side by side; they agree to about $10^{-3}$
across every sample size. The figure says the same thing twice:

- **Left**: the normalized gap against radius, one curve per sample size,
  each with a dotted line at its own $1 - L_n$. The curves sit on their
  dotted lines for roughly four decades of radius and only peel upward once
  the ball is large enough to start saturating at $\max_i \ell_i$ — the
  regime where robustness stops behaving like a linear penalty at all.
- **Right**: the small-radius plateau against sample size, on log-log axes,
  with the closed-form $1 - L_n$ overlaid. The two lines are
  indistinguishable, which is the real claim: the solver is not
  approximating the regularized problem, it *is* solving it.

!!! warning "The dual is piecewise linear, so `converged` stays `False`"

    The objective above is a minimum of finitely many affine functions of
    $\gamma$, and its minimizer is a kink. A fixed-step gradient descent
    cannot make the gradient norm small there — it oscillates across the
    kink forever and reports `converged=False` no matter how long it runs.
    That is fine here, and worth understanding rather than working around:
    the slopes on both sides of the kink are $O(\rho)$ in the small-radius
    regime, so the *value* is accurate to roughly $10^{-3}$ even though the
    *gradient* never is. It is also why an occasional measured gap comes
    out a hair negative when it should be exactly zero.

## Source

```py title="examples/05_wasserstein_lipschitz_equivalence.py"
--8<-- "examples/05_wasserstein_lipschitz_equivalence.py"
```

[^wlm]:
    Qinyu Wu, Jonathan Yu-Meng Li and Tiantian Mao, "On Generalization and
    Regularization via Wasserstein Distributionally Robust Optimization",
    *Management Science* (2025).
    [arXiv:2212.05716](https://arxiv.org/abs/2212.05716).
