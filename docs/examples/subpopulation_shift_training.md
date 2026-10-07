# Training under subpopulation shift

The earlier pages evaluate DRO on a fixed vector of losses. This one trains
a model with it, because that is the claim DRO is sold on: *a model fitted
on the worst-case expected loss degrades more gracefully than an
empirical-risk-minimization (ERM) model when the test population is a
different mixture of the training subpopulations.*

That claim is easy to illustrate and easy to oversell, so the experiment was
fixed before any shifted-test number was looked at, includes two controls
that could have embarrassed DRO, and reports the places where it loses.

!!! warning "This is the textbook favourable case"

    Subpopulation shift is the setting DRO was designed for[^dn][^hash].
    Nothing here says DRO helps under other kinds of shift, and the linear
    model is *not* misspecified for the training mixture: what changes at
    test time is $P(y \mid x)$, not the model class.

## The setup

Inputs are $x \sim \mathcal{N}(0, I_2)$ in both groups, so the group is
latent: no function of $x$ can recover it. The groups differ only in the
regression function, $y = a_g^\top x + 0.3\,\varepsilon$:

| Group | Share of training data | Slope $a_g$ |
| --- | --- | --- |
| A (majority) | 90% | $(1,\, 0)$ |
| B (minority) | 10% | $(1,\, 1.5)$ |

Training uses $n = 400$ points (360 from A, 40 from B), full batch, and the
model is `nn.Linear(2, 1)` with a bias. Everything is repeated over 10 seeds
that resample the training set and the initialization.

Because $x$ has the same law in both groups, the risk of a model $(w, b)$ on
group $g$ is exact, no test set needed:

$$
R_g(w, b) = \lVert w - a_g \rVert^2 + b^2 + \sigma_g^2 ,
$$

and the test risk at a mixing proportion $\pi$ of group B is the line

$$
R(\pi) = (1 - \pi)\, R_A + \pi\, R_B .
$$

Its slope $R_B - R_A$ is the model's *degradation rate*: how fast its risk
grows as the population moves from A toward B. A shared test set of 100,000
points per group is drawn once, only to cross-check the formula (the largest
discrepancy over all 50 models is $3.6 \times 10^{-3}$).

!!! note "Why the minority differs in a second feature"

    With one feature and opposite-sign slopes, DRO is hard to tell from
    shrinkage toward zero, which is what weight decay already does. Here the
    population-optimal weights at mixing proportion $\pi$ are
    $w^\star(\pi) = (1-\pi)\,a_A + \pi\,a_B$, a straight path that
    shrinkage cannot follow, and a ridge control checks that it does not.

## The models

All are trained from the same data and initialization within a seed, by the
same full-batch `torch.optim.Adam` loop from [Training with
`torch.optim`](../training.md):

- **ERM** is `KLAmbiguitySet(radius=0.0)`, the identical code path with no
  ambiguity.
- **KL-DRO** at radii $0.05$, $0.2$ and $0.5$, chosen a priori.
- **Ridge** control: ERM plus $0.3\,\lVert w \rVert^2$, a fixed penalty also
  chosen a priori, not tuned on any test result.
- A **group-reweighted ERM oracle**, used only as a reference. It uses the
  group labels, which sample-level DRO does not need.

A KL ball of radius $\rho$ around the empirical distribution can move at most
$m(\rho)$ of the probability mass onto group B, where $m$ solves
$m \log\frac{m}{0.1} + (1-m)\log\frac{1-m}{0.9} = \rho$ (moving group mass
while keeping the within-group proportions is the cheapest way to do it).
That gives $m = 0.207,\ 0.332,\ 0.495$ for the three radii. The adversary
never sees the labels, so it reaches only part of that.

!!! note "One joint solve per step"

    The loss has shape `(seeds, models, n)` with the radii as a tensor that
    broadcasts against it, so every step is one batched tilt bisection for
    all 10 seeds and all three radii. Models are independent and Adam is
    elementwise, so this is the same as training them one by one.

## Is it converged?

An under-converged DRO fit could fake the result, so the script asserts the
following before it reports anything:

| Check | Observed |
| --- | --- |
| Adam ERM vs closed-form least squares | $2.3 \times 10^{-5}$ |
| Adam ridge vs closed-form ridge | $1.1 \times 10^{-5}$ |
| Outer gradient norm of every model | $\le 1.6 \times 10^{-4}$ |
| KL-DRO ($\rho = 0.2$, seed 0) vs an independent joint L-BFGS over $(w, b, \log\eta)$ | $3.6 \times 10^{-7}$ |

Only the outer loop has to converge. The inner worst case is a bracketed
bisection resolved to the dtype's precision on every call, so there is no
inner tolerance left to bias the outer gradient and nothing to warm up
across steps. The L-BFGS reference minimizes the *dual* jointly (it is
jointly convex in $(w, b, \eta)$), from a different starting point and by a
different route, so agreement is not an artefact of shared code.

!!! tip "Why the inner solve needs no tuning"

    At the random initialization the per-sample losses are heavy-tailed
    (maximum around 100). Minimizing the KL dual over $\log\eta$ by
    fixed-step gradient descent would need a step small enough not to send
    the iterate into a flat region it cannot leave, which would spread the
    inner solve over many outer steps. The bisection instead brackets the
    exponential tilt on $[0, 1]$ by construction, so the heavy tail costs
    it nothing and the first Adam step is already exact.

## The evidence

Exact group risks, mean over 10 seeds, with the minority mass $q^\star_B$ of
the worst-case distribution (read from the gradient of the worst-case value
with respect to the loss, as in [Training with
`torch.optim`](../training.md#reading-off-the-worst-case-distribution)):

| Model | $R_A$ | $R_B$ | $R_B - R_A$ | worst group | $q^\star_B$ | allowed |
| --- | --- | --- | --- | --- | --- | --- |
| ERM | 0.111 | 1.965 | 1.854 | 1.965 | 0.100 | |
| ridge ($0.3$) | 0.150 | 2.118 | 1.968 | 2.118 | 0.100 | |
| KL-DRO $\rho = 0.05$ | 0.212 | 1.470 | 1.258 | 1.470 | 0.152 | 0.207 |
| KL-DRO $\rho = 0.2$ | 0.312 | 1.221 | 0.909 | 1.221 | 0.200 | 0.332 |
| KL-DRO $\rho = 0.5$ | 0.373 | 1.113 | 0.741 | 1.113 | 0.256 | 0.495 |
| oracle, minority mass $0.21$ | 0.175 | 1.591 | 1.416 | 1.591 | 0.21 | |
| oracle, minority mass $0.33$ | 0.308 | 1.202 | 0.894 | 1.202 | 0.33 | |
| oracle, minority mass $0.50$ | 0.575 | 0.775 | 0.200 | 0.775 | 0.50 | |

Test risk at three mixing proportions, with the paired per-seed gain over ERM
in brackets (positive means the model beats ERM):

| $\pi$ | ERM | ridge | KL $0.05$ | KL $0.2$ | KL $0.5$ |
| --- | --- | --- | --- | --- | --- |
| 0.1 (training mix) | 0.296 | 0.346 (-0.050) | 0.338 (-0.042) | 0.403 (-0.107) | 0.447 (-0.151) |
| 0.5 | 1.038 | 1.134 (-0.096) | 0.841 (+0.197) | 0.767 (+0.271) | 0.743 (+0.295) |
| 0.9 | 1.780 | 1.921 (-0.141) | 1.344 (+0.436) | 1.130 (+0.649) | 1.039 (+0.741) |

DRO is better than ERM exactly for $\pi > \pi^\star$, where
$\pi^\star = \Delta R_A / (\Delta R_A + |\Delta R_B|)$ uses the DRO-minus-ERM
changes in the two group risks:

| Radius | $\pi^\star$ (mean $\pm$ std over seeds) |
| --- | --- |
| 0.05 | $0.165 \pm 0.035$ |
| 0.2 | $0.209 \pm 0.039$ |
| 0.5 | $0.232 \pm 0.040$ |

The degradation rate falls with the radius in every seed, and the worst-case
minority mass rises with it in every seed. At $\pi = 0.9$ every radius beats
ERM in every seed.

## Reading the figure

- **Left**: exact test risk against $\pi$ for every model, with a band of one
  standard deviation over seeds. The dotted line is the training mix, and
  the dots mark where each DRO line crosses ERM's, with the seed spread of
  $\pi^\star$ as a horizontal bar.
- **Right**: the $R_A$ against $R_B$ trade-off. The green curve is the
  population-optimal path $w^\star(\pi)$ with $b = 0$, which is the exact
  Pareto frontier of the linear model: nothing can sit below it, and the
  script asserts that none does. The dashed line is the label-using oracle.

## Is it just regularization?

The ridge control is the point of the second feature. Shrinkage moves the
slope toward zero, which is away from *both* $a_A$ and $a_B$ at any penalty,
so ridge is worse than ERM on both groups (every seed) instead of trading
one for the other. Measured against ERM, the move of each model decomposes
into a part along the path direction $a_B - a_A$ and a part off it:

| Model | along path | off path |
| --- | --- | --- |
| ridge | $-0.038$ | $0.241$ |
| KL $0.05$ | $0.199$ | $0.055$ |
| KL $0.2$ | $0.315$ | $0.096$ |
| KL $0.5$ | $0.371$ | $0.122$ |

Closed-form sanity check against $w^\star(\pi)$: ERM sits next to
$w^\star(0.1)$ (nearest path point $\pi = 0.088$, distance $0.04$, which is
sampling noise). Each DRO model also lands near the path, at
$\pi = 0.22,\ 0.30,\ 0.34$ for the three radii, while ridge sits $0.22$ away
from it. Note that the DRO models land *further* along the path than the
minority mass $q^\star_B = 0.15,\ 0.20,\ 0.26$ alone would suggest, and they
are not exactly on it: the adversary reweights by loss, so it favours the
high-leverage minority points, and it also moves the intercept. DRO is not a
group reweighting with a different name.

## The negative control

If DRO simply upweights whatever is hard, it should lose when the hard group
is hard for a reason that does not matter at test time. So the same
pipeline is run on a second dataset where the minority has the *same* slope
$a_A$ and differs only in noise, $\sigma_B = 1$:

| Radius | $q^\star_B$ | mean $R_{\mathrm{DRO}} - R_{\mathrm{ERM}}$ | worst seed |
| --- | --- | --- | --- |
| 0.05 | 0.155 | $+0.0074$ | $+0.0001$ |
| 0.2 | 0.224 | $+0.0287$ | $+0.0045$ |
| 0.5 | 0.318 | $+0.0640$ | $+0.0061$ |

DRO upweights the noisy points even *more* than in the shifted case, and the
cost shows up on both groups (the slope is identical, so the two differences
coincide), hence at every $\pi$: it is never better than ERM. In the worst
seed at $\rho = 0.05$ the cost is only $10^{-4}$, so the script asserts the
paired mean and a small per-seed tolerance rather than a strict sign in every
seed.

## The trade-offs, honestly

!!! warning "Every radius costs in-distribution performance"

    At the training mix $\pi = 0.1$, ERM is better than every DRO radius by
    $0.04$ to $0.15$ in test risk, in every seed. DRO beats ERM only once
    the test share of group B exceeds $\pi^\star \approx 0.17$ to $0.23$,
    i.e. at least about 1.7 times the training share. If the test mix is
    the training mix, DRO is a pure loss here.

!!! note "No claim about an optimal radius"

    At high $\pi$ the risk keeps falling as the radius grows
    ($1.34 \to 1.13 \to 1.04$ at $\pi = 0.9$), while the cost at the
    training mix keeps rising. The script does not assert an interior
    optimal radius or exact crossing values; its assertions are the paired
    inequalities above plus loose ranges calibrated from runs of the script
    with slack.

!!! warning "Sample-level DRO is not the oracle"

    The oracle that reweights group B explicitly is a reference, not an
    upper bound for DRO. At minority mass $0.21$ it has *worse* minority risk
    ($1.59$) than DRO at $\rho = 0.05$ ($1.47$), because DRO reweights by
    loss rather than by group. But the oracle can also reach a balanced
    point ($R_A = 0.575$, $R_B = 0.775$ at mass $0.5$) that no radius here
    reaches, since the label-free adversary only uses about half to
    three-quarters of the minority mass its ball allows.

!!! note "What was fixed in advance"

    The slopes, the noise level, the three radii, the ridge penalty and the
    number of seeds were chosen before any shifted-test result was looked
    at, and were not changed afterwards. With 10 seeds, individual seeds are
    noisy; the assertions compare paired differences between models trained
    on the same data. The population is stylized, one linear model, two
    groups, Gaussian inputs: it shows the mechanism and where it loses, not
    a general guarantee.

## Source

The script runs in about 30 seconds on a CPU.

```py title="examples/07_subpopulation_shift_training.py"
--8<-- "examples/07_subpopulation_shift_training.py"
```

[^dn]:
    John C. Duchi and Hongseok Namkoong, "Learning Models with Uniform
    Performance via Distributionally Robust Optimization", *Annals of
    Statistics* 49(3), 1378-1406 (2021).
    [arXiv:1810.08750](https://arxiv.org/abs/1810.08750).

[^hash]:
    Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong and Percy
    Liang, "Fairness Without Demographics in Repeated Loss Minimization",
    *ICML* (2018). [arXiv:1806.08010](https://arxiv.org/abs/1806.08010). The
    group-label version of the same idea is Shiori Sagawa, Pang Wei Koh,
    Tatsunori Hashimoto and Percy Liang, "Distributionally Robust Neural
    Networks for Group Shifts: On the Importance of Regularization for
    Worst-Case Generalization", *ICLR* (2020).
    [arXiv:1911.08731](https://arxiv.org/abs/1911.08731).
