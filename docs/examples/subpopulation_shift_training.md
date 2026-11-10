# Training under subpopulation shift

The earlier pages evaluate DRO on a fixed vector of losses. This one trains
a model with it, in order to test the claim most often made for DRO. The
claim is that *a model fitted on the worst-case expected loss degrades more
gracefully than an empirical-risk-minimization (ERM) model when the test
population is a different mixture of the training subpopulations.*

This claim is easy to illustrate and easy to overstate. The experiment was
therefore fixed before any shifted-test number was examined. It includes
two controls under which DRO could have failed, and it reports the
settings in which DRO loses to ERM.

!!! warning "This is the standard favourable case"

    Subpopulation shift is the setting DRO was designed for[^dn][^hash].
    Nothing here says that DRO helps under other kinds of shift. The linear
    model is also *not* misspecified for the training mixture, because what
    changes at test time is $P(y \mid x)$ and not the model class.

## The setup

Inputs are $x \sim \mathcal{N}(0, I_2)$ in both groups, so the group is
latent and no function of $x$ can recover it. The groups differ only in
the regression function $y = a_g^\top x + 0.3\,\varepsilon$.

| Group | Share of training data | Slope $a_g$ |
| --- | --- | --- |
| A (majority) | 90% | $(1,\, 0)$ |
| B (minority) | 10% | $(1,\, 1.5)$ |

Training uses $n = 400$ points (360 from A, 40 from B) in full batch, and
the model is `nn.Linear(2, 1)` with a bias. Everything is repeated over 10
seeds, each of which resamples the training set and the initialization.

Because $x$ has the same law in both groups, the risk of a model $(w, b)$ on
group $g$ is available in closed form, without a test set,

$$
R_g(w, b) = \lVert w - a_g \rVert^2 + b^2 + \sigma_g^2 ,
$$

and the test risk at a mixing proportion $\pi$ of group B is the line

$$
R(\pi) = (1 - \pi)\, R_A + \pi\, R_B .
$$

Its slope $R_B - R_A$ is the model's *degradation rate*, which measures how
fast its risk grows as the population moves from A toward B. A shared test
set of 100,000 points per group is drawn once, only to cross-check the
formula. The largest discrepancy over all 50 models is
$3.6 \times 10^{-3}$.

!!! note "Why the minority differs in a second feature"

    With one feature and slopes of opposite sign, DRO is hard to
    distinguish from shrinkage toward zero, which weight decay already
    provides. Here the population-optimal weights at mixing proportion
    $\pi$ are $w^\star(\pi) = (1-\pi)\,a_A + \pi\,a_B$. This is a straight
    path that shrinkage cannot follow, and a ridge control checks that it
    does not.

## The models

All models are trained from the same data and initialization within a
seed, by the same full-batch `torch.optim.Adam` loop as in [Training with
`torch.optim`](../training.md).

- **ERM** is `KLAmbiguitySet(radius=0.0)`, the identical code path with no
  ambiguity.
- **KL-DRO** is trained at radii $0.05$, $0.2$ and $0.5$, chosen a priori.
- The **ridge** control is ERM plus $0.3\,\lVert w \rVert^2$, a fixed
  penalty that was also chosen a priori and not tuned on any test result.
- A **group-reweighted ERM oracle** serves only as a reference. It uses the
  group labels, which sample-level DRO does not need.

A KL ball of radius $\rho$ around the empirical distribution can move at
most $m(\rho)$ of the probability mass onto group B, where $m$ solves
$m \log\frac{m}{0.1} + (1-m)\log\frac{1-m}{0.9} = \rho$. By the
data-processing inequality, moving group mass while keeping the
within-group proportions fixed is the cheapest way to do so. This gives
$m = 0.207,\ 0.332,\ 0.495$ for the three radii. The adversary never sees
the labels, so it reaches only part of that mass.

!!! note "One joint solve per step"

    The loss has shape `(seeds, models, n)`, and the radii are a tensor
    that broadcasts against it, so every step is one batched tilt bisection
    for all 10 seeds and all three radii. The models are independent and
    Adam is elementwise, so this is equivalent to training them one by one.

## Is it converged?

An under-converged DRO fit could produce a misleading result, so the
script asserts the following checks before it reports anything.

| Check | Observed |
| --- | --- |
| Adam ERM vs closed-form least squares | $2.3 \times 10^{-5}$ |
| Adam ridge vs closed-form ridge | $1.1 \times 10^{-5}$ |
| Outer gradient norm of every model | $\le 1.6 \times 10^{-4}$ |
| KL-DRO ($\rho = 0.2$, seed 0) vs an independent joint L-BFGS over $(w, b, \log\eta)$ | $3.6 \times 10^{-7}$ |

Only the outer loop has to converge. The inner worst case is a bracketed
bisection resolved to the dtype's precision on every call, so no inner
tolerance biases the outer gradient, and no inner state has to be warm
started across steps. The L-BFGS reference minimizes the *dual* jointly
over the model parameters and the multiplier. The dual is jointly convex in
$(w, b, \eta)$, and L-BFGS starts from a different point and follows a
different route, so the agreement does not come from shared code.

!!! tip "Why the inner solve needs no tuning"

    At the random initialization the per-sample losses are strongly
    right-skewed. Across the 10 seeds the largest loss is between about 8
    and 45, while the mean loss is between about 0.6 and 3. Minimizing the
    KL dual over $\log\eta$ by fixed-step gradient descent would need a
    step size chosen for this scale, and the inner solve would then be
    spread over many outer steps. The bisection instead brackets the
    exponential tilt on $[0, 1]$ by construction, so the scale of the
    losses does not affect it, and the inner solve is exact from the first
    Adam step.

## The evidence

The table below reports exact group risks, averaged over 10 seeds,
together with the minority mass $q^\star_B$ of the worst-case distribution.
This mass is read from the gradient of the worst-case value with respect to
the loss, as in [Training with
`torch.optim`](../training.md#reading-off-the-worst-case-distribution).

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

The next table gives the test risk at three mixing proportions, with the
paired per-seed gain over ERM in brackets. A positive gain means that the
model beats ERM.

| $\pi$ | ERM | ridge | KL $0.05$ | KL $0.2$ | KL $0.5$ |
| --- | --- | --- | --- | --- | --- |
| 0.1 (training mix) | 0.296 | 0.346 (-0.050) | 0.338 (-0.042) | 0.403 (-0.107) | 0.447 (-0.151) |
| 0.5 | 1.038 | 1.134 (-0.096) | 0.841 (+0.197) | 0.767 (+0.271) | 0.743 (+0.295) |
| 0.9 | 1.780 | 1.921 (-0.141) | 1.344 (+0.436) | 1.130 (+0.649) | 1.039 (+0.741) |

DRO is better than ERM exactly for $\pi > \pi^\star$, where
$\pi^\star = \Delta R_A / (\Delta R_A + |\Delta R_B|)$ uses the DRO-minus-ERM
changes in the two group risks.

| Radius | $\pi^\star$ (mean $\pm$ std over seeds) |
| --- | --- |
| 0.05 | $0.165 \pm 0.035$ |
| 0.2 | $0.209 \pm 0.039$ |
| 0.5 | $0.232 \pm 0.040$ |

In every seed, the degradation rate falls as the radius grows and the
worst-case minority mass rises. At $\pi = 0.9$ every radius beats ERM in
every seed.

## Reading the figure

- The **left** panel shows the exact test risk against $\pi$ for every
  model, with a band of one standard deviation over seeds. The dotted line
  marks the training mix. The dots mark where each DRO line crosses the
  ERM line, and the horizontal bar through each dot shows the seed spread
  of $\pi^\star$.
- The **right** panel shows the trade-off between $R_A$ and $R_B$. The
  green curve is the population-optimal path $w^\star(\pi)$ with $b = 0$,
  which is the exact Pareto frontier of the linear model. No model can lie
  below it, and the script asserts that none does. The dashed line is the
  label-using oracle.

## Is it just regularization?

The ridge control is the reason for the second feature. Shrinkage moves the
slope toward zero, which is away from *both* $a_A$ and $a_B$ at any
penalty. Ridge is therefore worse than ERM on both groups, in every seed,
instead of trading one group against the other. Measured from ERM, the
displacement of each model decomposes into a component along the path
direction $a_B - a_A$ and a component orthogonal to it.

| Model | along path | off path |
| --- | --- | --- |
| ridge | $-0.038$ | $0.241$ |
| KL $0.05$ | $0.199$ | $0.055$ |
| KL $0.2$ | $0.315$ | $0.096$ |
| KL $0.5$ | $0.371$ | $0.122$ |

As a closed-form sanity check against $w^\star(\pi)$, ERM lies close to
$w^\star(0.1)$, with the nearest path point at $\pi = 0.088$ and a distance
of $0.04$ that is consistent with sampling noise. Each DRO model also lies
near the path, at $\pi = 0.22,\ 0.30,\ 0.34$ for the three radii, while
ridge lies $0.22$ away from it. The DRO models lie *further* along the path
than the minority masses $q^\star_B = 0.15,\ 0.20,\ 0.26$ alone would
suggest, and they are not exactly on it. The adversary weights each point
by its loss, so among the minority points it favours those with large
residuals, which are the points with the most leverage on $w_2$. It also
moves the intercept. Sample-level DRO is therefore not equivalent to a
group reweighting.

## The negative control

If DRO upweights any group with high loss, it should lose when that loss
comes from a source that does not change at test time. The same pipeline
is therefore run on a second dataset in which the minority has the *same*
slope $a_A$ and differs only in noise, with $\sigma_B = 1$.

| Radius | $q^\star_B$ | mean $R_{\mathrm{DRO}} - R_{\mathrm{ERM}}$ | worst seed |
| --- | --- | --- | --- |
| 0.05 | 0.155 | $+0.0074$ | $+0.0001$ |
| 0.2 | 0.224 | $+0.0287$ | $+0.0045$ |
| 0.5 | 0.318 | $+0.0640$ | $+0.0061$ |

DRO upweights the noisy points even *more* than the shifted points in the
first experiment. The resulting increase in risk is the same on both
groups, because the slopes are identical and the two group risks differ
only by the constant $\sigma_g^2$. DRO is therefore worse than ERM at every
$\pi$. In the worst seed at $\rho = 0.05$ the increase is only $10^{-4}$,
so the script asserts the paired mean and a small per-seed tolerance
rather than a strict sign in every seed.

## The trade-offs

!!! warning "Every radius increases the in-distribution risk"

    At the training mix $\pi = 0.1$, ERM is better than every DRO radius in
    every seed, by $0.04$ to $0.15$ in mean test risk. DRO beats ERM only
    once the test share of group B exceeds $\pi^\star \approx 0.17$ to
    $0.23$, which is about 1.7 to 2.3 times the training share. If the test
    mix equals the training mix, DRO only increases the risk in this
    setting.

!!! note "No claim about an optimal radius"

    At high $\pi$ the risk keeps falling as the radius grows
    ($1.34 \to 1.13 \to 1.04$ at $\pi = 0.9$), while the cost at the
    training mix keeps rising. The script does not assert an interior
    optimal radius or exact crossing values. Its assertions are the paired
    inequalities above and loose ranges set with slack from runs of the
    script.

!!! warning "Sample-level DRO is not the oracle"

    The oracle that reweights group B explicitly is a reference, not an
    upper bound for DRO. At minority mass $0.21$ it has a *higher* minority
    risk ($1.59$) than DRO at $\rho = 0.05$ ($1.47$), because DRO reweights
    by loss rather than by group. The oracle can also reach a balanced
    point ($R_A = 0.575$, $R_B = 0.775$ at mass $0.5$) that no radius here
    reaches, because the label-free adversary uses only about half to
    three-quarters of the minority mass its ball allows.

!!! note "What was fixed in advance"

    The slopes, the noise level, the three radii, the ridge penalty and the
    number of seeds were chosen before any shifted-test result was
    examined, and they were not changed afterwards. With 10 seeds the
    individual seeds are noisy, so the assertions compare paired
    differences between models trained on the same data. The population is
    stylized, with one linear model, two groups and Gaussian inputs. It
    illustrates the mechanism and the settings in which DRO loses, and it
    is not a general guarantee.

## Varying the radius

The static figure fixes three radii. The animation trains the same model
at forty radii from zero to one half in one batched Adam run, and shows
the radius increasing from frame to frame. The left panel shows the fitted
slope moving away from the ERM solution close to the optimal trade-off
path. The middle panel shows the exact test-risk line $R(\pi)$ rotating as
the radius grows, with its crossing $\pi^\star$ with the ERM line marked.
The right panel shows the mass the adversary puts on the minority points,
against the largest mass a KL ball of that radius allows. The animation
uses one seed and no controls, because it is meant to show that the
mechanism varies continuously with the radius. The statistics are in the
sections above.

<figure class="figure" markdown="span">
  ![The fitted slope, the exact test-risk line and the adversary's minority mass of KL-DRO as the radius grows](../assets/figures/shift-radius-light.gif#only-light){ loading=lazy }
  ![The fitted slope, the exact test-risk line and the adversary's minority mass of KL-DRO as the radius grows](../assets/figures/shift-radius-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the fitted slope on the $(w_1, w_2)$
  plane, moving from the ERM solution toward the minority slope. The
  middle panel shows the exact test risk against the minority share $\pi$
  for ERM and for KL-DRO at the current radius, with the crossing marked.
  The right panel shows the worst-case mass on the minority points against
  the largest mass the ball allows.</figcaption>
</figure>

## The figure

The figure is rendered from this script when the site is built, so it
always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Test risk of ERM, KL-DRO and the controls as the minority mixing proportion changes](../assets/figures/07_subpopulation_shift_training-light.png#only-light){ loading=lazy }
  ![Test risk of ERM, KL-DRO and the controls as the minority mixing proportion changes](../assets/figures/07_subpopulation_shift_training-dark.png#only-dark){ loading=lazy }
  <figcaption>The left panel shows the exact test risk against the minority proportion $\pi$ for ERM, KL-DRO at three radii and the ridge control, with the crossings $\pi^\star$ marked. The right panel shows majority against minority risk, where DRO moves close to the optimal trade-off path and ridge moves away from it.</figcaption>
</figure>

## Source

The script runs in under half a minute on a CPU.

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
