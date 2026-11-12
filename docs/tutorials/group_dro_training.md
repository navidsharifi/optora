# Group DRO under a spurious correlation

A model can be accurate on average and still misclassify almost every
example of one group. This happens when a feature that is easy to read
agrees with the label on most of the training set. Empirical risk
minimization (ERM) relies on that feature, and the examples where it
disagrees with the label are the ones the model gets wrong. Group DRO
trains on the worst group instead of the average[^sagawa]. This page
reproduces the qualitative findings of Sagawa et al. (2020) on a small
synthetic problem, with [`OnlineGroupDRO`](../api/dro/group_dro.md) and
[`group_losses`](../api/dro/group_dro.md).

## The objective

The training data come from $G$ known groups, here the four combinations
of a label $y$ and a spurious attribute $a$. With $R_g(\theta)$ the risk of
group $g$, group DRO solves

$$
\min_\theta \max_{g} R_g(\theta)
= \min_\theta \sup_{q \in \Delta_G} \sum_{g} q_g R_g(\theta).
$$

The inner supremum is a worst case over distributions on the groups, so on
a fixed dataset it is an ambiguity set that Optora already has, applied to
the vector of group losses. `CVaRAmbiguitySet(proportions, alpha=0.0)` is
exactly the maximum over the groups with positive proportion. On
minibatches the maximum of the minibatch group means is biased upward, so
training uses the online algorithm of Sagawa et al. instead. It keeps the
group weights $q$ as state, and on each minibatch it takes one
exponentiated-gradient ascent step on them before the model descends the
weighted loss,

$$
q_g \leftarrow \frac{q_g\, e^{\eta \hat\ell_g}}{\sum_h q_h\, e^{\eta \hat\ell_h}},
\qquad
\theta \leftarrow \theta - \gamma\, \nabla_\theta \sum_g q_g \hat\ell_g(\theta).
$$

## The problem

The data follow the spurious-correlation model of Sagawa, Raghunathan, et
al.[^overparam], with a one-dimensional core feature and a
one-dimensional spurious feature.

| Feature | Distribution | What it predicts |
| --- | --- | --- |
| Core | $y + 0.6\,\varepsilon$ | The label, in every group (Bayes accuracy of this feature alone about 95%) |
| Spurious | $a + 0.1\,\varepsilon$ | The attribute, almost exactly |
| 1000 noise features | $\mathcal{N}(0, 1/1000)$ each | Nothing, but there are enough of them to memorize single examples |

In the 420 training examples the attribute agrees with the label 400 times
(two majority groups of 200) and disagrees 20 times (two minority groups
of 10). The spurious feature is therefore right 95% of the time and far
less noisy than the core one, which makes it the shortcut that ERM takes.
Test accuracy is measured on 2000 fresh examples per group, and the
"average" accuracy is weighted by the training proportions, as in the
paper.

The model is logistic regression on all 1002 features, so it has more
parameters than training examples. Four methods are trained with
`torch.optim.Adam` on minibatches of 128 for 4000 steps, over 5 seeds, all
jointly as one batched loss.

| Method | L2 penalty | Adversary step size |
| --- | --- | --- |
| ERM | $0$ | $0$ |
| ERM + strong L2 | $0.1\,\lVert w \rVert^2$ | $0$ |
| Group DRO | $0$ | $0.01$ |
| Group DRO + strong L2 | $0.1\,\lVert w \rVert^2$ | $0.01$ |

All four run through the same code. ERM is `OnlineGroupDRO` with step size
zero and the training proportions as weights. Its loss is then the
proportion-weighted sum of the minibatch group means, which is an estimate
of the average loss.

## Result

The table shows the means over 5 seeds, as printed by the script.

| Method | Train average | Train worst group | Test average | Test worst group |
| --- | --- | --- | --- | --- |
| ERM | $1.000$ | $1.000$ | $0.969$ | $0.482$ |
| ERM + strong L2 | $0.974$ | $0.440$ | $0.977$ | $0.510$ |
| Group DRO | $1.000$ | $1.000$ | $0.969$ | $0.493$ |
| **Group DRO + strong L2** | $0.956$ | $0.921$ | $0.955$ | $\mathbf{0.938}$ |

- **Regularized group DRO raises the worst group.** Its worst-group test
  accuracy is $0.94$ against $0.48$ for ERM, and it is higher by more than
  30 points on every seed, while its average accuracy is $1.4$ points
  lower. Sagawa et al. (2020) report the same pattern on their image and
  language benchmarks, namely a large gain in worst-group accuracy for a
  small loss in average accuracy.
- **Regularization alone does not.** ERM with the same penalty still
  averages $0.51$ on the worst group. The penalty keeps it from memorizing
  examples through the noise features, but it still gives the spurious
  feature about as much weight as the core one, and the spurious feature
  is the one that misclassifies the minority groups.
- **Unregularized group DRO behaves like ERM.** Both classify every
  training example correctly within about 300 steps, using the noise
  features to memorize the 20 minority examples. Once every training group
  has a loss near zero, the ascent factor $e^{\eta \hat\ell_g}$ is close to
  one for every group, so the adversary's weights barely move. Their total
  on the minority groups stalls at $0.10$ instead of reaching $0.49$, and
  the model learns what ERM learned.

!!! warning "Group DRO needs regularization when the model can interpolate"

    This is the central point of Sagawa et al. (2020). An
    over-parameterized model can drive the training loss of every group
    toward zero, so the worst-group objective and the average objective
    have the same interpolating minimizers, and group DRO behaves like
    ERM. Strong L2 penalties or early stopping keep the training loss of
    the minority groups away from zero. In this setup the minority groups
    are memorized within about 300 steps, before early stopping could
    help, so the script uses the penalty.

!!! note "The worst-group objective, evaluated exactly"

    After training, the script evaluates the worst-group training loss with
    `CVaRAmbiguitySet(proportions, alpha=0.0)` applied to
    `group_losses(...)`, and asserts that it is bitwise the maximum of the
    group losses. KL and chi-square balls around the group proportions give
    the same value once the radius reaches $\log(1/p_{g^\star})$ and
    $1/p_{g^\star} - 1$, respectively. See [Group
    DRO](../training.md#group-dro) in the training guide.

## Reading the figure

- The **left** panel plots worst-group against average test accuracy,
  with one small point per seed and one large point per method mean.
  Three methods cluster near $(0.97, 0.5)$, and regularized group DRO lies
  alone at the top, slightly to the left.
- The **middle** panel shows worst-group accuracy during training, on the
  test set (solid) and the training set (dashed). Unregularized ERM and
  group DRO reach $1.0$ on the training set within about 300 steps and
  stay near $0.5$ on the test set. Regularized group DRO is the only
  method whose test curve rises well above $0.5$, and its training curve
  ends close to it.
- The **right** panel shows the total weight that `OnlineGroupDRO` puts
  on the two minority groups for the two group DRO methods, with the
  range over seeds shaded, against their training share of $0.048$.

## The boundary during training

The animation reruns the training for 3000 steps with three seeds and a
checkpoint every fifty steps. The left panel projects each model of the
first seed onto its two informative weights and draws the line on which
those two features give a zero margin, in the (core, spurious) plane, over
a sample of test points with the minority groups coloured. The three
methods whose worst-group accuracy stays near 50% settle on a boundary
close to the diagonal through both minority groups, which is why they
classify about half of each minority group correctly. Regularized group
DRO rotates its boundary toward the vertical, so that its prediction
depends mainly on the core feature. The right panel shows the adversary's
weight on the minority groups, which rises to about ten times their
training share only when the penalty keeps the groups from being fit
exactly. The projection leaves out the noise features, so a model that
memorizes the training set through them looks the same here as one that
does not. The dashed training curves tell the two apart.

<figure class="figure" markdown="span">
  ![The decision boundary of each method on the core and spurious features, the worst-group accuracy and the adversary's minority weight during training](../assets/figures/group-dro-boundary-light.gif#only-light){ loading=lazy }
  ![The decision boundary of each method on the core and spurious features, the worst-group accuracy and the adversary's minority weight during training](../assets/figures/group-dro-boundary-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows test points in the (core, spurious)
  plane, with the minority groups coloured and each method's boundary on
  those two features. The middle panel shows the worst-group accuracy per
  step, on the test set (solid) and the training set (dashed). The right
  panel shows the online adversary's total weight on the minority groups,
  with the range over seeds.</figcaption>
</figure>

## The figure

The figure below is rendered from this script when the site is built, so
it always shows the output of the current code.

<figure class="figure" markdown="span">
  ![Worst-group against average test accuracy, worst-group accuracy during training, and the adversary's minority weight](../assets/figures/09_group_dro_training-light.png#only-light){ loading=lazy }
  ![Worst-group against average test accuracy, worst-group accuracy during training, and the adversary's minority weight](../assets/figures/09_group_dro_training-dark.png#only-dark){ loading=lazy }
  <figcaption>The left panel shows worst-group against average test accuracy per seed and per method. The middle panel shows worst-group accuracy during training, on the test set (solid) and the training set (dashed). The right panel shows the online adversary's total weight on the minority groups.</figcaption>
</figure>

## Source

```py title="tutorials/09_group_dro_training.py"
--8<-- "tutorials/09_group_dro_training.py"
```

[^sagawa]:
    Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto and Percy Liang,
    "Distributionally Robust Neural Networks for Group Shifts: On the
    Importance of Regularization for Worst-Case Generalization", *ICLR*
    (2020). [arXiv:1911.08731](https://arxiv.org/abs/1911.08731).

[^overparam]:
    Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh and Percy Liang, "An
    Investigation of Why Overparameterization Exacerbates Spurious
    Correlations", *ICML* (2020).
    [arXiv:2005.04345](https://arxiv.org/abs/2005.04345).
