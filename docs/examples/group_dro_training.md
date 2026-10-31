# Group DRO under a spurious correlation

A model can be accurate on average and still fail one group of its data
almost every time. This happens when a feature that is easy to read agrees
with the label on most of the training set: empirical risk minimization
(ERM) leans on it, and the examples where it disagrees are the ones the
model gets wrong. Group DRO trains on the worst group instead of the
average[^sagawa]. This page reproduces the qualitative findings of Sagawa et
al. (2020) on a small synthetic problem, with
[`OnlineGroupDRO`](../api/dro/group_dro.md) and
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
a fixed dataset it is an ambiguity set Optora already has, applied to the
vector of group losses: `CVaRAmbiguitySet(proportions, alpha=0.0)` is
exactly the maximum over the groups. On minibatches the maximum of the
minibatch group means is biased upward, so training uses the online
algorithm of Sagawa et al. instead. It keeps the group weights $q$ as state
and, on each minibatch, takes one exponentiated-gradient ascent step on
them before the model descends the weighted loss:

$$
q_g \leftarrow \frac{q_g\, e^{\eta \hat\ell_g}}{\sum_h q_h\, e^{\eta \hat\ell_h}},
\qquad
\theta \leftarrow \theta - \gamma\, \nabla_\theta \sum_g q_g \hat\ell_g(\theta).
$$

## The problem

The spurious-correlation model of Sagawa, Raghunathan, et al.[^overparam],
with one dimension per feature:

| Feature | Distribution | What it predicts |
| --- | --- | --- |
| Core | $y + 0.6\,\varepsilon$ | The label, in every group (Bayes accuracy about 95%) |
| Spurious | $a + 0.1\,\varepsilon$ | The attribute, almost exactly |
| 1000 noise features | $\mathcal{N}(0, 1/1000)$ each | Nothing; enough of them to memorize single examples |

In the 420 training examples the attribute agrees with the label 400 times
(two majority groups of 200) and disagrees 20 times (two minority groups
of 10). The spurious feature is therefore right 95% of the time and far
less noisy than the core one, which is exactly the shortcut ERM takes.
Test accuracy is measured on 2000 fresh examples per group; "average" is
weighted by the training proportions, as in the paper.

The model is logistic regression on all 1002 features, more parameters
than training examples. Four methods are trained with `torch.optim.Adam` on
minibatches of 128 for 4000 steps, over 5 seeds, all jointly as one
batched loss:

| Method | L2 penalty | Adversary step size |
| --- | --- | --- |
| ERM | $0$ | $0$ |
| ERM + strong L2 | $0.1\,\lVert w \rVert^2$ | $0$ |
| Group DRO | $0$ | $0.01$ |
| Group DRO + strong L2 | $0.1\,\lVert w \rVert^2$ | $0.01$ |

All four run through the same code. ERM is `OnlineGroupDRO` with step size
zero and the training proportions as weights: its loss is then the
proportion-weighted minibatch group means, an estimate of the average loss.

## Result

Means over 5 seeds, as printed by the script:

| Method | Train average | Train worst group | Test average | Test worst group |
| --- | --- | --- | --- | --- |
| ERM | $1.000$ | $1.000$ | $0.969$ | $0.482$ |
| ERM + strong L2 | $0.974$ | $0.440$ | $0.977$ | $0.510$ |
| Group DRO | $1.000$ | $1.000$ | $0.969$ | $0.493$ |
| **Group DRO + strong L2** | $0.956$ | $0.921$ | $0.955$ | $\mathbf{0.938}$ |

- **Regularized group DRO fixes the worst group.** Its worst-group test
  accuracy is $0.94$ against $0.48$ for ERM, on every seed by more than 30
  points, for an average accuracy $1.4$ points lower. This is the shape of
  the paper's image and language benchmarks: a large worst-group gain for
  a small average loss.
- **Regularization alone does not.** ERM with the same penalty is still
  below $0.52$ on the worst group: it trades the noise features for the
  spurious one, which is the shortcut that hurts the minority.
- **Unregularized group DRO collapses to ERM.** Both fit every training
  example within about 200 steps, using the noise features to memorize
  the 20 minority examples. Once every training group has near-zero loss
  there is nothing for the adversary to upweight, so its minority weight
  stalls at $0.10$ instead of reaching $0.49$, and the model learns what
  ERM learned.

!!! warning "Group DRO needs regularization when the model can interpolate"

    This is the central point of Sagawa et al. (2020): the worst-group
    *training* loss of an over-parameterized model goes to zero, so group
    DRO has the same minimizers as ERM there. Strong L2 penalties or early
    stopping keep the training loss of the minority groups away from zero.
    In this setup the minority is memorized within 200 steps, before early
    stopping could help, so the script uses the penalty.

!!! note "The worst-group objective, evaluated exactly"

    After training, the script evaluates the worst-group training loss with
    `CVaRAmbiguitySet(proportions, alpha=0.0)` applied to
    `group_losses(...)`, and asserts it is bitwise the maximum of the group
    losses. KL and chi-square balls around the group proportions give the
    same value once the radius passes $\log(1/p_{g^\star})$, respectively
    $1/p_{g^\star} - 1$. See [Group DRO](../training.md#group-dro) in the
    training guide.

## Reading the figure

- **Left**: worst-group against average test accuracy, one small point per
  seed and a large one per method mean. Three methods cluster near
  $(0.97, 0.5)$; regularized group DRO sits alone at the top, slightly to
  the left.
- **Middle**: worst-group accuracy during training, on the test set
  (solid) and the training set (dashed). Unregularized ERM and group DRO
  reach $1.0$ on training within 200 steps and stay below $0.5$ on test.
  Regularized group DRO tracks its training curve.
- **Right**: the total weight `OnlineGroupDRO` puts on the two minority
  groups, with the range over seeds shaded, against their training share
  of $0.048$.

## The figure

Rendered from this script when the site is built, so it always shows the output of the current code:

<figure class="figure" markdown="span">
  ![Worst-group against average test accuracy, worst-group accuracy during training, and the adversary's minority weight](../assets/figures/09_group_dro_training-light.png#only-light){ loading=lazy }
  ![Worst-group against average test accuracy, worst-group accuracy during training, and the adversary's minority weight](../assets/figures/09_group_dro_training-dark.png#only-dark){ loading=lazy }
  <figcaption>Left: worst-group against average test accuracy per seed and per method. Middle: worst-group accuracy during training, test solid and train dashed. Right: the online adversary's total weight on the minority groups.</figcaption>
</figure>

## Source

```py title="examples/09_group_dro_training.py"
--8<-- "examples/09_group_dro_training.py"
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
