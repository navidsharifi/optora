"""Group DRO against ERM under a spurious correlation.

A model trained by empirical risk minimization (ERM) is accurate on average
and can still fail on a small group: when a feature that is easy to read
agrees with the label on most of the training data, the model leans on it,
and the examples where it disagrees are the ones it gets wrong. Group DRO
(Sagawa, Koh, Hashimoto, and Liang 2020) trains on the worst group's loss
instead. This script reproduces the paper's qualitative findings on a small
synthetic problem: group DRO raises worst-group accuracy a lot at a small
cost in average accuracy, but only with strong regularization. An
over-parameterized model trained without it fits every training group
perfectly, so the worst training group has no loss left to upweight, and
group DRO ends where ERM does.

Data. A label `y` and a spurious attribute `a`, both in {-1, +1}, define
four groups (y, a). The features are the spurious-correlation model of
Sagawa, Raghunathan, Koh, and Liang (2020), in one dimension each:

* a core feature `y + 0.6 * eps`, which predicts the label in every group
  (Bayes accuracy about 95%);
* a spurious feature `a + 0.1 * eps`, which reads `a` almost exactly;
* 1000 noise features, each `N(0, 1/1000)`, which a model can use to
  memorize individual training examples.

In training, `a = y` for 400 of 420 examples (the majority groups, 200
each) and `a = -y` for 20 (the minority groups, 10 each), so the spurious
feature is right 95% of the time and is far less noisy than the core one.
Test accuracy is measured on 2000 fresh examples per group; the average is
weighted by the training proportions, as in the paper.

Models. Logistic regression on all 1002 features, no bias, so it has more
parameters than training examples. Four methods, each trained with
`torch.optim.Adam` on minibatches of 128 drawn with replacement, over 5
seeds, all jointly as one batched loss:

* ERM, and ERM with a strong L2 penalty `0.1 ||w||^2`;
* group DRO, and group DRO with the same penalty.

Every method runs through the same `OnlineGroupDRO`: per minibatch,
`group_losses` averages the per-example losses within each group, `step`
moves the group weights by exponentiated-gradient ascent, and the model
descends the weighted loss (Algorithm 1 of Sagawa et al.). ERM is the same
module with step size zero and the training proportions as weights: its
objective is then the proportion-weighted minibatch group means, an
estimate of the average loss. After training, the worst-group training loss
is evaluated exactly with `CVaRAmbiguitySet(proportions, alpha=0.0)`, which
is the maximum of the group losses.

Expected result, matching the paper's table: ERM is about 97% accurate on
average and below 50% on the worst group. Strong regularization alone does
not change that. Group DRO without it fits the training groups perfectly
and collapses to ERM. Group DRO with it raises worst-group accuracy above
90% for a drop of about two points on average.

References:
    Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang,
    "Distributionally Robust Neural Networks for Group Shifts: On the
    Importance of Regularization for Worst-Case Generalization", ICLR 2020.
    Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh, and Percy Liang, "An
    Investigation of Why Overparameterization Exacerbates Spurious
    Correlations", ICML 2020.

Run:
    python examples/09_group_dro_training.py
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from _plotting import save_figure

from optora.dro import CVaRAmbiguitySet, OnlineGroupDRO, group_losses

SEED = 20261031
NUM_SEEDS = 5
# Groups g = 2 [y = +1] + [a = +1]: (y, a) = (-1, -1), (-1, +1), (+1, -1),
# (+1, +1). Groups 0 and 3 are the majority, 1 and 2 the minority.
GROUP_SIZES = (200, 10, 10, 200)
MINORITY = (1, 2)
TEST_GROUP_SIZE = 2000
NUM_GROUPS = len(GROUP_SIZES)
CORE_NOISE = 0.6
SPURIOUS_NOISE = 0.1
NOISE_FEATURES = 1000

STEPS = 4000
BATCH = 128
LEARNING_RATE = 0.01
ADVERSARY_STEP_SIZE = 0.01
STRONG_PENALTY = 0.1
EVALUATE_EVERY = 100

# (name, L2 penalty, adversary step size)
METHODS = (
    ("ERM", 0.0, 0.0),
    ("ERM + strong L2", STRONG_PENALTY, 0.0),
    ("group DRO", 0.0, ADVERSARY_STEP_SIZE),
    ("group DRO + strong L2", STRONG_PENALTY, ADVERSARY_STEP_SIZE),
)


def make_data(
    group_sizes: tuple[int, ...], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw examples of the spurious-correlation model.

    Args:
        group_sizes: Number of examples of each group.
        generator: Seeded generator.

    Returns:
        Features of shape `(n, 2 + NOISE_FEATURES)`, labels in {-1, +1} of
        shape `(n,)`, and group indices of shape `(n,)`.
    """
    groups = torch.cat(
        [torch.full((size,), group) for group, size in enumerate(group_sizes)]
    )
    labels = torch.where(groups >= 2, 1.0, -1.0)
    attribute = torch.where(groups % 2 == 1, 1.0, -1.0)
    size = len(groups)
    core = labels + CORE_NOISE * torch.randn(size, generator=generator)
    spurious = attribute + SPURIOUS_NOISE * torch.randn(size, generator=generator)
    noise = torch.randn(size, NOISE_FEATURES, generator=generator)
    features = torch.cat(
        [core[:, None], spurious[:, None], noise / NOISE_FEATURES**0.5], dim=-1
    )
    return features, labels, groups


def per_example_loss(
    weights: torch.Tensor, features: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Logistic loss of every model on every example.

    Args:
        weights: Model weights of shape `(seeds, methods, d)`.
        features: Features of shape `(seeds, n, d)` or `(n, d)`.
        labels: Labels of shape `(seeds, n)` or `(n,)`.

    Returns:
        Losses of shape `(seeds, methods, n)`.
    """
    if features.ndim == 2:
        features, labels = (
            features.expand(len(weights), -1, -1),
            labels.expand(len(weights), -1),
        )
    margins = torch.einsum("smd,snd->smn", weights, features) * labels[:, None, :]
    return F.softplus(-margins)


def group_accuracy(
    weights: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Accuracy of every model on every group, of shape `(seeds, methods, G)`."""
    if features.ndim == 2:
        features, labels = (
            features.expand(len(weights), -1, -1),
            labels.expand(len(weights), -1),
        )
        groups = groups.expand(len(weights), -1)
    margins = torch.einsum("smd,snd->smn", weights, features) * labels[:, None, :]
    return group_losses((margins > 0).to(margins.dtype), groups[:, None, :], NUM_GROUPS)


def train(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    proportions: torch.Tensor,
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train every method on every seed jointly.

    Args:
        features: Training features of shape `(seeds, n, d)`.
        labels: Training labels of shape `(seeds, n)`.
        groups: Training groups of shape `(seeds, n)`.
        proportions: Training group proportions of shape `(G,)`.
        test: Test features, labels, and groups, shared by every seed.

    Returns:
        The final weights of shape `(seeds, methods, d)`, and the history of
        worst-group train and test accuracy and of the adversary's weight
        on the minority groups, each of shape `(checkpoints, seeds,
        methods)`, with the steps of the checkpoints.
    """
    seeds, size, dimension = features.shape
    penalty = torch.tensor([method[1] for method in METHODS])
    step_size = torch.tensor([method[2] for method in METHODS])
    weights = torch.zeros(seeds, len(METHODS), dimension, requires_grad=True)
    adversary = OnlineGroupDRO(
        proportions.expand(seeds, len(METHODS), -1),
        step_size=step_size.expand(seeds, -1),
    )
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED + 1)
    history: dict[str, list[torch.Tensor]] = {
        "train worst": [],
        "test worst": [],
        "minority weight": [],
    }
    steps = []
    for step in range(STEPS + 1):
        if step % EVALUATE_EVERY == 0:
            with torch.no_grad():
                history["train worst"].append(
                    group_accuracy(weights, features, labels, groups).amin(-1)
                )
                history["test worst"].append(group_accuracy(weights, *test).amin(-1))
                history["minority weight"].append(
                    adversary.weights[..., list(MINORITY)].sum(-1)
                )
                steps.append(step)
        if step == STEPS:
            break
        index = torch.randint(size, (seeds, BATCH), generator=generator)
        batch_features = torch.gather(
            features, 1, index[..., None].expand(-1, -1, dimension)
        )
        loss = per_example_loss(weights, batch_features, torch.gather(labels, 1, index))
        group_loss = group_losses(
            loss, torch.gather(groups, 1, index)[:, None, :], NUM_GROUPS
        )
        adversary.step(group_loss)
        objective = adversary(group_loss) + penalty * torch.sum(weights**2, dim=-1)
        optimizer.zero_grad()
        objective.sum().backward()
        optimizer.step()
    result = {name: torch.stack(values) for name, values in history.items()}
    result["steps"] = torch.tensor(steps)
    return weights.detach(), result


def main() -> None:
    torch.manual_seed(SEED)
    datasets = [
        make_data(GROUP_SIZES, torch.Generator().manual_seed(SEED + seed))
        for seed in range(NUM_SEEDS)
    ]
    features, labels, groups = (
        torch.stack(parts) for parts in zip(*datasets, strict=True)
    )
    proportions = torch.tensor(GROUP_SIZES, dtype=torch.float32)
    proportions = proportions / proportions.sum()
    test = make_data(
        (TEST_GROUP_SIZE,) * NUM_GROUPS, torch.Generator().manual_seed(SEED - 1)
    )

    weights, history = train(features, labels, groups, proportions, test)

    train_accuracy = group_accuracy(weights, features, labels, groups)
    test_accuracy = group_accuracy(weights, *test)
    train_group_loss = group_losses(
        per_example_loss(weights, features, labels), groups[:, None, :], NUM_GROUPS
    )
    worst_group = CVaRAmbiguitySet(proportions, alpha=0.0)
    worst_train_loss = worst_group.worst_case_expectation(train_group_loss)
    assert torch.equal(worst_train_loss, train_group_loss.amax(-1))
    summary = {
        "train avg": train_accuracy @ proportions,
        "train worst": train_accuracy.amin(-1),
        "test avg": test_accuracy @ proportions,
        "test worst": test_accuracy.amin(-1),
        "worst train loss": worst_train_loss,
    }

    print(f"accuracy over {NUM_SEEDS} seeds, mean [min, max]")
    print(f"  {'method':<24}" + "".join(f"{name:>20}" for name in summary))
    for index, (name, _, _) in enumerate(METHODS):
        cells = "".join(
            f"{float(column.mean()):>8.3f} "
            f"[{float(column.min()):.2f}, {float(column.max()):.2f}]"
            for column in (values[:, index] for values in summary.values())
        )
        print(f"  {name:<24}{cells}")
    minority = history["minority weight"][-1]
    print(
        "\nadversary weight on the minority groups at the end "
        f"(training share {float(proportions[list(MINORITY)].sum()):.3f}):"
    )
    for index, (name, _, step_size) in enumerate(METHODS):
        if step_size > 0:
            print(f"  {name:<24}{float(minority[:, index].mean()):.3f}")

    erm, erm_l2, dro, dro_l2 = range(len(METHODS))
    test_worst, test_avg = summary["test worst"], summary["test avg"]
    # The qualitative result of Sagawa et al. (2020): worst-group accuracy
    # up by a wide margin on every seed, average accuracy down slightly on
    # average over seeds (a single seed can tie).
    assert bool(torch.all(test_worst[:, dro_l2] > test_worst[:, erm] + 0.3))
    assert bool(torch.all(test_worst[:, dro_l2] > test_worst[:, erm_l2] + 0.3))
    average_drop = float((test_avg[:, erm] - test_avg[:, dro_l2]).mean())
    assert 0.0 < average_drop < 0.05
    # Without regularization the model fits every training group, and group
    # DRO generalizes no better than ERM.
    assert bool(torch.all(summary["train worst"][:, dro] == 1.0))
    assert float((test_worst[:, dro] - test_worst[:, erm]).abs().mean()) < 0.05
    assert float(test_worst[:, erm_l2].mean()) < 0.6

    colors = {name: f"C{index}" for index, (name, _, _) in enumerate(METHODS)}
    fig, (ax_scatter, ax_curves, ax_weights) = plt.subplots(1, 3, figsize=(16, 4.6))
    for index, (name, _, _) in enumerate(METHODS):
        ax_scatter.scatter(
            test_avg[:, index], test_worst[:, index], color=colors[name], alpha=0.35
        )
        ax_scatter.scatter(
            test_avg[:, index].mean(),
            test_worst[:, index].mean(),
            color=colors[name],
            s=120,
            edgecolor=plt.rcParams["text.color"],
            label=name,
        )
    ax_scatter.set_xlabel("average test accuracy")
    ax_scatter.set_ylabel("worst-group test accuracy")
    ax_scatter.set_title("test accuracy (small points: seeds)")
    ax_scatter.legend(fontsize="small", loc="center left")

    steps = history["steps"]
    for index, (name, _, _) in enumerate(METHODS):
        ax_curves.plot(
            steps,
            history["test worst"][:, :, index].mean(-1),
            color=colors[name],
            label=f"{name}, test",
        )
        ax_curves.plot(
            steps,
            history["train worst"][:, :, index].mean(-1),
            color=colors[name],
            linestyle="--",
        )
    ax_curves.set_xlabel("step")
    ax_curves.set_ylabel("worst-group accuracy")
    ax_curves.set_title("worst group during training (dashed: train)")
    ax_curves.legend(fontsize="small")

    for index, (name, _, step_size) in enumerate(METHODS):
        if step_size > 0:
            weight = history["minority weight"][:, :, index]
            ax_weights.plot(steps, weight.mean(-1), color=colors[name], label=name)
            ax_weights.fill_between(
                steps, weight.amin(-1), weight.amax(-1), color=colors[name], alpha=0.15
            )
    ax_weights.axhline(
        float(proportions[list(MINORITY)].sum()),
        color=plt.rcParams["text.color"],
        linestyle=":",
        label="training share",
    )
    ax_weights.set_xlabel("step")
    ax_weights.set_ylabel("adversary weight on the minority groups")
    ax_weights.set_title("OnlineGroupDRO weights (range over seeds)")
    ax_weights.legend(fontsize="small")

    fig.tight_layout()
    save_figure(fig, "09_group_dro_training")


if __name__ == "__main__":
    main()
