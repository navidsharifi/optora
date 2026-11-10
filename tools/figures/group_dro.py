"""Group DRO against ERM under a spurious correlation, animated over training.

The animation belongs to example 09. It trains the example's four methods
on the spurious-correlation model and watches, step by step, the direction
each model gives its two informative features: ERM leans on the spurious
one and its boundary cuts between the two minority groups, regularized
group DRO turns the boundary onto the core feature. The other panels
follow the worst-group accuracy and the weight the online adversary puts
on the two minority groups.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from optora.dro import OnlineGroupDRO, group_losses
from tools.figures._base import Context, FigureSpec

HOLD_FRAMES = 10
SEED = 20261031
GROUP_SIZES = (200, 10, 10, 200)
MINORITY = (1, 2)
NUM_GROUPS = len(GROUP_SIZES)
TEST_GROUP_SIZE = 2000
SHOWN_PER_GROUP = 120
CORE_NOISE = 0.6
SPURIOUS_NOISE = 0.1
NOISE_FEATURES = 1000
BATCH = 128
LEARNING_RATE = 0.01
ADVERSARY_STEP_SIZE = 0.01
STRONG_PENALTY = 0.1
# (name, L2 penalty, adversary step size), as in example 09.
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

    Returns:
        Features of shape `(n, 2 + NOISE_FEATURES)` with the core and the
        spurious feature first, labels in {-1, +1} of shape `(n,)`, and group
        indices of shape `(n,)`.
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


def worst_group_accuracy(
    weights: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Return the lowest group accuracy of every model, shape `(seeds, methods)`.

    A zero margin, which every model has before its first step, counts as
    half correct so the curves start at chance rather than at zero.
    """
    if features.ndim == 2:
        features = features.expand(len(weights), -1, -1)
        labels = labels.expand(len(weights), -1)
        groups = groups.expand(len(weights), -1)
    margins = torch.einsum("smd,snd->smn", weights, features) * labels[:, None, :]
    correct = (margins > 0).to(margins.dtype) + 0.5 * (margins == 0).to(margins.dtype)
    return group_losses(correct, groups[:, None, :], NUM_GROUPS).amin(-1)


def train(
    seeds: int,
    steps: int,
    every: int,
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Train every method on every seed jointly and record checkpoints.

    Args:
        seeds: Independent training sets.
        steps: Adam steps.
        every: Steps between checkpoints.
        test: Test features, labels and groups shared by every seed.

    Returns:
        Checkpoint `steps`, the informative `weights` of shape
        `(checkpoints, seeds, methods, 2)`, the worst-group accuracy on
        `test` and `train`, and the adversary's `minority` weight, each of
        shape `(checkpoints, seeds, methods)`.
    """
    datasets = [
        make_data(GROUP_SIZES, torch.Generator().manual_seed(SEED + seed))
        for seed in range(seeds)
    ]
    features, labels, groups = (
        torch.stack(parts) for parts in zip(*datasets, strict=True)
    )
    proportions = torch.tensor(GROUP_SIZES, dtype=torch.float32)
    proportions = proportions / proportions.sum()
    penalty = torch.tensor([method[1] for method in METHODS])
    step_size = torch.tensor([method[2] for method in METHODS])
    size, dimension = features.shape[1:]
    weights = torch.zeros(seeds, len(METHODS), dimension, requires_grad=True)
    adversary = OnlineGroupDRO(
        proportions.expand(seeds, len(METHODS), -1),
        step_size=step_size.expand(seeds, -1),
    )
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED + 1)
    history: dict[str, list[torch.Tensor]] = {
        "steps": [],
        "weights": [],
        "test": [],
        "train": [],
        "minority": [],
    }
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            with torch.no_grad():
                history["steps"].append(torch.tensor(step))
                history["weights"].append(weights[..., :2].clone())
                history["test"].append(worst_group_accuracy(weights, *test))
                history["train"].append(
                    worst_group_accuracy(weights, features, labels, groups)
                )
                history["minority"].append(
                    adversary.weights[..., list(MINORITY)].sum(-1)
                )
        if step == steps:
            break
        index = torch.randint(size, (seeds, BATCH), generator=generator)
        batch = torch.gather(features, 1, index[..., None].expand(-1, -1, dimension))
        margins = torch.einsum("smd,snd->smn", weights, batch)
        loss = F.softplus(-margins * torch.gather(labels, 1, index)[:, None, :])
        group_loss = group_losses(
            loss, torch.gather(groups, 1, index)[:, None, :], NUM_GROUPS
        )
        adversary.step(group_loss)
        objective = adversary(group_loss) + penalty * (weights**2).sum(-1)
        optimizer.zero_grad()
        objective.sum().backward()
        optimizer.step()
    return {name: torch.stack(values) for name, values in history.items()}


def render_group_dro_boundary(context: Context) -> None:
    """Animate each method's boundary, worst-group accuracy and minority weight."""
    theme = context.theme
    frames = context.frames(60)
    steps = context.size(3000, 30)
    test = make_data(
        (TEST_GROUP_SIZE,) * NUM_GROUPS, torch.Generator().manual_seed(SEED - 1)
    )
    history = train(context.size(3, 1), steps, max(steps // frames, 1), test)
    frames = history["steps"].shape[0]
    colors = (theme.muted, theme.series[1], theme.series[6], theme.series[0])
    minority_color = theme.series[7]
    minority_share = sum(GROUP_SIZES[group] for group in MINORITY) / sum(GROUP_SIZES)

    fig, (plane, accuracy, weight) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.1, 1.2, 1]}
    )
    features, labels, groups = test
    for group in range(NUM_GROUPS):
        chosen = (groups == group).nonzero()[:SHOWN_PER_GROUP, 0]
        minority = group in MINORITY
        plane.scatter(
            features[chosen, 0],
            features[chosen, 1],
            marker="o" if labels[chosen[0]] > 0 else "^",
            s=16,
            color=minority_color if minority else theme.secondary,
            alpha=0.6 if minority else 0.3,
            linewidths=0,
        )
    boundaries = []
    for (name, _, _), color in zip(METHODS, colors, strict=True):
        (line,) = plane.plot([], [], color=color, linewidth=1.8, label=name)
        boundaries.append(line)
    plane.set_xlim(-3.2, 3.2)
    plane.set_ylim(-2.4, 2.4)
    plane.set_xlabel("core feature  ($\\approx y$; circles $y = +1$, triangles $-1$)")
    plane.set_ylabel("spurious feature  ($\\approx a$)")
    plane.set_title("decision boundary on the informative features")
    plane.legend(
        handles=[
            *boundaries,
            plt.Line2D(
                [], [], marker="o", linestyle="", color=theme.secondary, alpha=0.5
            ),
            plt.Line2D([], [], marker="o", linestyle="", color=minority_color),
        ],
        labels=[
            *(method[0] for method in METHODS),
            "majority, $a = y$",
            "minority, $a = -y$",
        ],
        loc="upper center",
        fontsize=7,
        ncol=3,
    )

    steps_np = history["steps"].numpy()
    test_lines, train_lines, minority_lines, bands = [], [], [], []
    for index, ((name, _, step_size), color) in enumerate(
        zip(METHODS, colors, strict=True)
    ):
        (test_line,) = accuracy.plot([], [], color=color, label=name)
        (train_line,) = accuracy.plot(
            [], [], color=color, linestyle="--", linewidth=1.2
        )
        test_lines.append(test_line)
        train_lines.append(train_line)
        if step_size > 0:
            (minority_line,) = weight.plot([], [], color=color, label=name)
            minority_lines.append((index, minority_line))
    accuracy.set_xlim(0, steps)
    accuracy.set_ylim(0, 1.02)
    accuracy.set_xlabel("Adam step")
    accuracy.set_ylabel("worst-group accuracy")
    accuracy.set_title("worst group during training (dashed: train)")
    accuracy.legend(loc="lower right", fontsize=7.5)
    note = accuracy.text(
        0.02, 0.03, "", transform=accuracy.transAxes, ha="left", va="bottom", fontsize=9
    )

    weight.axhline(minority_share, color=theme.ink, linestyle=":", linewidth=1.2)
    weight.text(
        steps,
        minority_share + 0.01,
        "training share  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    weight.set_xlim(0, steps)
    weight.set_ylim(0, float(history["minority"].max()) * 1.2)
    weight.set_xlabel("Adam step")
    weight.set_ylabel("adversary weight on the minority groups")
    weight.set_title("online group weights (band: range over seeds)")
    weight.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        upto = frame + 1
        for index, line in enumerate(boundaries):
            core, spurious = history["weights"][frame, 0, index].tolist()
            direction = torch.tensor([-spurious, core])
            direction = direction / direction.norm().clamp(min=1e-12)
            line.set_data(
                [-8 * direction[0], 8 * direction[0]],
                [-8 * direction[1], 8 * direction[1]],
            )
        for index, (test_line, train_line) in enumerate(
            zip(test_lines, train_lines, strict=True)
        ):
            test_line.set_data(
                steps_np[:upto], history["test"][:upto, :, index].mean(-1)
            )
            train_line.set_data(
                steps_np[:upto], history["train"][:upto, :, index].mean(-1)
            )
        for band in bands:
            band.remove()
        bands.clear()
        for index, line in minority_lines:
            values = history["minority"][:upto, :, index]
            line.set_data(steps_np[:upto], values.mean(-1))
            bands.append(
                weight.fill_between(
                    steps_np[:upto],
                    values.amin(-1).numpy(),
                    values.amax(-1).numpy(),
                    color=line.get_color(),
                    alpha=0.15,
                    linewidth=0,
                )
            )
        note.set_text(f"step {int(history['steps'][frame])}")

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "group-dro-boundary", fps=10
    )


FIGURES = (
    FigureSpec(
        name="group-dro-boundary",
        page="examples/group_dro_training.md",
        outputs=("group-dro-boundary-{theme}.gif",),
        render=render_group_dro_boundary,
    ),
)
