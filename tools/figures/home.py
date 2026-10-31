"""The home-page hero: where each ambiguity set's adversary moves the mass."""

from __future__ import annotations

from matplotlib import pyplot as plt

from tools.figures._base import (
    SET_LABELS,
    Context,
    FigureSpec,
    set_color,
    worst_case_distributions,
)
from tools.figures.problems import demo_instance
from tools.figures.sets import FAMILIES, budget_schedule

HOLD_FRAMES = 10


def render_mass_migration(context: Context) -> None:
    """Animate the seven worst-case distributions as every budget grows."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss, points = instance.nominal, instance.loss, instance.points
    frames = context.frames(60)
    fractions = budget_schedule(frames)
    results = []
    for family in FAMILIES:
        radii = fractions * family.saturation(instance)
        values, maximizers = worst_case_distributions(
            lambda radius, family=family: family.build(instance, radius), loss, radii
        )
        results.append((values.numpy(), maximizers.numpy()))

    fig, axes_grid = plt.subplots(2, 4, figsize=(10.0, 5.2), dpi=90, sharex=True)
    axes_list = list(axes_grid.ravel())
    value_axes = axes_list[-1]
    width = float(points[1] - points[0]) * 0.8
    bar_sets = []
    for axes, family, (values, maximizers) in zip(
        axes_list[:-1], FAMILIES, results, strict=True
    ):
        color = set_color(theme, family.key)
        axes.bar(
            points.numpy(), nominal.numpy(), width=width, color=theme.muted, alpha=0.35
        )
        bars = axes.bar(points.numpy(), maximizers[0], width=width * 0.62, color=color)
        bar_sets.append(bars)
        axes.set_ylim(0, 1.0)
        axes.set_title(SET_LABELS[family.key], fontsize=10, loc="center")
        axes.tick_params(labelsize=7)
        value_axes.plot(
            fractions.numpy(), values, color=color, label=SET_LABELS[family.key]
        )
    for axes in axes_grid[1]:
        axes.set_xlabel("scenario (loss rises to the right)", fontsize=8)
    for axes in axes_grid[:, 0]:
        axes.set_ylabel("probability mass", fontsize=8)
    markers = [
        value_axes.plot([], [], "o", color=set_color(theme, family.key), markersize=5)[
            0
        ]
        for family in FAMILIES
    ]
    value_axes.axhline(float(nominal @ loss), color=theme.muted, linewidth=1)
    value_axes.axhline(float(loss.max()), color=theme.muted, linewidth=1)
    value_axes.set_xlim(0, 1)
    value_axes.set_ylim(float(nominal @ loss) - 0.1, float(loss.max()) + 0.1)
    value_axes.set_title("worst-case value", fontsize=10, loc="center")
    value_axes.set_xlabel("budget fraction", fontsize=8)
    value_axes.tick_params(labelsize=7)
    value_axes.legend(fontsize=6.5, ncol=2, loc="upper left")
    title = fig.suptitle("", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        fraction = float(fractions[frame])
        for bars, (values, maximizers), marker in zip(
            bar_sets, results, markers, strict=True
        ):
            for bar, height in zip(bars, maximizers[frame], strict=True):
                bar.set_height(height)
            marker.set_data([fraction], [values[frame]])
        title.set_text(
            "one nominal (grey), one loss, seven adversaries  -  "
            f"budget at {100 * fraction:3.0f}% of each ball's saturation"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "mass-migration", fps=10)


FIGURES = (
    FigureSpec(
        name="mass-migration",
        page="index.md",
        outputs=("mass-migration-{theme}.gif",),
        render=render_mass_migration,
    ),
)
