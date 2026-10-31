"""Float32 accuracy of every ambiguity set against a float64 solve."""

from __future__ import annotations

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tools.figures._base import SET_LABELS, Context, FigureSpec
from tools.figures.problems import cast, random_instance
from tools.figures.sets import FAMILIES

FRACTIONS = (1e-3, 1e-2, 0.1, 0.3, 0.6, 0.9, 1.0)
DOCUMENTED_BOUND = 128.0


def float32_errors(size: int) -> np.ndarray:
    """Return the float32 error of each family at each budget fraction.

    The error is `|v32 - v64| / max(|v64|, spread)` in units of the float32
    machine epsilon, the quantity `docs/numerics.md` bounds by 128.
    """
    instance = random_instance(size)
    instance32 = cast(instance, torch.float32)
    spread = float(instance.loss.max() - instance.loss.min())
    epsilon = torch.finfo(torch.float32).eps
    errors = np.zeros((len(FAMILIES), len(FRACTIONS)))
    for row, family in enumerate(FAMILIES):
        saturation = family.saturation(instance)
        radii = torch.tensor(FRACTIONS, dtype=torch.float64) * saturation
        with torch.no_grad():
            exact = family.build(instance, radii).worst_case_expectation(
                instance.loss.expand(len(FRACTIONS), -1)
            )
            single = family.build(
                instance32, radii.to(torch.float32)
            ).worst_case_expectation(instance32.loss.expand(len(FRACTIONS), -1))
        scale = torch.maximum(exact.abs(), torch.tensor(spread, dtype=torch.float64))
        errors[row] = ((single.double() - exact).abs() / scale / epsilon).numpy()
    return errors


def render_float32_heatmap(context: Context) -> None:
    """Draw the float32-vs-float64 error heatmap."""
    theme = context.theme
    errors = float32_errors(context.size(256, 16))
    cmap = LinearSegmentedColormap.from_list("sequential", list(theme.sequential))
    fig, axes = plt.subplots(figsize=(8.6, 4.0))
    sns.heatmap(
        errors,
        ax=axes,
        cmap=cmap,
        vmin=0,
        vmax=DOCUMENTED_BOUND / 4,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.5},
        linewidths=2,
        linecolor=theme.surface,
        cbar_kws={"fraction": 0.03, "pad": 0.02, "label": "error in float32 ulps"},
        xticklabels=[f"{fraction:g}" for fraction in FRACTIONS],
        yticklabels=[SET_LABELS[family.key] for family in FAMILIES],
    )
    colorbar = axes.collections[0].colorbar
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=theme.muted, labelsize=8)
    colorbar.ax.yaxis.label.set_color(theme.secondary)
    axes.tick_params(rotation=0, labelsize=8.5)
    axes.set_xlabel("radius, as a fraction of the ball's saturation radius")
    axes.set_title(
        "float32 error against a float64 solve of the same inputs "
        f"(documented bound: {DOCUMENTED_BOUND:g} ulps), n = {context.size(256, 16)}"
    )
    fig.tight_layout()
    context.save_static(fig, "float32-errors")


FIGURES = (
    FigureSpec(
        name="float32-errors",
        page="numerics.md",
        outputs=("float32-errors-{theme}.svg",),
        render=render_float32_heatmap,
    ),
)
