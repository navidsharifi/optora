"""Seven ambiguity-set geometries on the probability simplex, over the budget."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation

from tools.figures._base import (
    SET_LABELS,
    Context,
    FigureSpec,
    set_color,
    worst_case_distributions,
)
from tools.figures.problems import Instance
from tools.figures.sets import FAMILIES, budget_schedule

VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
HOLD_FRAMES = 10


def simplex_instance() -> Instance:
    """Return the three-scenario instance drawn on the simplex."""
    points = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    nominal = torch.tensor([0.55, 0.30, 0.15], dtype=torch.float64)
    loss = torch.tensor([0.20, 0.55, 1.00], dtype=torch.float64)
    cost = torch.abs(points[:, None] - points[None, :])
    return Instance(points=points, nominal=nominal, loss=loss, cost=cost)


def barycentric_grid(resolution: int) -> torch.Tensor:
    """Return every distribution on the simplex with denominator `resolution`."""
    i, j = np.meshgrid(np.arange(resolution + 1), np.arange(resolution + 1))
    i, j = i.ravel(), j.ravel()
    keep = i + j <= resolution
    grid = np.stack([i[keep], j[keep], resolution - i[keep] - j[keep]], axis=1)
    return torch.as_tensor(grid / resolution, dtype=torch.float64)


def to_xy(q: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map distributions of shape `(..., 3)` to simplex coordinates."""
    return np.asarray(q) @ VERTICES


@dataclass
class SimplexPanel:
    """The drawable state of one family's panel."""

    axes: Axes
    color: str
    divergence: np.ndarray
    path: np.ndarray
    values: np.ndarray
    artists: list[object]
    trail: object
    point: object


def _prepare_axes(axes: Axes, instance: Instance, title: str) -> None:
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_xlim(-0.08, 1.08)
    axes.set_ylim(-0.12, 0.98)
    axes.set_title(title, fontsize=10, loc="center")
    for vertex, loss in zip(VERTICES, instance.loss.numpy(), strict=True):
        offset = np.array([0.0, -0.07]) if vertex[1] == 0 else np.array([0.0, 0.03])
        axes.text(
            *(vertex + offset),
            f"$\\ell = {loss:.2f}$",
            ha="center",
            va="center",
            fontsize=8,
        )


def render_simplex(
    context: Context,
    stem: str,
    resolution: int,
    fractions: torch.Tensor,
    animate: bool,
) -> None:
    """Draw every family's ball and worst case on the simplex at `fractions`.

    Args:
        context: Rendering context.
        stem: Output file stem.
        resolution: Barycentric grid resolution.
        fractions: Budget fractions of each family's saturation radius, one
            per frame; a static figure draws the last one.
        animate: Whether to write a GIF over `fractions` or one SVG.
    """
    theme = context.theme
    instance = simplex_instance()
    grid = barycentric_grid(resolution)
    xy = to_xy(grid)
    triangulation = Triangulation(xy[:, 0], xy[:, 1])
    frames = int(fractions.shape[0])
    nominal_xy = to_xy(instance.nominal)

    fig, axes_grid = plt.subplots(2, 4, figsize=(10.0, 5.4), dpi=90)
    axes_list = list(axes_grid.ravel())
    value_axes = axes_list[-1]
    panels: list[SimplexPanel] = []
    for axes, family in zip(axes_list[:-1], FAMILIES, strict=True):
        color = set_color(theme, family.key)
        saturation = family.saturation(instance)
        radii = fractions * saturation
        values, maximizers = worst_case_distributions(
            lambda radius, family=family: family.build(instance, radius),
            instance.loss,
            radii,
        )
        divergence = family.divergence(instance, grid).detach().numpy()
        _prepare_axes(axes, instance, SET_LABELS[family.key])
        axes.plot(
            [*VERTICES[:, 0], VERTICES[0, 0]],
            [*VERTICES[:, 1], VERTICES[0, 1]],
            color=theme.axis,
            linewidth=1.0,
        )
        axes.plot(*nominal_xy, "o", color=theme.muted, markersize=7, zorder=5)
        (trail,) = axes.plot([], [], color=color, linewidth=1.5, alpha=0.8, zorder=6)
        (point,) = axes.plot([], [], "o", color=color, markersize=8, zorder=7)
        value_axes.plot(
            fractions.numpy(),
            values.numpy(),
            color=color,
            label=SET_LABELS[family.key],
        )
        panels.append(
            SimplexPanel(
                axes=axes,
                color=color,
                divergence=divergence,
                path=to_xy(maximizers),
                values=values.numpy(),
                artists=[],
                trail=trail,
                point=point,
            )
        )
    markers = [
        value_axes.plot([], [], "o", color=panel.color, markersize=6)[0]
        for panel in panels
    ]
    value_axes.axhline(
        float(instance.nominal @ instance.loss), color=theme.muted, linewidth=1
    )
    value_axes.axhline(float(instance.loss.max()), color=theme.muted, linewidth=1)
    value_axes.set_xlim(0, 1)
    value_axes.set_xlabel("budget fraction of saturation")
    value_axes.set_ylabel("worst-case expected loss")
    value_axes.set_title("value", fontsize=10, loc="center")
    value_axes.legend(loc="upper left", fontsize=7, ncol=2)
    budget = (
        ""
        if animate
        else f"  -  budget at {100 * float(fractions[-1]):.0f}% of saturation"
    )
    fig.suptitle(
        "the same nominal (grey dot), the same loss, seven ambiguity sets" + budget,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    level_floor = -1e-9

    def draw(frame: int) -> None:
        fraction = float(fractions[frame])
        for panel, family, marker in zip(panels, FAMILIES, markers, strict=True):
            radius = fraction * family.saturation(instance)
            for artist in panel.artists:
                artist.remove()
            panel.artists.clear()
            ceiling = float(panel.divergence.max())
            fill_level = min(radius, ceiling * 0.9999)
            filled = panel.axes.tricontourf(
                triangulation,
                panel.divergence,
                levels=[level_floor, fill_level],
                colors=[panel.color],
                alpha=0.28,
            )
            panel.artists.append(filled)
            if radius < ceiling:
                boundary = panel.axes.tricontour(
                    triangulation,
                    panel.divergence,
                    levels=[radius],
                    colors=[panel.color],
                    linewidths=1.6,
                )
                panel.artists.append(boundary)
            panel.trail.set_data(panel.path[: frame + 1, 0], panel.path[: frame + 1, 1])
            panel.point.set_data([panel.path[frame, 0]], [panel.path[frame, 1]])
            marker.set_data([fraction], [panel.values[frame]])

    if not animate:
        draw(frames - 1)
        context.save_static(fig, stem)
        return

    def update(frame: int) -> None:
        draw(min(frame, frames - 1))

    context.save_animation(fig, update, frames + HOLD_FRAMES, stem, fps=10)


def render_animation(context: Context) -> None:
    """Animate the balls growing to saturation."""
    render_simplex(
        context,
        stem="simplex-geometry",
        resolution=context.size(110, 30),
        fractions=budget_schedule(context.frames(60)),
        animate=True,
    )


def render_gallery(context: Context) -> None:
    """Draw every ball at forty percent of its saturation radius."""
    render_simplex(
        context,
        stem="simplex-gallery",
        resolution=context.size(140, 30),
        fractions=torch.tensor([0.4], dtype=torch.float64),
        animate=False,
    )


FIGURES: Sequence[FigureSpec] = (
    FigureSpec(
        name="simplex-geometry",
        page="formulations.md",
        outputs=("simplex-geometry-{theme}.gif",),
        render=render_animation,
    ),
    FigureSpec(
        name="simplex-gallery",
        page="formulations.md",
        outputs=("simplex-gallery-{theme}.svg",),
        render=render_gallery,
    ),
)
