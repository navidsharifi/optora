r"""Conditional value-at-risk, animated over its level.

The first animation shows CVaR as the mean of the upper tail. The other two
belong to the tutorial page on recovering CVaR from a phi-divergence ball:
they set the CVaR ball beside the total-variation ball of radius
$1 - \alpha$ that contains it, first on the probability simplex and then as
likelihood ratios to the nominal.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation

from optora.dro import CVaRAmbiguitySet, TotalVariationAmbiguitySet
from tools.figures._base import (
    Context,
    FigureSpec,
    Theme,
    set_color,
    worst_case_distributions,
)
from tools.figures.geometry import (
    VERTICES,
    _prepare_axes,
    barycentric_grid,
    simplex_instance,
    to_xy,
)
from tools.figures.problems import Instance, demo_instance

HOLD_FRAMES = 10


def render_cvar_tail(context: Context) -> None:
    """Animate the tail the CVaR adversary fills as alpha falls from one to zero."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss = instance.nominal, instance.loss
    frames = context.frames(64)
    # Sweep alpha from 1 (the nominal mean) down to the mass of the top scenario.
    top_mass = float(nominal[loss == loss.max()].sum())
    alphas = torch.logspace(0, np.log10(top_mass), frames, dtype=loss.dtype)
    values, maximizers = worst_case_distributions(
        lambda alpha: CVaRAmbiguitySet(nominal, alpha), loss, alphas
    )
    color = set_color(theme, "cvar")
    order = torch.argsort(loss, descending=True)
    sorted_loss = loss[order].numpy()
    sorted_nominal = nominal[order].numpy()
    edges = np.concatenate([[0.0], np.cumsum(sorted_nominal)])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    # Scenarios sorted by loss, each as wide as its nominal mass: the quantile
    # function of the loss under the nominal.
    left.bar(
        edges[:-1],
        sorted_loss,
        width=sorted_nominal,
        align="edge",
        color=theme.muted,
        alpha=0.35,
    )
    tail = left.bar(
        edges[:-1],
        sorted_loss,
        width=sorted_nominal,
        align="edge",
        color=color,
    )
    for patch in tail:
        patch.set_width(0.0)
    level = left.axhline(values[0].item(), color=theme.ink, linewidth=1.2)
    cut = left.axvline(1.0, color=theme.ink, linewidth=1.2)
    left.set_xlim(0, 1)
    left.set_ylim(0, float(loss.max()) * 1.12)
    left.set_xlabel("cumulative nominal mass, scenarios sorted by loss")
    left.set_ylabel("scenario loss")
    left.set_title("the upper $\\alpha$-tail, with likelihood ratio $1/\\alpha$")
    note = left.text(
        0.98, 0.95, "", transform=left.transAxes, ha="right", va="top", fontsize=9
    )

    right.plot(alphas.numpy(), values.numpy(), color=color)
    (marker,) = right.plot([], [], "o", color=color, markersize=8)
    right.axhline(float(nominal @ loss), color=theme.muted, linewidth=1)
    right.axhline(float(loss.max()), color=theme.muted, linewidth=1)
    right.set_xscale("log")
    right.invert_xaxis()
    right.set_xlabel("tail probability $\\alpha$  (log, decreasing)")
    right.set_ylabel("$\\mathrm{CVaR}_\\alpha$")
    right.set_title("value vs level")
    right.text(
        1.0,
        float(nominal @ loss) + 0.03,
        "$\\alpha = 1$: nominal mean",
        fontsize=8,
        color=theme.secondary,
        ha="right",
    )
    right.text(
        1.0,
        float(loss.max()) - 0.03,
        "$\\alpha \\to 0$: max loss",
        fontsize=8,
        color=theme.secondary,
        ha="right",
        va="top",
    )
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        alpha = alphas[frame].item()
        # The maximizer fills whole scenarios in loss order until mass alpha,
        # so the filled width of each bar is its share of the tail.
        filled = maximizers[frame][order].numpy() * alpha
        for patch, width in zip(tail, filled, strict=True):
            patch.set_width(min(width, patch.get_x() + 1.0))
        level.set_ydata([values[frame].item()] * 2)
        cut.set_xdata([alpha] * 2)
        marker.set_data([alpha], [values[frame].item()])
        note.set_text(f"$\\alpha$ = {alpha:.3f}\nCVaR = {values[frame].item():.3f}")

    context.save_animation(fig, update, frames, "cvar-tail", fps=10)


BridgePaths = dict[str, tuple[torch.Tensor, torch.Tensor]]


def bridge_paths(instance: Instance, alphas: torch.Tensor) -> BridgePaths:
    r"""Return the value and maximizer of each set the bridge figures compare.

    Args:
        instance: Problem instance.
        alphas: CVaR levels of shape `(A,)`.

    Returns:
        `(values, maximizers)` for the CVaR ball at each level (`cvar`), the
        total-variation ball of radius $1 - \alpha$ that contains it
        (`tv_matched`), and the total-variation ball of radius
        $(1 - \alpha) / \alpha$ sometimes suggested in its place
        (`tv_suggested`).
    """
    nominal, loss = instance.nominal, instance.loss
    return {
        "cvar": worst_case_distributions(
            lambda alpha: CVaRAmbiguitySet(nominal, alpha), loss, alphas
        ),
        "tv_matched": worst_case_distributions(
            lambda alpha: TotalVariationAmbiguitySet(nominal, 1.0 - alpha),
            loss,
            alphas,
        ),
        "tv_suggested": worst_case_distributions(
            lambda alpha: TotalVariationAmbiguitySet(nominal, (1.0 - alpha) / alpha),
            loss,
            alphas,
        ),
    }


def draw_bridge_values(
    axes: Axes, theme: Theme, instance: Instance, alphas: np.ndarray, paths: BridgePaths
) -> list[Line2D]:
    r"""Draw the bridge value curves over alpha and return their moving markers.

    The hollow circles are $\alpha\,\mathrm{CVaR}_\alpha + (1 - \alpha)
    \max \ell$, computed from the CVaR curve alone; they land on the
    total-variation curve of radius $1 - \alpha$ because the two are equal.
    """
    cvar_color = set_color(theme, "cvar")
    tv_color = set_color(theme, "total_variation")
    max_loss = float(instance.loss.max())
    cvar_values = paths["cvar"][0].numpy()
    axes.axhline(max_loss, color=theme.muted, linewidth=1)
    axes.plot(
        alphas,
        paths["tv_suggested"][0].numpy(),
        "--",
        color=tv_color,
        label="TV, radius $(1-\\alpha)/\\alpha$",
    )
    axes.plot(
        alphas,
        paths["tv_matched"][0].numpy(),
        color=tv_color,
        label="TV, radius $1-\\alpha$",
    )
    axes.plot(
        alphas,
        alphas * cvar_values + (1.0 - alphas) * max_loss,
        "o",
        color=tv_color,
        markevery=6,
        markersize=5,
        markerfacecolor="none",
        markeredgecolor=tv_color,
        label="$\\alpha\\,\\mathrm{CVaR}_\\alpha + (1-\\alpha)\\max\\,\\ell$",
    )
    axes.plot(
        alphas,
        cvar_values,
        color=cvar_color,
        linewidth=2.6,
        label="$\\mathrm{CVaR}_\\alpha$ (phi ball)",
    )
    axes.set_xscale("log")
    axes.invert_xaxis()
    axes.set_xlabel("tail probability $\\alpha$  (log, decreasing)")
    axes.set_ylabel("worst-case expected loss")
    axes.set_title("value vs level")
    axes.legend(loc="lower right", fontsize=7.5)
    return [
        axes.plot([], [], marker, color=color, markersize=8)[0]
        for marker, color in (("o", cvar_color), ("o", tv_color), ("D", tv_color))
    ]


def render_cvar_tv_simplex(context: Context) -> None:
    r"""Animate the CVaR ball inside the TV ball of radius $1 - \alpha$."""
    theme = context.theme
    instance = simplex_instance()
    nominal = instance.nominal
    frames = context.frames(60)
    alphas = torch.logspace(0.0, -1.0, frames, dtype=torch.float64)
    paths = bridge_paths(instance, alphas)
    cvar_color = set_color(theme, "cvar")
    tv_color = set_color(theme, "total_variation")

    grid = barycentric_grid(context.size(140, 30))
    xy = to_xy(grid)
    triangulation = Triangulation(xy[:, 0], xy[:, 1])
    ratio = (grid / nominal).amax(dim=-1).numpy()
    total_variation = (0.5 * (grid - nominal).abs().sum(dim=-1)).numpy()
    ratio_ceiling = float(ratio.max()) * 0.9999
    tv_ceiling = float(total_variation.max())

    fig, (simplex, values) = plt.subplots(
        1, 2, figsize=(10.4, 4.7), dpi=90, gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    _prepare_axes(
        simplex, instance, "CVaR ball inside the TV ball of radius $1-\\alpha$"
    )
    simplex.plot(
        [*VERTICES[:, 0], VERTICES[0, 0]],
        [*VERTICES[:, 1], VERTICES[0, 1]],
        color=theme.axis,
        linewidth=1.0,
    )
    simplex.tricontour(
        triangulation,
        (grid @ instance.loss).numpy(),
        levels=9,
        colors=[theme.grid],
        linewidths=0.8,
    )
    simplex.plot(*to_xy(nominal), "o", color=theme.muted, markersize=7, zorder=6)
    simplex.text(
        0.5,
        -0.16,
        "dashed: TV radius $(1-\\alpha)/\\alpha$    grey: lines of equal expected loss",
        ha="center",
        fontsize=8,
        color=theme.secondary,
    )
    note = simplex.text(-0.04, 0.9, "", fontsize=9, va="top")
    cvar_path = to_xy(paths["cvar"][1])
    tv_paths = (to_xy(paths["tv_matched"][1]), to_xy(paths["tv_suggested"][1]))
    (trail,) = simplex.plot([], [], color=cvar_color, linewidth=1.5, zorder=7)
    (cvar_point,) = simplex.plot([], [], "o", color=cvar_color, markersize=9, zorder=9)
    tv_points = (
        simplex.plot([], [], "o", color=tv_color, markersize=9, zorder=8)[0],
        simplex.plot(
            [],
            [],
            "D",
            color=tv_color,
            markerfacecolor="none",
            markeredgecolor=tv_color,
            markersize=9,
            zorder=8,
        )[0],
    )
    markers = draw_bridge_values(values, theme, instance, alphas.numpy(), paths)
    fig.tight_layout(w_pad=2.0)
    artists: list[object] = []

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        alpha = float(alphas[frame])
        for artist in artists:
            artist.remove()
        artists.clear()
        artists.append(
            simplex.tricontourf(
                triangulation,
                ratio,
                levels=[0.0, min(1.0 / alpha, ratio_ceiling)],
                colors=[cvar_color],
                alpha=0.35,
            )
        )
        if 1.0 / alpha < ratio_ceiling:
            artists.append(
                simplex.tricontour(
                    triangulation,
                    ratio,
                    levels=[1.0 / alpha],
                    colors=[cvar_color],
                    linewidths=1.6,
                )
            )
        for radius, style in (
            (1.0 - alpha, "solid"),
            ((1.0 - alpha) / alpha, "dashed"),
        ):
            if 0.0 < radius < tv_ceiling:
                artists.append(
                    simplex.tricontour(
                        triangulation,
                        total_variation,
                        levels=[radius],
                        colors=[tv_color],
                        linewidths=1.6,
                        linestyles=style,
                    )
                )
        trail.set_data(cvar_path[: frame + 1, 0], cvar_path[: frame + 1, 1])
        cvar_point.set_data([cvar_path[frame, 0]], [cvar_path[frame, 1]])
        for point, path in zip(tv_points, tv_paths, strict=True):
            point.set_data([path[frame, 0]], [path[frame, 1]])
        for marker, key in zip(
            markers, ("cvar", "tv_matched", "tv_suggested"), strict=True
        ):
            marker.set_data([alpha], [float(paths[key][0][frame])])
        note.set_text(f"$\\alpha$ = {alpha:.3f}")

    context.save_animation(fig, update, frames + HOLD_FRAMES, "cvar-tv-simplex", fps=10)


def render_cvar_tv_ratio(context: Context) -> None:
    """Animate the likelihood ratios of the CVaR and TV worst cases over alpha."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss = instance.nominal, instance.loss
    frames = context.frames(64)
    alphas = torch.logspace(0.0, np.log10(0.05), frames, dtype=torch.float64)
    paths = bridge_paths(instance, alphas)
    order = torch.argsort(loss)
    positions = np.arange(instance.size)
    keys = ("cvar", "tv_matched")
    ratios = {key: (paths[key][1] / nominal)[:, order].numpy() for key in keys}
    floor = 0.03
    ceiling = 1.6 * max(float(ratio.max()) for ratio in ratios.values())
    titles = (
        "CVaR: every tail ratio rises to $1/\\alpha$",
        "TV, radius $1-\\alpha$: the surplus lands on one scenario",
    )
    colors = (set_color(theme, "cvar"), set_color(theme, "total_variation"))

    fig, panels = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=90, sharey=True)
    bars = []
    caps = []
    notes = []
    for axes, title, color in zip(panels, titles, colors, strict=True):
        axes.bar(positions, 1.0, width=0.8, color=theme.muted, alpha=0.25)
        bars.append(
            axes.bar(
                positions,
                np.zeros(instance.size),
                width=0.6,
                bottom=floor,
                color=color,
            )
        )
        axes.axhline(1.0, color=theme.muted, linewidth=1.0)
        caps.append(
            axes.axhline(1.0, color=theme.ink, linewidth=1.2, linestyle=(0, (4, 3)))
        )
        axes.set_yscale("log")
        axes.set_ylim(floor, ceiling)
        axes.set_xticks(
            positions,
            [f"{value:.2f}" for value in loss[order].numpy()],
            rotation=60,
            fontsize=7.5,
        )
        axes.set_xlabel("scenario, labelled by its loss (sorted)")
        axes.set_title(title, fontsize=10)
        notes.append(
            axes.text(0.03, 0.96, "", transform=axes.transAxes, va="top", fontsize=9)
        )
    panels[0].set_ylabel("likelihood ratio $q^\\star_i / p_i$  (log)")
    panels[0].text(-0.4, 1.0, "nominal", va="bottom", fontsize=8, color=theme.secondary)
    fig.tight_layout(w_pad=1.5)
    max_loss = float(loss.max())

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        alpha = float(alphas[frame])
        for patches, key in zip(bars, keys, strict=True):
            for patch, height in zip(patches, ratios[key][frame], strict=True):
                patch.set_height(max(float(height), floor) - floor)
        for cap in caps:
            cap.set_ydata([1.0 / alpha] * 2)
        cvar_value = float(paths["cvar"][0][frame])
        tv_value = float(paths["tv_matched"][0][frame])
        mixture = alpha * cvar_value + (1.0 - alpha) * max_loss
        notes[0].set_text(
            f"$\\alpha$ = {alpha:.3f}    CVaR = {cvar_value:.3f}\n"
            "dashed line: the cap $1/\\alpha$"
        )
        notes[1].set_text(
            f"TV value = {tv_value:.3f}\n"
            f"$\\alpha\\,$CVaR + $(1-\\alpha)\\,$max loss = {mixture:.3f}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "cvar-tv-ratio", fps=10)


FIGURES = (
    FigureSpec(
        name="cvar-tail",
        page="formulations.md",
        outputs=("cvar-tail-{theme}.gif",),
        render=render_cvar_tail,
    ),
    FigureSpec(
        name="cvar-tv-simplex",
        page="tutorials/cvar_from_phi_divergence.md",
        outputs=("cvar-tv-simplex-{theme}.gif",),
        render=render_cvar_tv_simplex,
    ),
    FigureSpec(
        name="cvar-tv-ratio",
        page="tutorials/cvar_from_phi_divergence.md",
        outputs=("cvar-tv-ratio-{theme}.gif",),
        render=render_cvar_tv_ratio,
    ),
)
