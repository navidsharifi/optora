r"""Radius sweeps of the KL and total-variation balls, animated.

Both animations belong to the first two example pages and draw their
four-scenario instance. The first grows the KL radius and watches the
worst-case distribution tilt toward the costly scenarios while the
constraint $D_{\mathrm{KL}}(q^\star \,\|\, p) = \rho$ stays tight, until the
ball contains the point mass on the worst scenario. The second grows the
total-variation radius and watches the adversary drain the cheapest
scenarios, one at a time, into the most expensive one.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from optora.dro import KLAmbiguitySet, TotalVariationAmbiguitySet
from tools.figures._base import (
    Context,
    FigureSpec,
    Theme,
    set_bar_heights,
    set_color,
    worst_case_distributions,
)

HOLD_FRAMES = 10
# The four-scenario instance examples 01 and 02 share.
NOMINAL = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
LOSS = torch.tensor([0.5, 1.5, 3.0, 8.0], dtype=torch.float64)
POSITIONS = np.arange(NOMINAL.shape[0])


def _scenario_axes(axes: Axes, theme: Theme, colors: list[str]) -> BarContainer:
    """Draw the nominal as wide grey bars and return the adversary's bars."""
    axes.bar(POSITIONS, NOMINAL.numpy(), width=0.8, color=theme.muted, alpha=0.3)
    bars = axes.bar(POSITIONS, NOMINAL.numpy(), width=0.5, color=colors)
    axes.set_xticks(POSITIONS, [f"{value:.1f}" for value in LOSS.tolist()])
    axes.set_xlabel("scenario, labelled by its loss")
    axes.set_ylabel("probability mass")
    axes.set_ylim(0, 1.0)
    return bars


def render_kl_radius(context: Context) -> None:
    r"""Animate the KL adversary, the worst-case value and the tight constraint."""
    theme = context.theme
    frames = context.frames(56)
    radii = torch.logspace(-3, 1, frames, dtype=torch.float64)
    values, maximizers = worst_case_distributions(
        lambda radius: KLAmbiguitySet(NOMINAL, radius), LOSS, radii
    )
    divergence = torch.special.xlogy(maximizers, maximizers / NOMINAL).sum(-1)
    nominal_value = float(NOMINAL @ LOSS)
    max_loss = float(LOSS.max())
    saturation = -float(torch.log(NOMINAL[LOSS.argmax()]))
    color = set_color(theme, "kl")

    fig, (support, value, constraint) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.1, 1.3, 1]}
    )
    bars = _scenario_axes(support, theme, [color] * NOMINAL.shape[0])
    support.set_title("worst-case distribution $q^\\star$ (grey: nominal)")
    note = support.text(
        0.98, 0.97, "", transform=support.transAxes, ha="right", va="top", fontsize=9
    )

    radii_np = radii.numpy()
    value.plot(
        radii_np, values.numpy(), color=color, label="$\\sup_q\\, \\mathbb{E}_q[\\ell]$"
    )
    value.axhline(
        nominal_value,
        color=theme.muted,
        linewidth=1,
        linestyle="--",
        label="$\\mathbb{E}_p[\\ell]$",
    )
    value.axhline(
        max_loss,
        color=theme.muted,
        linewidth=1,
        linestyle=":",
        label="$\\max_i \\ell_i$",
    )
    value.axvline(
        saturation,
        color=theme.ink,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="$\\rho = -\\log p_{\\arg\\max \\ell}$",
    )
    (marker,) = value.plot([], [], "o", color=color, markersize=8)
    value.set_xscale("log")
    value.set_xlim(radii_np[0], radii_np[-1])
    value.set_ylim(0, max_loss * 1.12)
    value.set_xlabel("KL radius $\\rho$  (log)")
    value.set_ylabel("worst-case expected loss")
    value.set_title("value: nominal mean to maximum loss")
    value.legend(loc="upper left", fontsize=7.5)

    constraint.plot(
        radii_np,
        radii_np,
        color=theme.muted,
        linewidth=1,
        linestyle="--",
        label="$\\rho$",
    )
    constraint.plot(
        radii_np,
        divergence.numpy(),
        color=color,
        label="$D_{\\mathrm{KL}}(q^\\star \\,\\|\\, p)$",
    )
    (dot,) = constraint.plot([], [], "o", color=color, markersize=8)
    constraint.set_xscale("log")
    constraint.set_yscale("log")
    constraint.set_xlim(radii_np[0], radii_np[-1])
    constraint.set_ylim(radii_np[0] * 0.8, radii_np[-1] * 1.2)
    constraint.set_xlabel("KL radius $\\rho$  (log)")
    constraint.set_ylabel("divergence of the adversary  (log)")
    constraint.set_title("the constraint binds until saturation")
    constraint.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        radius = float(radii[frame])
        set_bar_heights(bars, maximizers[frame])
        marker.set_data([radius], [float(values[frame])])
        dot.set_data([radius], [float(divergence[frame])])
        regime = (
            "tight: $D_{\\mathrm{KL}}(q^\\star\\|p) = \\rho$"
            if radius < saturation
            else "saturated: $q^\\star$ is the point mass"
        )
        note.set_text(
            f"$\\rho$ = {radius:.3g}\nvalue = {float(values[frame]):.3f}\n{regime}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "kl-radius", fps=10)


def render_tv_shift(context: Context) -> None:
    """Animate the total-variation adversary draining scenarios as the radius grows."""
    theme = context.theme
    frames = context.frames(56)
    movable = float(1.0 - NOMINAL[LOSS.argmax()])
    radii = torch.linspace(0.0, movable, frames, dtype=torch.float64)
    values, maximizers = worst_case_distributions(
        lambda radius: TotalVariationAmbiguitySet(NOMINAL, radius), LOSS, radii
    )
    moved = 0.5 * (maximizers - NOMINAL).abs().sum(-1)
    # One shade per scenario, darker for a higher loss.
    colors = [theme.sequential[index] for index in (2, 3, 4, 6)]
    cumulative = torch.cat(
        [torch.zeros(frames, 1, dtype=torch.float64), maximizers.cumsum(-1)], dim=-1
    ).numpy()
    color = set_color(theme, "total_variation")

    fig, (support, stack, value) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.1, 1.3, 1]}
    )
    bars = _scenario_axes(support, theme, colors)
    support.set_title("worst-case distribution $q^\\star$ (grey: nominal)")
    note = support.text(
        0.02, 0.97, "", transform=support.transAxes, ha="left", va="top", fontsize=9
    )

    radii_np = radii.numpy()
    cursor = stack.axvline(0.0, color=theme.ink, linewidth=1.2)
    stack.set_xlim(0, movable)
    stack.set_ylim(0, 1)
    stack.set_xlabel("total-variation radius $\\rho$")
    stack.set_ylabel("mass of each scenario, stacked")
    stack.set_title("the cheapest scenarios drain one at a time")
    layers: list[object] = []

    value.plot(radii_np, values.numpy(), color=color)
    value.axhline(float(NOMINAL @ LOSS), color=theme.muted, linewidth=1, linestyle="--")
    value.axhline(float(LOSS.max()), color=theme.muted, linewidth=1, linestyle=":")
    (marker,) = value.plot([], [], "o", color=color, markersize=8)
    value.set_xlim(0, movable)
    value.set_ylim(0, float(LOSS.max()) * 1.12)
    value.set_xlabel("total-variation radius $\\rho$")
    value.set_ylabel("worst-case expected loss")
    value.set_title("value: linear pieces, one per drained scenario")
    value.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=shade) for shade in colors],
        labels=[f"scenario with loss {value:.1f}" for value in LOSS.tolist()],
        loc="lower right",
        fontsize=7.5,
    )
    value.text(
        movable,
        float(NOMINAL @ LOSS) + 0.15,
        "nominal mean  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    value.text(
        movable,
        float(LOSS.max()) - 0.15,
        "maximum loss  ",
        ha="right",
        va="top",
        fontsize=8,
        color=theme.secondary,
    )
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        radius = float(radii[frame])
        set_bar_heights(bars, maximizers[frame])
        for layer in layers:
            layer.remove()  # type: ignore[attr-defined]
        layers.clear()
        for index, shade in enumerate(colors):
            layers.append(
                stack.fill_between(
                    radii_np[: frame + 1],
                    cumulative[: frame + 1, index],
                    cumulative[: frame + 1, index + 1],
                    color=shade,
                    linewidth=0,
                )
            )
        cursor.set_xdata([radius, radius])
        marker.set_data([radius], [float(values[frame])])
        note.set_text(
            f"$\\rho$ = {radius:.3f}\n"
            f"mass moved = {float(moved[frame]):.3f}\n"
            f"value = {float(values[frame]):.3f}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "tv-shift", fps=10)


FIGURES = (
    FigureSpec(
        name="kl-radius",
        page="examples/kl_dro_radius_sweep.md",
        outputs=("kl-radius-{theme}.gif",),
        render=render_kl_radius,
    ),
    FigureSpec(
        name="tv-shift",
        page="examples/worst_case_distribution_shift.md",
        outputs=("tv-shift-{theme}.gif",),
        render=render_tv_shift,
    ),
)
