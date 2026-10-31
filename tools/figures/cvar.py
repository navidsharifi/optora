"""Conditional value-at-risk as the mean of the upper tail, animated over the level."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt

from optora.dro import CVaRAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color, worst_case_distributions
from tools.figures.problems import demo_instance


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


FIGURES = (
    FigureSpec(
        name="cvar-tail",
        page="formulations.md",
        outputs=("cvar-tail-{theme}.gif",),
        render=render_cvar_tail,
    ),
)
