"""The KL worst case as an exponential tilt of the nominal, animated over the radius."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt

from optora.dro import KLAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color, worst_case_distributions
from tools.figures.problems import demo_instance


def render_kl_tilt(context: Context) -> None:
    """Animate the KL maximizer and its value as the radius grows to saturation."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss = instance.nominal, instance.loss
    top_mass = nominal[loss == loss.max()].sum()
    saturation = float(-torch.log(top_mass))
    frames = context.frames(72)
    radii = torch.logspace(-3, np.log10(1.25 * saturation), frames, dtype=loss.dtype)
    values, maximizers = worst_case_distributions(
        lambda radius: KLAmbiguitySet(nominal, radius), loss, radii
    )
    nominal_value = float(nominal @ loss)
    color = set_color(theme, "kl")
    index = np.arange(instance.size)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    left.bar(index, nominal.numpy(), width=0.8, color=theme.muted, alpha=0.35)
    bars = left.bar(index, maximizers[0].numpy(), width=0.55, color=color)
    left.set_ylim(0, 0.42)
    left.set_xticks(index)
    left.set_xticklabels([f"{value:.2f}" for value in loss.numpy()], rotation=60)
    left.set_xlabel("scenario loss  $\\ell_i$")
    left.set_ylabel("probability mass")
    left.set_title("worst case $q_\\beta \\propto$ nominal $\\cdot e^{\\beta \\ell}$")
    left.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=theme.muted, alpha=0.35),
            plt.Rectangle((0, 0), 1, 1, color=color),
        ],
        labels=["nominal", "worst case"],
        loc="upper left",
    )
    note = left.text(
        0.98, 0.95, "", transform=left.transAxes, ha="right", va="top", fontsize=9
    )

    right.plot(radii.numpy(), values.numpy(), color=color)
    right.axhline(nominal_value, color=theme.muted, linewidth=1, linestyle="-")
    right.axhline(float(loss.max()), color=theme.muted, linewidth=1, linestyle="-")
    right.axvline(saturation, color=theme.axis, linewidth=1)
    right.text(
        saturation,
        nominal_value + 0.02,
        "saturation $-\\log P^\\star$ ",
        fontsize=8,
        color=theme.secondary,
        va="bottom",
        ha="right",
    )
    right.text(
        radii[0].item(),
        nominal_value + 0.02,
        "$\\mathbb{E}_{\\mathrm{nominal}}[\\ell]$",
        fontsize=8,
        color=theme.secondary,
        va="bottom",
    )
    right.text(
        radii[0].item(),
        float(loss.max()) - 0.02,
        "$\\max_i \\ell_i$",
        fontsize=8,
        color=theme.secondary,
        va="top",
    )
    (marker,) = right.plot([], [], "o", color=color, markersize=8)
    right.set_xscale("log")
    right.set_xlabel("KL radius")
    right.set_ylabel("worst-case expected loss")
    right.set_title("value vs radius")
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        for bar, height in zip(bars, maximizers[frame].numpy(), strict=True):
            bar.set_height(height)
        marker.set_data([radii[frame].item()], [values[frame].item()])
        note.set_text(
            f"radius = {radii[frame].item():.3g}\nvalue = {values[frame].item():.3f}"
        )

    context.save_animation(fig, update, frames, "kl-tilt", fps=10)


FIGURES = (
    FigureSpec(
        name="kl-tilt",
        page="formulations.md",
        outputs=("kl-tilt-{theme}.gif",),
        render=render_kl_tilt,
    ),
)
