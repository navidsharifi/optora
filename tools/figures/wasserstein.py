"""The Wasserstein worst case and its piecewise-linear dual, over the radius."""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt

from optora.dro import WassersteinAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color
from tools.figures.problems import demo_instance
from tools.figures.sets import FAMILY_BY_KEY

HOLD_FRAMES = 8


def render_wasserstein_dual(context: Context) -> None:
    """Animate mass transport along the line and the dual minimizer gamma*."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss, cost, points = (
        instance.nominal,
        instance.loss,
        instance.cost,
        instance.points,
    )
    frames = context.frames(56)
    saturation = FAMILY_BY_KEY["wasserstein"].saturation(instance)
    radii = torch.linspace(0.0, saturation, frames, dtype=loss.dtype)
    radii = radii.clone().requires_grad_()
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost, radii)
    batched = loss.expand(frames, -1).clone().requires_grad_()
    values = ambiguity_set.worst_case_expectation(batched)
    maximizers, multipliers = torch.autograd.grad(values.sum(), (batched, radii))
    values = values.detach()
    color = set_color(theme, "wasserstein")

    gammas = torch.linspace(0.0, 1.15 * float(multipliers.max()), 300, dtype=loss.dtype)
    best_response = (
        (loss[None, None, :] - gammas[:, None, None] * cost[None, :, :])
        .max(dim=-1)
        .values
    )
    expectations = best_response @ nominal
    duals = gammas[:, None] * radii.detach()[None, :] + expectations[:, None]

    fig = plt.figure(figsize=(9.6, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 2.2])
    loss_axes = fig.add_subplot(grid[0, 0])
    mass_axes = fig.add_subplot(grid[1, 0], sharex=loss_axes)
    dual_axes = fig.add_subplot(grid[:, 1])

    loss_axes.plot(points.numpy(), loss.numpy(), color=theme.ink, linewidth=1.5)
    loss_axes.plot(points.numpy(), loss.numpy(), "o", color=theme.ink, markersize=5)
    loss_axes.set_ylabel("loss")
    loss_axes.tick_params(labelbottom=False)
    loss_axes.set_title("scenarios on a line with ground cost $|x_i - x_j|$")

    width = float(points[1] - points[0]) * 0.8
    mass_axes.bar(
        points.numpy(), nominal.numpy(), width=width, color=theme.muted, alpha=0.35
    )
    bars = mass_axes.bar(
        points.numpy(), maximizers[0].numpy(), width=width * 0.65, color=color
    )
    mass_axes.set_ylim(0, 1.0)
    mass_axes.set_xlabel("support point $x$")
    mass_axes.set_ylabel("probability mass")
    mass_axes.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=theme.muted, alpha=0.35),
            plt.Rectangle((0, 0), 1, 1, color=color),
        ],
        labels=["nominal", "worst case"],
        loc="upper left",
    )
    note = mass_axes.text(
        0.02, 0.72, "", transform=mass_axes.transAxes, ha="left", va="top", fontsize=9
    )

    (dual_line,) = dual_axes.plot(gammas.numpy(), duals[:, 0].numpy(), color=color)
    (minimizer,) = dual_axes.plot([], [], "o", color=color, markersize=8)
    dual_axes.set_xlim(0, float(gammas[-1]))
    dual_axes.set_ylim(float(nominal @ loss) - 0.05, float(loss.max()) + 0.35)
    dual_axes.set_xlabel("dual variable $\\gamma$")
    dual_axes.set_ylabel(
        "$\\gamma\\,\\rho + \\mathbb{E}_{\\mathrm{nominal}}"
        "[\\max_j(\\ell_j - \\gamma\\, c_{\\cdot j})]$"
    )
    dual_axes.set_title("the dual is piecewise linear: bisect its slope")
    dual_axes.axhline(float(loss.max()), color=theme.muted, linewidth=1)
    dual_axes.text(
        float(gammas[-1]),
        float(loss.max()) + 0.02,
        "$\\max_i \\ell_i$  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        for bar, height in zip(bars, maximizers[frame].numpy(), strict=True):
            bar.set_height(height)
        dual_line.set_ydata(duals[:, frame].numpy())
        gamma = float(multipliers[frame])
        minimizer.set_data([gamma], [values[frame].item()])
        note.set_text(
            f"radius $\\rho$ = {radii[frame].item():.3f}\n"
            f"value = {values[frame].item():.3f}\n"
            f"$\\gamma^\\star$ = {gamma:.2f}"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "wasserstein-dual", fps=10
    )


FIGURES = (
    FigureSpec(
        name="wasserstein-dual",
        page="formulations.md",
        outputs=("wasserstein-dual-{theme}.gif",),
        render=render_wasserstein_dual,
    ),
)
