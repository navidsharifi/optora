"""The Sinkhorn ball's kernel smoothing, animated as the entropic strength vanishes."""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from optora.dro import SinkhornAmbiguitySet, WassersteinAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color, worst_case_distributions
from tools.figures.problems import demo_instance

RADIUS = 0.08
HOLD_FRAMES = 8


def gibbs_kernel(cost: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return the row-stochastic Gibbs kernel of `cost` at strength `epsilon`."""
    logits = -cost / epsilon
    return torch.softmax(logits, dim=-1)


def render_sinkhorn_kernel(context: Context) -> None:
    """Animate the kernel, the smoothed nominal and the worst case over epsilon."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss, cost, points = (
        instance.nominal,
        instance.loss,
        instance.cost,
        instance.points,
    )
    frames = context.frames(44)
    epsilons = torch.logspace(0, -2, frames, dtype=loss.dtype)
    kernels = [gibbs_kernel(cost, float(epsilon)) for epsilon in epsilons]
    smoothed = [nominal @ kernel for kernel in kernels]
    worst_cases = []
    values = []
    for epsilon in epsilons:
        value, maximizer = worst_case_distributions(
            lambda radius, epsilon=epsilon: SinkhornAmbiguitySet(
                nominal, cost, float(epsilon), radius
            ),
            loss,
            torch.tensor([RADIUS], dtype=loss.dtype),
        )
        values.append(float(value[0]))
        worst_cases.append(maximizer[0])
    limit_value, limit = worst_case_distributions(
        lambda radius: WassersteinAmbiguitySet(nominal, cost, radius),
        loss,
        torch.tensor([RADIUS], dtype=loss.dtype),
    )
    color = set_color(theme, "sinkhorn")
    limit_color = set_color(theme, "wasserstein")
    cmap = LinearSegmentedColormap.from_list("sequential", list(theme.sequential))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.6, 4.3), gridspec_kw={"width_ratios": [1, 1.35]}
    )
    image = left.imshow(kernels[0].numpy(), cmap=cmap, vmin=0, vmax=1, origin="lower")
    left.grid(False)
    left.set_xlabel("destination scenario $j$")
    left.set_ylabel("source scenario $i$")
    left.set_title("Gibbs kernel rows $Q_{ij} \\propto e^{-c_{ij}/\\epsilon}$")
    colorbar = fig.colorbar(image, ax=left, fraction=0.046, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=theme.muted, labelsize=8)

    width = float(points[1] - points[0]) * 0.8
    right.bar(
        points.numpy(), nominal.numpy(), width=width, color=theme.muted, alpha=0.35
    )
    (smoothed_line,) = right.step(
        points.numpy(), smoothed[0].numpy(), where="mid", color=theme.ink, linewidth=1.4
    )
    bars = right.bar(
        points.numpy(), worst_cases[0].numpy(), width=width * 0.62, color=color
    )
    right.bar(
        points.numpy(),
        limit[0].numpy(),
        width=width * 0.62,
        facecolor="none",
        edgecolor=limit_color,
        linewidth=1.5,
    )
    right.set_ylim(0, max(0.5, float(limit.max()) * 1.1))
    right.set_xlabel("support point $x$")
    right.set_ylabel("probability mass")
    right.set_title(f"worst case at radius {RADIUS}")
    right.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=theme.muted, alpha=0.35),
            plt.Line2D([], [], color=theme.ink, linewidth=1.4),
            plt.Rectangle((0, 0), 1, 1, color=color),
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="none", edgecolor=limit_color, linewidth=1.5
            ),
        ],
        labels=[
            "nominal",
            "smoothed nominal $P^0 = \\mathrm{nominal}^\\top Q$",
            "Sinkhorn worst case",
            "Wasserstein worst case ($\\epsilon \\to 0$ limit)",
        ],
        loc="upper left",
        fontsize=8,
    )
    note = right.text(
        0.98, 0.95, "", transform=right.transAxes, ha="right", va="top", fontsize=9
    )
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        image.set_data(kernels[frame].numpy())
        smoothed_line.set_ydata(smoothed[frame].numpy())
        for bar, height in zip(bars, worst_cases[frame].numpy(), strict=True):
            bar.set_height(height)
        note.set_text(
            f"$\\epsilon$ = {float(epsilons[frame]):.3g}\n"
            f"value = {values[frame]:.3f}\n"
            f"Wasserstein limit = {float(limit_value[0]):.3f}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "sinkhorn-kernel", fps=8)


FIGURES = (
    FigureSpec(
        name="sinkhorn-kernel",
        page="formulations.md",
        outputs=("sinkhorn-kernel-{theme}.gif",),
        render=render_sinkhorn_kernel,
    ),
)
