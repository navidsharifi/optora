"""The MMD ball: its kernel geometry and why the closed form is only a bound."""

from __future__ import annotations

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from optora.dro import MMDAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color
from tools.figures.problems import demo_instance
from tools.figures.sets import BANDWIDTH, FAMILY_BY_KEY


def render_mmd_ball(context: Context) -> None:
    """Draw the Gram matrix and the exact worst case against its closed-form bound."""
    theme = context.theme
    instance = demo_instance()
    nominal, loss = instance.nominal, instance.loss
    gram = instance.gram(BANDWIDTH)
    saturation = FAMILY_BY_KEY["mmd"].saturation(instance)
    radii = torch.linspace(
        0.0, 1.15 * saturation, context.size(80, 12), dtype=loss.dtype
    )
    exact = MMDAmbiguitySet(nominal, gram, radii).worst_case_expectation(
        loss.expand(radii.shape[0], -1)
    )
    # Dropping q >= 0 gives the closed form E[loss] + radius * |loss|_{K^-1}.
    dual_norm = torch.sqrt(loss @ torch.linalg.solve(gram, loss))
    bound = float(nominal @ loss) + radii * dual_norm
    color = set_color(theme, "mmd")
    cmap = LinearSegmentedColormap.from_list("sequential", list(theme.sequential))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    labels = [f"{value:.2f}" for value in instance.points.numpy()]
    sns.heatmap(
        gram.numpy(),
        ax=left,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"fraction": 0.046, "pad": 0.03},
    )
    left.collections[0].colorbar.outline.set_visible(False)
    left.collections[0].colorbar.ax.tick_params(colors=theme.muted, labelsize=8)
    left.tick_params(labelsize=7, rotation=0)
    left.set_xlabel("support point $x_j$")
    left.set_ylabel("support point $x_i$")
    left.set_title(f"Laplacian Gram matrix $e^{{-|x_i - x_j| / {BANDWIDTH}}}$")

    right.plot(radii.numpy(), bound.numpy(), color=theme.muted, linewidth=1.6)
    right.plot(radii.numpy(), exact.detach().numpy(), color=color)
    right.axhline(float(loss.max()), color=theme.muted, linewidth=1)
    right.axvline(saturation, color=theme.axis, linewidth=1)
    right.set_xlim(0, float(radii[-1]))
    right.set_ylim(float(nominal @ loss) - 0.05, float(bound[-1]) + 0.05)
    right.set_xlabel("MMD radius")
    right.set_ylabel("worst-case expected loss")
    right.set_title("exact cone program vs the closed-form bound")
    right.legend(
        handles=[
            plt.Line2D([], [], color=color),
            plt.Line2D([], [], color=theme.muted, linewidth=1.6),
        ],
        labels=[
            "exact: interior-point solve with $q \\geq 0$",
            "bound: $\\mathbb{E}[\\ell] + \\rho\\, \\|\\ell\\|_{K^{-1}}$",
        ],
        loc="upper left",
    )
    right.text(
        saturation,
        float(nominal @ loss),
        "  saturation",
        fontsize=8,
        color=theme.secondary,
        va="bottom",
    )
    right.text(
        float(radii[-1]),
        float(loss.max()) + 0.01,
        "$\\max_i \\ell_i$  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    fig.tight_layout(w_pad=2.5)
    context.save_static(fig, "mmd-ball")


FIGURES = (
    FigureSpec(
        name="mmd-ball",
        page="formulations.md",
        outputs=("mmd-ball-{theme}.svg",),
        render=render_mmd_ball,
    ),
)
