"""The phi-divergence generators behind the named sets, and their conjugates."""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt

from optora.dro import chi_square_conjugate, cvar_conjugate, kl_conjugate
from tools.figures._base import Context, FigureSpec, set_color

ALPHA = 0.5


def render_generators(context: Context) -> None:
    """Draw phi(t) for KL, chi-square, total variation and CVaR, with phi*."""
    theme = context.theme
    t = torch.linspace(1e-3, 3.0, 400, dtype=torch.float64)
    s = torch.linspace(-1.5, 1.5, 400, dtype=torch.float64)
    generators = {
        "kl": t * torch.log(t) - t + 1.0,
        "chi_square": (t - 1.0) ** 2,
        "total_variation": torch.abs(t - 1.0) / 2.0,
        "cvar": torch.where(t <= 1.0 / ALPHA, torch.zeros_like(t), torch.nan),
    }
    conjugates = {
        "kl": kl_conjugate(s),
        "chi_square": chi_square_conjugate(s),
        "total_variation": torch.where(s <= 0.5, torch.clamp(s, min=-0.5), torch.nan),
        "cvar": cvar_conjugate(s, ALPHA),
    }
    labels = {
        "kl": "KL  $t\\log t - t + 1$",
        "chi_square": "chi-square  $(t-1)^2$",
        "total_variation": "total variation  $|t-1|/2$",
        "cvar": f"CVaR  $0$ on $[0, 1/\\alpha]$, $\\alpha={ALPHA}$",
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for key, values in generators.items():
        left.plot(
            t.numpy(), values.numpy(), color=set_color(theme, key), label=labels[key]
        )
    left.axvline(1.0 / ALPHA, color=set_color(theme, "cvar"), linewidth=2)
    left.text(1.0 / ALPHA + 0.05, 1.2, "$+\\infty$", color=theme.secondary, fontsize=9)
    left.set_xlim(0, 3)
    left.set_ylim(-0.05, 2.0)
    left.set_xlabel("likelihood ratio  $t = q_i / \\mathrm{nominal}_i$")
    left.set_ylabel("$\\varphi(t)$")
    left.set_title("generator $\\varphi$: the price of reweighting a scenario")
    left.legend(loc="upper left")

    for key, values in conjugates.items():
        right.plot(s.numpy(), values.numpy(), color=set_color(theme, key))
    right.set_xlim(-1.5, 1.5)
    right.set_ylim(-0.6, 3.2)
    right.set_xlabel("scaled shift  $s = (\\ell - \\lambda)/\\eta$")
    right.set_ylabel("$\\varphi^*(s)$")
    right.set_title("conjugate $\\varphi^*$: what enters the dual")
    right.text(
        -1.45,
        2.95,
        "total variation: $+\\infty$ past $s = 1/2$",
        fontsize=8,
        color=theme.secondary,
    )
    fig.tight_layout(w_pad=2.0)
    context.save_static(fig, "phi-generators")


FIGURES = (
    FigureSpec(
        name="phi-generators",
        page="formulations.md",
        outputs=("phi-generators-{theme}.svg",),
        render=render_generators,
    ),
)
