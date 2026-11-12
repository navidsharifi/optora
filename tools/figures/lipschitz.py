r"""Wasserstein DRO as Lipschitz regularization, animated over the radius.

The animation belongs to tutorial 05. It shrinks the type-1 Wasserstein
radius and watches the discrete dual

$$\inf_{\gamma \ge 0}\; \gamma\rho + \sum_i p_i \max_j (\ell_j - \gamma c_{ij})$$

settle its minimizer on the sample Lipschitz modulus $L_n$: normalized by
$\rho$, the dual is the line $\gamma$ to the right of $L_n$ and a wall to
its left, so the kink is the minimizer for every small enough radius and
the worst case equals $\mathbb{E}_p[\ell] + \rho L_n$. The other panels
follow the price of radius $\mathrm{d}V/\mathrm{d}\rho$, which is that
minimizer by the envelope theorem, and the normalized gap to the surrogate
built from the true Lipschitz constant, for three sample sizes.
"""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt

from optora.dro import WassersteinAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color

HOLD_FRAMES = 10
LIPSCHITZ_CONSTANT = 1.0
SAMPLE_SIZES = (16, 64, 256)
SHOWN_SIZE = 64


def empirical_measure(size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tutorial 05's quantile discretization of the standard normal.

    Returns:
        The uniform nominal, the softplus loss on the support, and the
        absolute ground cost, as in the tutorial script.
    """
    quantiles = (torch.arange(size, dtype=torch.float64) + 0.5) / size
    support = torch.special.ndtri(quantiles)
    loss = torch.nn.functional.softplus(support)
    cost = (support[:, None] - support[None, :]).abs()
    nominal = torch.full_like(support, 1.0 / size)
    return nominal, loss, cost


def sample_lipschitz(loss: torch.Tensor, cost: torch.Tensor) -> float:
    """Return the Lipschitz modulus of `loss` restricted to the sample."""
    increments = (loss[:, None] - loss[None, :]).abs()
    separation = cost + torch.eye(cost.shape[-1], dtype=cost.dtype)
    return float((increments / separation).max())


def normalized_dual(
    nominal: torch.Tensor,
    loss: torch.Tensor,
    cost: torch.Tensor,
    radii: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    r"""Return $(D(\gamma) - \mathbb{E}_p[\ell]) / \rho$ on a grid of $\gamma$.

    Args:
        nominal: Nominal distribution of shape `(n,)`.
        loss: Losses of shape `(n,)`.
        cost: Ground cost of shape `(n, n)`.
        radii: Radii of shape `(R,)`.
        gammas: Dual grid of shape `(G,)`.

    Returns:
        The normalized dual objective of shape `(R, G)`.
    """
    transported = (loss[None, None, :] - gammas[:, None, None] * cost[None]).amax(-1)
    excess = transported @ nominal - nominal @ loss
    return gammas[None, :] + excess[None, :] / radii[:, None]


def value_and_price(
    nominal: torch.Tensor, loss: torch.Tensor, cost: torch.Tensor, radii: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the exact worst case and its derivative $\mathrm{d}V/\mathrm{d}\rho$.

    The derivative is the dual minimizer $\gamma^\star$ (envelope theorem),
    read from autograd rather than from the dual grid.
    """
    radii = radii.detach().requires_grad_()
    ambiguity_set = WassersteinAmbiguitySet(nominal, cost, radius=radii)
    value = ambiguity_set.worst_case_expectation(loss.expand(radii.shape[0], -1))
    (price,) = torch.autograd.grad(value.sum(), radii)
    return value.detach(), price


def render_lipschitz_dual(context: Context) -> None:
    """Animate the dual kink, the price of radius and the gap as the radius shrinks."""
    theme = context.theme
    frames = context.frames(48)
    radii = torch.logspace(0.5, -4.0, frames, dtype=torch.float64)
    gammas = torch.linspace(0.0, 1.4, context.size(561, 71), dtype=torch.float64)
    nominal, loss, cost = empirical_measure(SHOWN_SIZE)
    shown_modulus = sample_lipschitz(loss, cost)
    dual = normalized_dual(nominal, loss, cost, radii, gammas)
    dual_argmin = dual.argmin(-1)
    prices, gaps, moduli = {}, {}, {}
    for size in SAMPLE_SIZES:
        measure = empirical_measure(size)
        value, prices[size] = value_and_price(*measure, radii)
        surrogate = measure[0] @ measure[1] + radii * LIPSCHITZ_CONSTANT
        gaps[size] = (surrogate - value) / radii
        moduli[size] = sample_lipschitz(*measure[1:])
    color = set_color(theme, "wasserstein")
    size_colors = [theme.sequential[index] for index in (3, 4, 6)]

    fig, (landscape, price, gap) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    gammas_np = gammas.numpy()
    landscape.plot(
        gammas_np,
        gammas_np,
        color=theme.muted,
        linewidth=1,
        linestyle="--",
        label="$\\gamma$",
    )
    (dual_line,) = landscape.plot(
        [], [], color=color, label="$(D(\\gamma) - \\mathbb{E}_p[\\ell]) / \\rho$"
    )
    (dual_marker,) = landscape.plot([], [], "o", color=color, markersize=8)
    landscape.axvline(
        shown_modulus,
        color=theme.ink,
        linewidth=1.2,
        linestyle=":",
        label=f"$L_n$ = {shown_modulus:.3f}",
    )
    landscape.axvline(
        LIPSCHITZ_CONSTANT,
        color=theme.muted,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="$\\mathrm{Lip}(\\ell)$ = 1",
    )
    landscape.set_xlim(float(gammas[0]), float(gammas[-1]))
    landscape.set_ylim(0, 1.6)
    landscape.set_xlabel("dual variable $\\gamma$")
    landscape.set_ylabel("normalized dual objective")
    landscape.set_title(f"the dual's kink sits at $L_n$  ($n$ = {SHOWN_SIZE})")
    landscape.legend(loc="upper left", fontsize=7.5)
    note = landscape.text(
        0.98,
        0.03,
        "",
        transform=landscape.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    radii_np = radii.numpy()
    price_markers, gap_markers = {}, {}
    for size, shade in zip(SAMPLE_SIZES, size_colors, strict=True):
        price.plot(radii_np, prices[size].numpy(), color=shade, label=f"$n$ = {size}")
        price.axhline(moduli[size], color=shade, linewidth=1, linestyle=":")
        (price_markers[size],) = price.plot([], [], "o", color=shade, markersize=7)
        gap.plot(radii_np, gaps[size].numpy(), color=shade, label=f"$n$ = {size}")
        gap.axhline(
            LIPSCHITZ_CONSTANT - moduli[size], color=shade, linewidth=1, linestyle=":"
        )
        (gap_markers[size],) = gap.plot([], [], "o", color=shade, markersize=7)
    price.axhline(
        LIPSCHITZ_CONSTANT, color=theme.muted, linewidth=1.2, linestyle=(0, (4, 3))
    )
    price.text(
        radii_np[-1],
        LIPSCHITZ_CONSTANT + 0.01,
        "$\\mathrm{Lip}(\\ell)$  ",
        ha="right",
        va="bottom",
        fontsize=8,
        color=theme.secondary,
    )
    for axes, label, title in (
        (
            price,
            "$\\mathrm{d}V / \\mathrm{d}\\rho$",
            "the price of radius settles at $L_n$ (dotted)",
        ),
        (
            gap,
            "(surrogate $-$ exact) / $\\rho$",
            "the gap flattens at $\\mathrm{Lip} - L_n$ (dotted)",
        ),
    ):
        axes.set_xscale("log")
        axes.invert_xaxis()
        axes.set_xlim(radii_np[0], radii_np[-1])
        axes.set_xlabel("Wasserstein radius $\\rho$  (log, decreasing)")
        axes.set_ylabel(label)
        axes.set_title(title)
    price.set_ylim(0, 1.08)
    price.legend(loc="lower left", fontsize=7.5)
    top = max(float(values.max()) for values in gaps.values())
    gap.set_ylim(-0.02, top * 1.1)
    gap.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        radius = float(radii[frame])
        dual_line.set_data(gammas_np, dual[frame].numpy())
        argmin = int(dual_argmin[frame])
        dual_marker.set_data([float(gammas[argmin])], [float(dual[frame, argmin])])
        for size in SAMPLE_SIZES:
            price_markers[size].set_data([radius], [float(prices[size][frame])])
            gap_markers[size].set_data([radius], [float(gaps[size][frame])])
        note.set_text(
            f"$\\rho$ = {radius:.2e}\n"
            f"$\\gamma^\\star$ = {float(prices[SHOWN_SIZE][frame]):.3f}\n"
            f"gap / $\\rho$ = {float(gaps[SHOWN_SIZE][frame]):.4f}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "lipschitz-dual", fps=10)


FIGURES = (
    FigureSpec(
        name="lipschitz-dual",
        page="tutorials/wasserstein_lipschitz_equivalence.md",
        outputs=("lipschitz-dual-{theme}.gif",),
        render=render_lipschitz_dual,
    ),
)
