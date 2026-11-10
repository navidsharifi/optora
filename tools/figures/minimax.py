r"""The robust decision problem of examples 03 and 04, animated.

Both animations draw the scalar location problem
$\min_x \sup_{q} \mathbb{E}_q[(\xi - x)^2]$ on the outcomes
$\{1, 2, 3, 10\}$ under a uniform nominal. The first grows the radius and
compares the four geometries: the robust objective in $x$, its minimizer,
and the adversary each ball fields at its own decision. The second fixes
the KL ball and takes a gradient method apart: the iterate rolling down the
robust objective, and the one-dimensional dual landscape the inner solve
minimizes at every step.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt

from optora.core import AmbiguitySet
from optora.dro import KLAmbiguitySet
from tools.figures._base import (
    SET_LABELS,
    Context,
    FigureSpec,
    set_bar_heights,
    set_color,
)
from tools.figures.problems import Instance
from tools.figures.sets import FAMILY_BY_KEY

HOLD_FRAMES = 10
BISECTION_STEPS = 40
GEOMETRIES = ("kl", "chi_square", "total_variation", "wasserstein")
OUTCOMES = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
NOMINAL = torch.full_like(OUTCOMES, 1.0 / OUTCOMES.numel())
# Example 03's instance: squared-distance ground cost, uniform nominal.
INSTANCE = Instance(
    points=OUTCOMES,
    nominal=NOMINAL,
    loss=(OUTCOMES - OUTCOMES.mean()) ** 2,
    cost=(OUTCOMES[:, None] - OUTCOMES[None, :]) ** 2,
)
KL_RADIUS = 0.15
STEP_SIZE = 0.01


def squared_losses(x: torch.Tensor) -> torch.Tensor:
    """Return the squared-error loss of decisions `x`, shape `(..., n)`."""
    return (x[..., None] - OUTCOMES) ** 2


def value_and_slope(
    ambiguity_set: AmbiguitySet, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the robust objective and its derivative at decisions `x`."""
    x = x.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(squared_losses(x))
    (slope,) = torch.autograd.grad(value.sum(), x)
    return value.detach(), slope


def adversary(ambiguity_set: AmbiguitySet, x: torch.Tensor) -> torch.Tensor:
    """Return the worst-case distribution at decisions `x` (Danskin), `(..., n)`."""
    loss = squared_losses(x).detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(loss)
    (maximizer,) = torch.autograd.grad(value.sum(), loss)
    return maximizer


def minimizer(ambiguity_set: AmbiguitySet, batch: int) -> torch.Tensor:
    """Bisect the derivative of the robust objective for a set with `batch` radii.

    Returns:
        The minimizers of shape `(batch, 1)`, one per radius of the set.
    """
    lower = OUTCOMES[:1].expand(batch, 1).clone()
    upper = OUTCOMES[-1:].expand(batch, 1).clone()
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        positive = value_and_slope(ambiguity_set, middle)[1] > 0.0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    return 0.5 * (lower + upper)


def render_decision_geometries(context: Context) -> None:
    """Animate the four robust objectives, decisions and adversaries over the radius."""
    theme = context.theme
    frames = context.frames(48)
    radii = torch.linspace(0.0, 0.4, frames, dtype=torch.float64)
    grid = torch.linspace(2.0, 9.0, context.size(241, 61), dtype=torch.float64)
    erm_decision = float(NOMINAL @ OUTCOMES)
    nominal_loss = squared_losses(grid) @ NOMINAL

    landscapes, decisions, adversaries = {}, {}, {}
    for key in GEOMETRIES:
        ambiguity_set = FAMILY_BY_KEY[key].build(INSTANCE, radii[:, None])
        landscapes[key] = value_and_slope(ambiguity_set, grid.expand(frames, -1))[0]
        point = minimizer(ambiguity_set, frames)
        decisions[key] = point[:, 0]
        adversaries[key] = adversary(ambiguity_set, point)[:, 0]
    minima = {
        key: value_and_slope(
            FAMILY_BY_KEY[key].build(INSTANCE, radii[:, None]), decisions[key][:, None]
        )[0][:, 0]
        for key in GEOMETRIES
    }

    fig, (landscape, paths, support) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.3, 1, 1.1]}
    )
    landscape.plot(
        grid.numpy(),
        nominal_loss.numpy(),
        color=theme.muted,
        linestyle="--",
        linewidth=1.4,
        label="nominal (ERM)",
    )
    landscape.plot(
        [erm_decision],
        [float(nominal_loss.min())],
        "o",
        color=theme.muted,
        markersize=7,
    )
    lines, markers, dots, bar_sets = {}, {}, {}, {}
    width = 0.8 / len(GEOMETRIES)
    support.bar(
        np.arange(OUTCOMES.numel()),
        NOMINAL.numpy(),
        width=0.85,
        color=theme.muted,
        alpha=0.3,
    )
    for index, key in enumerate(GEOMETRIES):
        color = set_color(theme, key)
        (lines[key],) = landscape.plot([], [], color=color, label=SET_LABELS[key])
        (markers[key],) = landscape.plot([], [], "o", color=color, markersize=8)
        paths.plot(radii.numpy(), decisions[key].numpy(), color=color)
        (dots[key],) = paths.plot([], [], "o", color=color, markersize=8)
        bar_sets[key] = support.bar(
            np.arange(OUTCOMES.numel()) + (index - 1.5) * width,
            NOMINAL.numpy(),
            width=width,
            color=color,
        )
    top = max(float(values.max()) for values in landscapes.values())
    landscape.set_xlim(float(grid[0]), float(grid[-1]))
    landscape.set_ylim(0, top * 1.05)
    landscape.set_xlabel("decision $x$")
    landscape.set_ylabel("worst-case expected squared error")
    landscape.set_title("robust objective in $x$, one curve per geometry")
    landscape.legend(loc="upper left", fontsize=7.5)
    note = landscape.text(
        0.5,
        0.03,
        "",
        transform=landscape.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    paths.axhline(erm_decision, color=theme.muted, linewidth=1.2, linestyle="--")
    paths.text(
        0.0, erm_decision + 0.08, "ERM", fontsize=8, color=theme.secondary, va="bottom"
    )
    paths.set_xlim(0.0, float(radii[-1]))
    paths.set_xlabel("ambiguity radius $\\rho$")
    paths.set_ylabel("robust decision $x^\\star$")
    paths.set_title("where each decision goes as $\\rho$ grows")

    support.set_xticks(
        np.arange(OUTCOMES.numel()), [f"{value:g}" for value in OUTCOMES.tolist()]
    )
    support.set_ylim(0, 1.0)
    support.set_xlabel("outcome $\\xi$")
    support.set_ylabel("probability mass")
    support.set_title("each adversary at its own $x^\\star$ (grey: nominal)")
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        radius = float(radii[frame])
        for key in GEOMETRIES:
            lines[key].set_data(grid.numpy(), landscapes[key][frame].numpy())
            markers[key].set_data(
                [float(decisions[key][frame])], [float(minima[key][frame])]
            )
            dots[key].set_data([radius], [float(decisions[key][frame])])
            set_bar_heights(bar_sets[key], adversaries[key][frame])
        note.set_text(
            f"$\\rho$ = {radius:.3f}\n"
            + "\n".join(
                f"{SET_LABELS[key]}: $x^\\star$ = {float(decisions[key][frame]):.3f}"
                for key in GEOMETRIES
            )
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "decision-geometries", fps=10
    )


def kl_dual(loss: torch.Tensor, radius: float, log_eta: torch.Tensor) -> torch.Tensor:
    r"""Evaluate the KL dual $\eta\rho + \eta \log \mathbb{E}_p[e^{\ell/\eta}]$.

    Args:
        loss: Per-scenario losses of shape `(..., n)`.
        radius: KL radius.
        log_eta: Grid in $\log\eta$ of shape `(G,)`.

    Returns:
        The dual objective of shape `(..., G)`.
    """
    eta = torch.exp(log_eta)
    log_mgf = torch.logsumexp(
        torch.log(NOMINAL) + loss[..., None, :] / eta[:, None], dim=-1
    )
    return eta * radius + eta * log_mgf


def render_convergence_diagnostics(context: Context) -> None:
    """Animate the outer iterate on the robust objective and the inner dual."""
    theme = context.theme
    ambiguity_set = KLAmbiguitySet(NOMINAL, KL_RADIUS)
    steps = context.size(150, 12)
    grid = torch.linspace(2.0, 9.0, context.size(241, 61), dtype=torch.float64)
    log_eta = torch.linspace(-6.0, 4.0, context.size(400, 80), dtype=torch.float64)

    # The same fixed-step iteration `GradientDescent` runs, unrolled by hand so
    # the whole trajectory is kept.
    point = NOMINAL @ OUTCOMES
    iterates, values = [point], []
    for _ in range(steps):
        value, slope = value_and_slope(ambiguity_set, point)
        values.append(value)
        point = point - STEP_SIZE * slope
        iterates.append(point)
    values.append(value_and_slope(ambiguity_set, point)[0])
    trajectory = torch.stack(iterates)
    objective = torch.stack(values)
    landscape_values = value_and_slope(ambiguity_set, grid)[0]
    nominal_loss = squared_losses(grid) @ NOMINAL
    duals = kl_dual(squared_losses(trajectory), KL_RADIUS, log_eta)
    grid_minima, grid_argmin = duals.min(dim=-1)

    frames = context.frames(60)
    shown = torch.linspace(0, steps, frames).round().long()
    color = set_color(theme, "kl")

    fig, (landscape, path, dual) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.2, 1, 1.2]}
    )
    landscape.plot(
        grid.numpy(),
        nominal_loss.numpy(),
        color=theme.muted,
        linestyle="--",
        linewidth=1.4,
        label="nominal (ERM)",
    )
    landscape.plot(
        grid.numpy(),
        landscape_values.numpy(),
        color=color,
        label=f"KL worst case, $\\rho$ = {KL_RADIUS}",
    )
    (trail,) = landscape.plot([], [], "o", color=color, markersize=4, alpha=0.4)
    (ball,) = landscape.plot([], [], "o", color=theme.ink, markersize=9)
    landscape.set_xlim(float(grid[0]), float(grid[-1]))
    landscape.set_ylim(0, float(landscape_values.max()) * 1.05)
    landscape.set_xlabel("decision $x$")
    landscape.set_ylabel("worst-case expected squared error")
    landscape.set_title("outer: the iterate rolls down the robust objective")
    landscape.legend(loc="upper left", fontsize=7.5)
    note = landscape.text(
        0.5,
        0.03,
        "",
        transform=landscape.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    (path_line,) = path.plot([], [], color=color)
    (path_dot,) = path.plot([], [], "o", color=theme.ink, markersize=8)
    path.axhline(
        float(trajectory[-1]), color=theme.muted, linewidth=1, linestyle=(0, (4, 3))
    )
    path.set_xlim(0, steps)
    low, high = float(trajectory.min()), float(trajectory.max())
    margin = 0.1 * (high - low)
    path.set_ylim(low - margin, high + margin)
    path.set_xlabel("outer iteration $k$")
    path.set_ylabel("decision $x_k$")
    path.set_title("the trajectory a solver would throw away")

    (dual_line,) = dual.plot([], [], color=color, label="dual objective")
    argmin = dual.axvline(
        0.0,
        color=theme.ink,
        linewidth=1.2,
        linestyle="--",
        label="grid minimizer",
    )
    level = dual.axhline(
        0.0, color=theme.muted, linewidth=1.2, label="library worst case"
    )
    dual.set_xlim(float(log_eta[0]), float(log_eta[-1]))
    dual.set_ylim(float(duals.min()) * 0.9, float(duals.max()) * 1.02)
    dual.set_xlabel("$\\log \\eta$")
    dual.set_ylabel("$\\eta\\rho + \\eta \\log \\mathbb{E}_p[e^{\\ell/\\eta}]$")
    dual.set_title("inner: the dual landscape at $x_k$")
    dual.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        step = int(shown[min(frame, frames - 1)])
        trail.set_data(trajectory[: step + 1].numpy(), objective[: step + 1].numpy())
        ball.set_data([float(trajectory[step])], [float(objective[step])])
        path_line.set_data(np.arange(step + 1), trajectory[: step + 1].numpy())
        path_dot.set_data([step], [float(trajectory[step])])
        dual_line.set_data(log_eta.numpy(), duals[step].numpy())
        argmin.set_xdata([float(log_eta[grid_argmin[step]])] * 2)
        level.set_ydata([float(objective[step])] * 2)
        note.set_text(
            f"$k$ = {step}\n"
            f"$x_k$ = {float(trajectory[step]):.4f}\n"
            f"worst case = {float(objective[step]):.4f}\n"
            f"grid minimum = {float(grid_minima[step]):.4f}"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "convergence-diagnostics", fps=10
    )


FIGURES = (
    FigureSpec(
        name="decision-geometries",
        page="examples/robust_decision_across_ambiguity_sets.md",
        outputs=("decision-geometries-{theme}.gif",),
        render=render_decision_geometries,
    ),
    FigureSpec(
        name="convergence-diagnostics",
        page="examples/convergence_diagnostics.md",
        outputs=("convergence-diagnostics-{theme}.gif",),
        render=render_convergence_diagnostics,
    ),
)
