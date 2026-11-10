"""The Sinkhorn ball, animated: kernel smoothing, the Wasserstein limit, and smoothness.

The first animation belongs to the formulations page and shows the Gibbs
kernel sharpen as the entropic strength vanishes. The other two belong to
the example on the Sinkhorn relaxation of Wasserstein DRO: one sweeps the
entropic strength and watches the robust objective, its derivative and its
minimizer converge to the exact Wasserstein ones; the other sweeps the
decision at a fixed strength and watches the Wasserstein adversary jump
between vertices while the Sinkhorn adversary glides.
"""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from optora.core import AmbiguitySet
from optora.dro import SinkhornAmbiguitySet, WassersteinAmbiguitySet
from tools.figures._base import (
    Context,
    FigureSpec,
    set_bar_heights,
    set_color,
    worst_case_distributions,
)
from tools.figures.problems import demo_instance

RADIUS = 0.08
HOLD_FRAMES = 8
ADVERSARY_EPSILON = 0.02
BISECTION_STEPS = 40


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


def quadratic_losses(theta: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return the squared-error loss of decisions `theta`, shape `(G, n)`."""
    return (theta[:, None] - points) ** 2


def value_and_slope(
    ambiguity_set: AmbiguitySet, theta: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the robust objective and its derivative at the decisions `theta`."""
    theta = theta.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(quadratic_losses(theta, points))
    (slope,) = torch.autograd.grad(value.sum(), theta)
    return value.detach(), slope


def minimizer(ambiguity_set: AmbiguitySet, points: torch.Tensor) -> torch.Tensor:
    """Return the minimizer of the robust objective by bisecting its derivative."""
    lower, upper = points[:1].clone(), points[-1:].clone()
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        positive = value_and_slope(ambiguity_set, middle, points)[1] > 0.0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    return 0.5 * (lower + upper)


def adversaries(
    ambiguity_set: AmbiguitySet, theta: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Return the worst-case distribution at each decision, shape `(G, n)` (Danskin)."""
    losses = quadratic_losses(theta, points).detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(losses)
    (maximizers,) = torch.autograd.grad(value.sum(), losses)
    return maximizers


def render_sinkhorn_relaxation(context: Context) -> None:
    """Animate the robust objective, its derivative and its minimizer over epsilon."""
    theme = context.theme
    instance = demo_instance(size=context.size(24, 12))
    points, nominal, cost = instance.points, instance.nominal, instance.cost
    frames = context.frames(40)
    epsilons = torch.logspace(0.0, -3.0, frames, dtype=points.dtype)
    grid = torch.linspace(0.0, 1.0, context.size(161, 41), dtype=points.dtype)

    wasserstein = WassersteinAmbiguitySet(nominal, cost, RADIUS)
    exact_value, exact_slope = value_and_slope(wasserstein, grid, points)
    exact_minimizer = minimizer(wasserstein, points)
    exact_minimum = value_and_slope(wasserstein, exact_minimizer, points)[0]
    values, slopes, minimizers, minima = [], [], [], []
    for epsilon in epsilons.tolist():
        ball = SinkhornAmbiguitySet(nominal, cost, epsilon, RADIUS)
        value, slope = value_and_slope(ball, grid, points)
        point = minimizer(ball, points)
        values.append(value)
        slopes.append(slope)
        minimizers.append(point)
        minima.append(value_and_slope(ball, point, points)[0])
    sinkhorn_values, sinkhorn_slopes = torch.stack(values), torch.stack(slopes)
    sinkhorn_minimizers, sinkhorn_minima = torch.cat(minimizers), torch.cat(minima)
    value_gap = (sinkhorn_values - exact_value).abs().amax(-1).clamp(min=1e-12)
    point_gap = (sinkhorn_minimizers - exact_minimizer).abs().clamp(min=1e-12)

    color, limit_color = set_color(theme, "sinkhorn"), set_color(theme, "wasserstein")
    fig, (landscape, derivative, gaps) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.2, 1.2, 1]}
    )
    landscape.plot(
        grid.numpy(), exact_value.numpy(), color=limit_color, label="Wasserstein"
    )
    (value_line,) = landscape.plot([], [], color=color, label="Sinkhorn")
    landscape.plot(
        [float(exact_minimizer)],
        [float(exact_minimum)],
        "o",
        color=limit_color,
        markersize=8,
    )
    (minimum_marker,) = landscape.plot([], [], "D", color=color, markersize=7)
    landscape.set_xlim(0, 1)
    landscape.set_ylim(0, float(exact_value.max()) * 1.08)
    landscape.set_xlabel("decision $\\theta$")
    landscape.set_ylabel("worst-case loss")
    landscape.set_title("robust objective and its minimizer")
    landscape.legend(loc="lower right", fontsize=7.5)
    note = landscape.text(
        0.03, 0.97, "", transform=landscape.transAxes, ha="left", va="top", fontsize=9
    )

    derivative.plot(
        grid.numpy(), exact_slope.numpy(), color=limit_color, label="Wasserstein"
    )
    (slope_line,) = derivative.plot([], [], color=color, label="Sinkhorn")
    derivative.axhline(0.0, color=theme.muted, linewidth=0.8)
    margin = 0.08 * float(exact_slope.max() - exact_slope.min())
    derivative.set_xlim(0, 1)
    derivative.set_ylim(
        float(exact_slope.min()) - margin, float(exact_slope.max()) + margin
    )
    derivative.set_xlabel("decision $\\theta$")
    derivative.set_ylabel("$\\mathrm{d}V / \\mathrm{d}\\theta$")
    derivative.set_title("the derivative: jumps against a smooth curve")
    derivative.legend(loc="upper left", fontsize=7.5)

    epsilons_np = epsilons.numpy()
    gaps.plot(
        epsilons_np,
        value_gap.numpy(),
        color=color,
        alpha=0.45,
        label="$\\sup_\\theta |V_\\epsilon - V_W|$",
    )
    gaps.plot(
        epsilons_np,
        point_gap.numpy(),
        color=color,
        alpha=0.45,
        linestyle="--",
        label="$|\\theta_\\epsilon - \\theta_W|$",
    )
    (value_dot,) = gaps.plot([], [], "o", color=color, markersize=7)
    (point_dot,) = gaps.plot([], [], "s", color=color, markersize=7)
    gaps.set_xscale("log")
    gaps.set_yscale("log")
    gaps.set_xlim(epsilons_np[0], epsilons_np[-1])
    gaps.set_xlabel("entropic strength $\\epsilon$ (decreasing)")
    gaps.set_ylabel("gap to the Wasserstein solution")
    gaps.set_title("convergence as $\\epsilon \\to 0$")
    gaps.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        value_line.set_data(grid.numpy(), sinkhorn_values[frame].numpy())
        slope_line.set_data(grid.numpy(), sinkhorn_slopes[frame].numpy())
        minimum_marker.set_data(
            [float(sinkhorn_minimizers[frame])], [float(sinkhorn_minima[frame])]
        )
        value_dot.set_data([epsilons_np[frame]], [float(value_gap[frame])])
        point_dot.set_data([epsilons_np[frame]], [float(point_gap[frame])])
        note.set_text(
            f"$\\epsilon$ = {epsilons_np[frame]:.3g}\n"
            f"$\\sup_\\theta |V_\\epsilon - V_W|$ = {float(value_gap[frame]):.1e}\n"
            "$\\theta_\\epsilon - \\theta_W$ = "
            f"{float(sinkhorn_minimizers[frame] - exact_minimizer):+.1e}"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "sinkhorn-relaxation", fps=8
    )


def render_sinkhorn_adversary(context: Context) -> None:
    """Animate both adversaries and both derivatives as the decision sweeps."""
    theme = context.theme
    instance = demo_instance(size=context.size(24, 12))
    points, nominal, cost = instance.points, instance.nominal, instance.cost
    frames = context.frames(64)
    thetas = torch.linspace(0.05, 0.95, frames, dtype=points.dtype)
    grid = torch.linspace(0.0, 1.0, context.size(161, 41), dtype=points.dtype)

    wasserstein = WassersteinAmbiguitySet(nominal, cost, RADIUS)
    sinkhorn = SinkhornAmbiguitySet(nominal, cost, ADVERSARY_EPSILON, RADIUS)
    exact_adversary = adversaries(wasserstein, thetas, points)
    smooth_adversary = adversaries(sinkhorn, thetas, points)
    exact_slope = value_and_slope(wasserstein, grid, points)[1]
    smooth_slope = value_and_slope(sinkhorn, grid, points)[1]
    exact_at = value_and_slope(wasserstein, thetas, points)[1]
    smooth_at = value_and_slope(sinkhorn, thetas, points)[1]

    def moved(adversary: torch.Tensor) -> torch.Tensor:
        """Total-variation distance of each frame's adversary from the previous."""
        shift = 0.5 * (adversary[1:] - adversary[:-1]).abs().sum(-1)
        return torch.cat([shift[:1] * 0.0, shift])

    exact_moved, smooth_moved = moved(exact_adversary), moved(smooth_adversary)

    color, limit_color = set_color(theme, "sinkhorn"), set_color(theme, "wasserstein")
    fig, (support, derivative) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    width = float(points[1] - points[0]) * 0.8
    support.bar(
        points.numpy(), nominal.numpy(), width=width, color=theme.muted, alpha=0.3
    )
    smooth_bars = support.bar(
        points.numpy(), nominal.numpy(), width=width * 0.62, color=color
    )
    exact_bars = support.bar(
        points.numpy(),
        nominal.numpy(),
        width=width * 0.62,
        facecolor="none",
        edgecolor=limit_color,
        linewidth=1.5,
    )
    decision = support.axvline(
        0.0, color=theme.ink, linewidth=1.2, linestyle="--", alpha=0.7
    )
    ceiling = float(max(exact_adversary.max(), smooth_adversary.max())) * 1.15
    support.set_xlim(-width, 1.0 + width)
    support.set_ylim(0, ceiling)
    support.set_xlabel("scenario $x$; dashed: the decision $\\theta$")
    support.set_ylabel("probability mass")
    support.set_title("worst case of $(\\theta - x)^2$ at the current decision")
    support.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=theme.muted, alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, color=color),
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="none", edgecolor=limit_color, linewidth=1.5
            ),
        ],
        labels=[
            "nominal",
            f"Sinkhorn adversary ($\\epsilon$ = {ADVERSARY_EPSILON})",
            "Wasserstein adversary",
        ],
        loc="upper right",
        fontsize=8,
    )
    note = support.text(
        0.02, 0.97, "", transform=support.transAxes, ha="left", va="top", fontsize=9
    )

    derivative.plot(
        grid.numpy(), exact_slope.numpy(), color=limit_color, label="Wasserstein"
    )
    derivative.plot(grid.numpy(), smooth_slope.numpy(), color=color, label="Sinkhorn")
    derivative.axhline(0.0, color=theme.muted, linewidth=0.8)
    (exact_dot,) = derivative.plot([], [], "o", color=limit_color, markersize=8)
    (smooth_dot,) = derivative.plot([], [], "D", color=color, markersize=7)
    derivative.set_xlim(0, 1)
    derivative.set_xlabel("decision $\\theta$")
    derivative.set_ylabel("$\\mathrm{d}V / \\mathrm{d}\\theta$")
    derivative.set_title("the derivative each adversary implies")
    derivative.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        theta = float(thetas[frame])
        set_bar_heights(smooth_bars, smooth_adversary[frame])
        set_bar_heights(exact_bars, exact_adversary[frame])
        decision.set_xdata([theta, theta])
        exact_dot.set_data([theta], [float(exact_at[frame])])
        smooth_dot.set_data([theta], [float(smooth_at[frame])])
        note.set_text(
            f"$\\theta$ = {theta:.3f}\n"
            "mass moved since the last frame\n"
            f"Wasserstein {float(exact_moved[frame]):.3f}, "
            f"Sinkhorn {float(smooth_moved[frame]):.3f}"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "sinkhorn-adversary", fps=8
    )


FIGURES = (
    FigureSpec(
        name="sinkhorn-kernel",
        page="formulations.md",
        outputs=("sinkhorn-kernel-{theme}.gif",),
        render=render_sinkhorn_kernel,
    ),
    FigureSpec(
        name="sinkhorn-relaxation",
        page="examples/sinkhorn_relaxation_of_wasserstein_dro.md",
        outputs=("sinkhorn-relaxation-{theme}.gif",),
        render=render_sinkhorn_relaxation,
    ),
    FigureSpec(
        name="sinkhorn-adversary",
        page="examples/sinkhorn_relaxation_of_wasserstein_dro.md",
        outputs=("sinkhorn-adversary-{theme}.gif",),
        render=render_sinkhorn_adversary,
    ),
)
