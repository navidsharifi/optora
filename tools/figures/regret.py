r"""Distributionally robust regret against robust loss, animated over the radius.

Both animations belong to the example page on Wasserstein distributionally
robust regret optimization (Fiechtner and Blanchet 2025). The first draws
the quadratic case, where the regret under a distribution $q$ is
$(\theta - \mathbb{E}_q[\xi])^2$ and the robust regret is the squared
distance to the farther end of the reachable-mean interval, so the whole
landscape has a closed form in terms of two worst-case expectations. The
second draws the paper's newsvendor, where the benchmark maximization is a
maximum over the scenarios themselves.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from optora.dro import WassersteinAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_bar_heights
from tools.figures.problems import Instance, demo_instance

HOLD_FRAMES = 10
ORDER_COST, SALE_PRICE = 1.0, 3.0
BISECTION_STEPS = 30


def worst_case(
    ambiguity_set: WassersteinAmbiguitySet, loss: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the worst-case expectation of `loss` and its maximizer (Danskin)."""
    loss = loss.detach().requires_grad_()
    value = ambiguity_set.worst_case_expectation(loss)
    (adversary,) = torch.autograd.grad(value.sum(), loss)
    return value.detach(), adversary


def argmin_by_bisection(
    derivative: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Minimize convex functions by bisecting the sign of their derivative."""
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        positive = derivative(middle) > 0.0
        upper = torch.where(positive, middle, upper)
        lower = torch.where(positive, lower, middle)
    return 0.5 * (lower + upper)


def wasserstein_ball(
    instance: Instance, radii: torch.Tensor, power: float
) -> WassersteinAmbiguitySet:
    """Return the ball with cost `|x_i - x_j| ** power` and one radius per row."""
    return WassersteinAmbiguitySet(
        instance.nominal, instance.cost**power, radius=radii.unsqueeze(-1)
    )


def _policy_axes(
    axes: Axes, radii: np.ndarray, label: str, log: bool, legend: str
) -> None:
    """Style a decisions-versus-radius panel."""
    if log:
        axes.set_xscale("log")
    axes.set_xlim(radii[0], radii[-1])
    axes.set_xlabel("Wasserstein radius $\\delta$" + ("  (log)" if log else ""))
    axes.set_ylabel(label)
    axes.legend(loc=legend, fontsize=7.5)


def render_regret_landscape(context: Context) -> None:
    r"""Animate the robust-loss and robust-regret landscapes of a quadratic loss."""
    theme = context.theme
    instance = demo_instance(size=context.size(24, 12))
    points, nominal = instance.points, instance.nominal
    frames = context.frames(40)
    mean = float(nominal @ points)
    # Stop before either end of the reachable-mean interval saturates; up to
    # there the interval is symmetric about the mean and DRRO sits exactly
    # on the empirical-risk decision.
    reach = min(mean - float(points.min()), float(points.max()) - mean)
    radii = torch.linspace(0.01 * reach, 0.9 * reach, frames, dtype=points.dtype)
    ball = wasserstein_ball(instance, radii, power=1.0)
    grid = torch.linspace(0.0, 1.0, context.size(161, 41), dtype=points.dtype)

    # Reachable means: the worst case of xi and of -xi. Regret under q is
    # (theta - mean_q)^2, so the robust regret is the larger of the two
    # squared distances, and the two maximizers are the regret adversaries.
    mean_high, push_up = worst_case(ball, points.expand(frames, 1, -1))
    mean_low, push_down = worst_case(ball, -points.expand(frames, 1, -1))
    mean_low = -mean_low
    branches = torch.stack(
        [(grid - mean_low) ** 2, (grid - mean_high) ** 2], dim=0
    )  # (2, frames, G)
    regret = branches.amax(0)
    robust_loss = ball.worst_case_expectation((grid[None, :, None] - points) ** 2)
    nominal_loss = ((grid[:, None] - points) ** 2 @ nominal).expand(frames, -1)
    theta_wdro = argmin_by_bisection(
        lambda theta: (
            2.0
            * (
                worst_case(ball, (theta[:, None, None] - points) ** 2)[1]
                * (theta[:, None, None] - points)
            ).sum(-1)[:, 0]
        ),
        torch.zeros(frames, dtype=points.dtype),
        torch.ones(frames, dtype=points.dtype),
    )
    theta_drro = 0.5 * (mean_low + mean_high)[:, 0]
    wdro_adversary = worst_case(ball, (theta_wdro[:, None, None] - points) ** 2)[1][
        :, 0
    ]

    erm_color, wdro_color, drro_color = theme.muted, theme.series[0], theme.series[3]
    fig, (support, landscape, paths) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.1, 1.3, 1]}
    )
    width = float(points[1] - points[0]) * 0.8
    support.bar(
        points.numpy(), nominal.numpy(), width=width, color=theme.muted, alpha=0.3
    )
    wdro_bars = support.bar(
        points.numpy(), nominal.numpy(), width=width * 0.5, color=wdro_color
    )
    up_bars = support.bar(
        points.numpy() - width * 0.25,
        nominal.numpy(),
        width=width * 0.25,
        color=drro_color,
    )
    down_bars = support.bar(
        points.numpy() + width * 0.25,
        nominal.numpy(),
        width=width * 0.25,
        color=drro_color,
        alpha=0.5,
    )
    support.set_ylim(0, 1.0)
    support.set_xlabel("scenario $\\xi$")
    support.set_ylabel("probability")
    support.set_title("the adversaries")
    support.text(
        0.5,
        0.97,
        "grey: nominal\nblue: worst case for the loss at $\\theta_{\\mathrm{WDRO}}$\n"
        "orange: the two regret adversaries at $\\theta_{\\mathrm{DRRO}}$\n"
        "(mean pushed up, and down in the paler bars)",
        transform=support.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=theme.secondary,
    )

    landscape.plot(
        grid.numpy(),
        nominal_loss[0].numpy(),
        color=erm_color,
        label="nominal loss (ERM)",
    )
    (wdro_line,) = landscape.plot(
        [], [], color=wdro_color, label="worst-case loss (WDRO)"
    )
    branch_lines = [
        landscape.plot([], [], color=drro_color, linewidth=1.0, alpha=0.45)[0]
        for _ in range(2)
    ]
    (regret_line,) = landscape.plot(
        [], [], color=drro_color, label="robust regret (DRRO)"
    )
    (erm_marker,) = landscape.plot(
        [mean], [float(nominal_loss[0].min())], "o", color=erm_color, markersize=7
    )
    (wdro_marker,) = landscape.plot([], [], "o", color=wdro_color, markersize=8)
    (drro_marker,) = landscape.plot([], [], "D", color=drro_color, markersize=8)
    landscape.set_xlim(0, 1)
    landscape.set_ylim(0, float(robust_loss.max()) * 1.45)
    landscape.set_xlabel("decision $\\theta$")
    landscape.set_ylabel("objective")
    landscape.set_title("objectives in $\\theta$")
    landscape.legend(loc="upper left", fontsize=7.5)
    note = landscape.text(
        0.98, 0.97, "", transform=landscape.transAxes, ha="right", va="top", fontsize=9
    )

    radii_np = radii.numpy()
    paths.axhline(mean, color=erm_color, linewidth=1.2, label="ERM")
    paths.plot(radii_np, theta_wdro.numpy(), color=wdro_color, label="WDRO")
    paths.plot(radii_np, theta_drro.numpy(), color=drro_color, label="DRRO")
    (wdro_dot,) = paths.plot([], [], "o", color=wdro_color, markersize=8)
    (drro_dot,) = paths.plot([], [], "D", color=drro_color, markersize=8)
    _policy_axes(paths, radii_np, "decision $\\theta_\\delta$", False, "upper left")
    paths.set_title("DRRO stays at ERM, WDRO drifts")
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        set_bar_heights(wdro_bars, wdro_adversary[frame])
        set_bar_heights(up_bars, push_up[frame, 0])
        set_bar_heights(down_bars, push_down[frame, 0])
        wdro_line.set_data(grid.numpy(), robust_loss[frame].numpy())
        for line, branch in zip(branch_lines, branches[:, frame], strict=True):
            line.set_data(grid.numpy(), branch.numpy())
        regret_line.set_data(grid.numpy(), regret[frame].numpy())
        wdro_marker.set_data(
            [theta_wdro[frame].item()], [robust_loss[frame].min().item()]
        )
        drro_marker.set_data([theta_drro[frame].item()], [regret[frame].min().item()])
        wdro_dot.set_data([radii_np[frame]], [theta_wdro[frame].item()])
        drro_dot.set_data([radii_np[frame]], [theta_drro[frame].item()])
        note.set_text(
            f"$\\delta$ = {radii_np[frame]:.3f}\n"
            "$\\theta_{\\mathrm{WDRO}} - \\theta_{\\mathrm{ERM}}$ = "
            f"{theta_wdro[frame].item() - mean:+.3f}\n"
            "$\\theta_{\\mathrm{DRRO}} - \\theta_{\\mathrm{ERM}}$ = "
            f"{theta_drro[frame].item() - mean:+.1e}"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "regret-landscape", fps=10
    )


def newsvendor_loss(theta: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Newsvendor loss in profit form for decisions of shape `(..., 1)`."""
    return ORDER_COST * theta - SALE_PRICE * torch.minimum(theta, demand)


def render_regret_newsvendor(context: Context) -> None:
    """Animate the four newsvendor policies and their adversaries over the radius."""
    theme = context.theme
    instance = demo_instance(size=context.size(32, 12))
    demand, nominal = instance.points, instance.nominal
    size = instance.size
    frames = context.frames(40)
    radii = torch.logspace(-4.5, -1.3, frames, dtype=demand.dtype)
    ball = wasserstein_ball(instance, radii, power=2.0)
    hindsight = -(SALE_PRICE - ORDER_COST) * demand
    lower = torch.zeros(frames, dtype=demand.dtype)
    upper = torch.ones(frames, dtype=demand.dtype)

    def loss_of(theta: torch.Tensor) -> torch.Tensor:
        return newsvendor_loss(theta[:, None, None], demand)

    def subgradient(adversary: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return ORDER_COST - SALE_PRICE * (adversary * (demand > theta[:, None])).sum(-1)

    def ex_ante_adversary(theta: torch.Tensor) -> torch.Tensor:
        # Exact for a piecewise-linear loss: the benchmark maximizer is a scenario.
        value, adversary = worst_case(
            ball, loss_of(theta) - newsvendor_loss(demand[None, :, None], demand)
        )
        best = value.argmax(-1)
        return adversary[torch.arange(frames), best]

    theta_erm = argmin_by_bisection(
        lambda theta: subgradient(nominal.expand(frames, -1), theta), lower, upper
    )
    theta_wdro = argmin_by_bisection(
        lambda theta: subgradient(worst_case(ball, loss_of(theta))[1][:, 0], theta),
        lower,
        upper,
    )
    theta_ex_post = argmin_by_bisection(
        lambda theta: subgradient(
            worst_case(ball, loss_of(theta) - hindsight)[1][:, 0], theta
        ),
        lower,
        upper,
    )
    theta_ex_ante = argmin_by_bisection(
        lambda theta: subgradient(ex_ante_adversary(theta), theta), lower, upper
    )
    adversaries = {
        "wdro": worst_case(ball, loss_of(theta_wdro))[1][:, 0],
        "ex_post": worst_case(ball, loss_of(theta_ex_post) - hindsight)[1][:, 0],
        "ex_ante": ex_ante_adversary(theta_ex_ante),
    }
    policies = {
        "erm": theta_erm,
        "wdro": theta_wdro,
        "ex_post": theta_ex_post,
        "ex_ante": theta_ex_ante,
    }
    labels = {
        "erm": "ERM",
        "wdro": "Wasserstein DRO",
        "ex_post": "ex-post DRRO (cost form)",
        "ex_ante": "ex-ante DRRO",
    }
    colors = {
        "erm": theme.muted,
        "wdro": theme.series[0],
        "ex_post": theme.series[2],
        "ex_ante": theme.series[3],
    }

    fig, (support, paths) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    width = 1.0 / (size - 1)
    support.bar(
        demand.numpy(), nominal.numpy(), width=width * 0.9, color=theme.muted, alpha=0.3
    )
    lines = {
        key: support.plot(
            [], [], color=colors[key], linewidth=1.8, drawstyle="steps-mid"
        )[0]
        for key in adversaries
    }
    markers = {
        key: support.axvline(0.0, color=colors[key], linewidth=1.4, linestyle="--")
        for key in policies
    }
    support.set_xlim(-width, 1.0 + width)
    support.set_ylim(0, 0.7)
    support.set_xlabel("demand scenario $\\xi$")
    support.set_ylabel("probability")
    support.set_title("adversarial demand per policy; dashed: order quantity")
    note = support.text(
        0.02, 0.97, "", transform=support.transAxes, va="top", fontsize=9
    )

    radii_np = radii.numpy()
    dots = {}
    for key, theta in policies.items():
        paths.step(
            radii_np, theta.numpy(), where="post", color=colors[key], label=labels[key]
        )
        dots[key] = paths.plot([], [], "o", color=colors[key], markersize=7)[0]
    orders = torch.stack(list(policies.values()))
    paths.set_ylim(float(orders.min()) - 0.05, float(orders.max()) + 0.05)
    _policy_axes(paths, radii_np, "order quantity", True, "lower left")
    paths.set_title("DRO orders less, regret orders more")
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        for key, line in lines.items():
            line.set_data(demand.numpy(), adversaries[key][frame].numpy())
        for key, marker in markers.items():
            marker.set_xdata([policies[key][frame].item()] * 2)
        for key, dot in dots.items():
            dot.set_data([radii_np[frame]], [policies[key][frame].item()])
        note.set_text(f"$\\delta$ = {radii_np[frame]:.1e}")

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "regret-newsvendor", fps=10
    )


FIGURES = (
    FigureSpec(
        name="regret-landscape",
        page="examples/distributionally_robust_regret.md",
        outputs=("regret-landscape-{theme}.gif",),
        render=render_regret_landscape,
    ),
    FigureSpec(
        name="regret-newsvendor",
        page="examples/distributionally_robust_regret.md",
        outputs=("regret-newsvendor-{theme}.gif",),
        render=render_regret_newsvendor,
    ),
)
