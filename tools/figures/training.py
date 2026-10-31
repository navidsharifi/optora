"""Training dynamics: empirical risk against KL-DRO under subpopulation shift."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt

from optora.dro import KLAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color

SLOPE_MAJORITY = (1.0, 0.0)
SLOPE_MINORITY = (1.0, 1.5)
NOISE = 0.3
MINORITY_SHARE = 0.1
RADIUS = 0.2
HOLD_FRAMES = 10


@dataclass(frozen=True, slots=True)
class Trajectory:
    """What one training run records per step.

    Attributes:
        weights: Model weights, shape `(steps, 2)`.
        risk_majority: Exact risk on the majority group, shape `(steps,)`.
        risk_minority: Exact risk on the minority group, shape `(steps,)`.
        minority_mass: Worst-case mass on minority samples, shape `(steps,)`.
    """

    weights: torch.Tensor
    risk_majority: torch.Tensor
    risk_minority: torch.Tensor
    minority_mass: torch.Tensor


def train(
    features: torch.Tensor,
    targets: torch.Tensor,
    minority: torch.Tensor,
    radius: float,
    steps: int,
) -> Trajectory:
    """Fit a linear model to the KL-DRO objective with Adam and record its path.

    `radius == 0` is empirical risk minimization on the identical code path.
    The group risks come from the closed form of the data-generating process,
    `|w - a_g|^2 + b^2 + sigma^2`, since the inputs are standard normal in
    both groups.
    """
    dtype = features.dtype
    nominal = torch.full((features.shape[0],), 1.0 / features.shape[0], dtype=dtype)
    ambiguity_set = KLAmbiguitySet(nominal, radius)
    weight = torch.zeros(2, dtype=dtype, requires_grad=True)
    bias = torch.zeros((), dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    majority = torch.tensor(SLOPE_MAJORITY, dtype=dtype)
    minority_slope = torch.tensor(SLOPE_MINORITY, dtype=dtype)
    weights, risk_majority, risk_minority, minority_mass = [], [], [], []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (features @ weight + bias - targets) ** 2
        loss.retain_grad()
        value = ambiguity_set.worst_case_expectation(loss)
        value.backward()
        optimizer.step()
        with torch.no_grad():
            weights.append(weight.clone())
            risk_majority.append(((weight - majority) ** 2).sum() + bias**2 + NOISE**2)
            risk_minority.append(
                ((weight - minority_slope) ** 2).sum() + bias**2 + NOISE**2
            )
            minority_mass.append(loss.grad[minority].sum())
    return Trajectory(
        weights=torch.stack(weights),
        risk_majority=torch.stack(risk_majority),
        risk_minority=torch.stack(risk_minority),
        minority_mass=torch.stack(minority_mass),
    )


def render_training_dynamics(context: Context) -> None:
    """Animate the two models' weight paths, group risks and minority mass."""
    theme = context.theme
    dtype = torch.float64
    generator = torch.Generator().manual_seed(7)
    size = 400
    minority = torch.arange(size) < int(MINORITY_SHARE * size)
    features = torch.randn(size, 2, generator=generator, dtype=dtype)
    slopes = torch.where(
        minority[:, None],
        torch.tensor(SLOPE_MINORITY, dtype=dtype),
        torch.tensor(SLOPE_MAJORITY, dtype=dtype),
    )
    targets = (features * slopes).sum(-1) + NOISE * torch.randn(
        size, generator=generator, dtype=dtype
    )
    steps = context.size(160, 12)
    erm = train(features, targets, minority, 0.0, steps)
    dro = train(features, targets, minority, RADIUS, steps)
    frames = context.frames(60)
    shown = torch.linspace(0, steps - 1, frames).round().long()
    erm_color, dro_color = theme.muted, set_color(theme, "kl")

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(10.4, 3.9))
    left.plot(
        *zip(SLOPE_MAJORITY, SLOPE_MINORITY, strict=True),
        color=theme.axis,
        linewidth=1.2,
    )
    left.plot(*SLOPE_MAJORITY, "s", color=theme.ink, markersize=7)
    left.plot(*SLOPE_MINORITY, "s", color=theme.ink, markersize=7)
    left.text(
        SLOPE_MAJORITY[0] + 0.03,
        SLOPE_MAJORITY[1] - 0.05,
        "majority slope $a_A$",
        fontsize=8,
    )
    left.text(
        SLOPE_MINORITY[0] + 0.03, SLOPE_MINORITY[1], "minority slope $a_B$", fontsize=8
    )
    (erm_path,) = left.plot([], [], color=erm_color, linewidth=1.8)
    (dro_path,) = left.plot([], [], color=dro_color, linewidth=1.8)
    (erm_dot,) = left.plot([], [], "o", color=erm_color, markersize=7)
    (dro_dot,) = left.plot([], [], "o", color=dro_color, markersize=7)
    left.set_xlim(-0.1, 1.45)
    left.set_ylim(-0.1, 1.65)
    left.set_xlabel("$w_1$")
    left.set_ylabel("$w_2$")
    left.set_title("weights: DRO walks toward the minority")
    left.legend(
        handles=[
            plt.Line2D([], [], color=erm_color),
            plt.Line2D([], [], color=dro_color),
        ],
        labels=["ERM (radius 0)", f"KL-DRO (radius {RADIUS})"],
        loc="upper left",
    )

    lines = {}
    for key, color in (("erm", erm_color), ("dro", dro_color)):
        (lines[key, "A"],) = middle.plot([], [], color=color, linewidth=1.6, alpha=0.55)
        (lines[key, "B"],) = middle.plot([], [], color=color, linewidth=2.2)
    middle.set_xlim(0, steps)
    top = float(max(erm.risk_minority.max(), dro.risk_minority.max())) * 1.05
    middle.set_ylim(0, top)
    middle.set_xlabel("Adam step")
    middle.set_ylabel("exact group risk")
    middle.set_title("risk per group (bold: minority)")
    middle.legend(
        handles=[
            plt.Line2D([], [], color=theme.secondary, linewidth=1.6, alpha=0.55),
            plt.Line2D([], [], color=theme.secondary, linewidth=2.2),
        ],
        labels=["majority group", "minority group"],
        loc="upper right",
    )

    (mass_line,) = right.plot([], [], color=dro_color)
    right.axhline(MINORITY_SHARE, color=theme.muted, linewidth=1)
    right.text(
        steps,
        MINORITY_SHARE + 0.01,
        "nominal share  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    right.set_xlim(0, steps)
    right.set_ylim(0, float(dro.minority_mass.max()) * 1.25)
    right.set_xlabel("Adam step")
    right.set_ylabel("worst-case mass on the minority")
    right.set_title("the adversary's weight $\\sum_{i \\in B} q^\\star_i$")
    fig.tight_layout(w_pad=1.5)

    def update(frame: int) -> None:
        step = int(shown[min(frame, frames - 1)])
        erm_path.set_data(erm.weights[: step + 1, 0], erm.weights[: step + 1, 1])
        dro_path.set_data(dro.weights[: step + 1, 0], dro.weights[: step + 1, 1])
        erm_dot.set_data([erm.weights[step, 0]], [erm.weights[step, 1]])
        dro_dot.set_data([dro.weights[step, 0]], [dro.weights[step, 1]])
        axis = torch.arange(step + 1)
        lines["erm", "A"].set_data(axis, erm.risk_majority[: step + 1])
        lines["erm", "B"].set_data(axis, erm.risk_minority[: step + 1])
        lines["dro", "A"].set_data(axis, dro.risk_majority[: step + 1])
        lines["dro", "B"].set_data(axis, dro.risk_minority[: step + 1])
        mass_line.set_data(axis, dro.minority_mass[: step + 1])

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "training-dynamics", fps=12
    )


FIGURES = (
    FigureSpec(
        name="training-dynamics",
        page="training.md",
        outputs=("training-dynamics-{theme}.gif",),
        render=render_training_dynamics,
    ),
)
