r"""KL-DRO under subpopulation shift, animated over the radius.

The animation belongs to example 07. It trains the example's linear model
on the KL worst case at forty radii at once, from zero (empirical risk
minimization) to one half, and turns the radius like a dial: the fitted
slope walks from the majority slope toward the minority one along the
optimal trade-off path, the exact test-risk line
$R(\pi) = (1 - \pi) R_A + \pi R_B$ tilts until it crosses the ERM line at
$\pi^\star$, and the mass the adversary puts on the minority rises toward
the most the ball allows.
"""

from __future__ import annotations

import torch
from matplotlib import pyplot as plt

from optora.dro import KLAmbiguitySet
from tools.figures._base import Context, FigureSpec, set_color

HOLD_FRAMES = 10
SEED = 20261004
NUM_TRAIN = 400
MINORITY_FRACTION = 0.1
NOISE = 0.3
SLOPE_MAJORITY = (1.0, 0.0)
SLOPE_MINORITY = (1.0, 1.5)
MAX_RADIUS = 0.5
LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 0.98


def training_set(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw example 07's stratified training set.

    Returns:
        Inputs of shape `(n, 2)`, labels of shape `(n,)`, and the boolean
        minority indicator of shape `(n,)`; the minority points come last.
    """
    minority = torch.arange(NUM_TRAIN) >= NUM_TRAIN - round(
        MINORITY_FRACTION * NUM_TRAIN
    )
    inputs = torch.randn(NUM_TRAIN, 2, generator=generator, dtype=torch.float64)
    slopes = torch.where(
        minority[:, None],
        torch.tensor(SLOPE_MINORITY, dtype=torch.float64),
        torch.tensor(SLOPE_MAJORITY, dtype=torch.float64),
    )
    noise = torch.randn(NUM_TRAIN, generator=generator, dtype=torch.float64)
    return inputs, (inputs * slopes).sum(-1) + NOISE * noise, minority


def train(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    minority: torch.Tensor,
    radii: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one linear model per radius with Adam on the KL worst case.

    Args:
        inputs: Training inputs of shape `(n, 2)`.
        labels: Training labels of shape `(n,)`.
        minority: Minority indicator of shape `(n,)`.
        radii: KL radii of shape `(R,)`; zero is empirical risk minimization.
        steps: Adam steps, with example 07's learning rate and decay.

    Returns:
        Slopes of shape `(R, 2)`, intercepts of shape `(R,)`, and the mass the
        worst-case distribution puts on the minority, of shape `(R,)`.
    """
    nominal = torch.full((NUM_TRAIN,), 1.0 / NUM_TRAIN, dtype=torch.float64)
    ambiguity_set = KLAmbiguitySet(nominal, radii)
    weight = torch.zeros(radii.shape[0], 2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(radii.shape[0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LEARNING_RATE_DECAY)

    def losses() -> torch.Tensor:
        return (weight @ inputs.T + bias[:, None] - labels) ** 2

    for _ in range(steps):
        optimizer.zero_grad()
        ambiguity_set.worst_case_expectation(losses()).sum().backward()
        optimizer.step()
        scheduler.step()
    loss = losses()
    loss.retain_grad()
    ambiguity_set.worst_case_expectation(loss).sum().backward()
    assert loss.grad is not None
    return weight.detach(), bias.detach(), loss.grad[:, minority].sum(-1)


def group_risks(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return the exact risks `||w - a_g||^2 + b^2 + sigma^2`, shape `(..., 2)`."""
    slopes = torch.tensor([SLOPE_MAJORITY, SLOPE_MINORITY], dtype=weight.dtype)
    distance = ((weight[..., None, :] - slopes) ** 2).sum(-1)
    return distance + bias[..., None] ** 2 + NOISE**2


def allowed_minority_mass(radii: torch.Tensor) -> torch.Tensor:
    """Return the most mass a KL ball of each radius can place on the minority.

    Moving the group mass from the nominal share `f` to `m` costs the binary
    divergence `m log(m/f) + (1-m) log((1-m)/(1-f))` at least, so the largest
    feasible `m` solves that for the radius, by bisection.
    """
    share = MINORITY_FRACTION
    lower = torch.full_like(radii, share)
    upper = torch.full_like(radii, 1.0 - 1e-12)
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        cost = torch.special.xlogy(middle, middle / share) + torch.special.xlogy(
            1.0 - middle, (1.0 - middle) / (1.0 - share)
        )
        feasible = cost <= radii
        lower = torch.where(feasible, middle, lower)
        upper = torch.where(feasible, upper, middle)
    return lower


def render_shift_radius(context: Context) -> None:
    """Animate the fitted slope, the test-risk line and the minority mass over rho."""
    theme = context.theme
    frames = context.frames(40)
    radii = torch.linspace(0.0, MAX_RADIUS, frames, dtype=torch.float64)
    generator = torch.Generator().manual_seed(SEED)
    inputs, labels, minority = training_set(generator)
    weight, bias, minority_mass = train(
        inputs, labels, minority, radii, context.size(300, 12)
    )
    risks = group_risks(weight, bias)
    mixing = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    test_risk = (1.0 - mixing[None, :]) * risks[:, :1] + mixing[None, :] * risks[:, 1:]
    # The crossing pi* of each radius's line with the ERM line (radius zero).
    gap = risks - risks[0]
    crossing = gap[:, 0] / (gap[:, 0] - gap[:, 1])
    allowed = allowed_minority_mass(radii)
    erm_color, color = theme.muted, set_color(theme, "kl")

    fig, (plane, lines, mass) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1, 1.3, 1]}
    )
    plane.plot(
        *zip(SLOPE_MAJORITY, SLOPE_MINORITY, strict=True),
        color=theme.axis,
        linewidth=1.2,
        label="optimal trade-off path $w^\\star(\\pi)$",
    )
    plane.plot(*SLOPE_MAJORITY, "s", color=theme.ink, markersize=7)
    plane.plot(*SLOPE_MINORITY, "s", color=theme.ink, markersize=7)
    plane.text(
        SLOPE_MAJORITY[0] + 0.03,
        SLOPE_MAJORITY[1] - 0.05,
        "majority slope $a_A$",
        fontsize=8,
    )
    plane.text(
        SLOPE_MINORITY[0] + 0.03, SLOPE_MINORITY[1], "minority slope $a_B$", fontsize=8
    )
    plane.plot(
        [float(weight[0, 0])],
        [float(weight[0, 1])],
        "o",
        color=erm_color,
        markersize=7,
        label="ERM",
    )
    (trail,) = plane.plot(
        [], [], color=color, linewidth=1.8, label="KL-DRO as $\\rho$ grows"
    )
    (dot,) = plane.plot([], [], "o", color=color, markersize=8)
    plane.set_xlim(0.6, 1.45)
    plane.set_ylim(-0.1, 1.65)
    plane.set_xlabel("$w_1$")
    plane.set_ylabel("$w_2$")
    plane.set_title("the fitted slope walks toward the minority")
    plane.legend(loc="center right", fontsize=7.5)

    mixing_np = mixing.numpy()
    lines.plot(mixing_np, test_risk[0].numpy(), color=erm_color, label="ERM")
    (dro_line,) = lines.plot([], [], color=color, label="KL-DRO at the current $\\rho$")
    (cross,) = lines.plot(
        [], [], "D", color=theme.ink, markersize=7, label="crossing $\\pi^\\star$"
    )
    lines.axvline(
        MINORITY_FRACTION,
        color=theme.muted,
        linewidth=1,
        linestyle=":",
        label="training mix",
    )
    lines.set_xlim(0, 1)
    lines.set_ylim(0, float(test_risk.max()) * 1.05)
    lines.set_xlabel("test share $\\pi$ of the minority group")
    lines.set_ylabel("exact test risk $R(\\pi)$")
    lines.set_title("DRO pays at the training mix, wins past $\\pi^\\star$")
    lines.legend(loc="upper left", fontsize=7.5)
    note = lines.text(
        0.98, 0.03, "", transform=lines.transAxes, ha="right", va="bottom", fontsize=9
    )

    radii_np = radii.numpy()
    mass.plot(
        radii_np,
        allowed.numpy(),
        color=theme.ink,
        linestyle="--",
        linewidth=1.2,
        label="most the ball allows",
    )
    mass.plot(
        radii_np,
        minority_mass.numpy(),
        color=color,
        label="worst-case mass $q^\\star_B$",
    )
    (mass_dot,) = mass.plot([], [], "o", color=color, markersize=8)
    mass.axhline(MINORITY_FRACTION, color=theme.muted, linewidth=1)
    mass.text(
        float(radii[-1]),
        MINORITY_FRACTION + 0.01,
        "nominal share  ",
        ha="right",
        fontsize=8,
        color=theme.secondary,
    )
    mass.set_xlim(0, float(radii[-1]))
    mass.set_ylim(0, float(allowed.max()) * 1.1)
    mass.set_xlabel("KL radius $\\rho$")
    mass.set_ylabel("mass on the minority points")
    mass.set_title("the adversary's weight on the minority")
    mass.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        radius = float(radii[frame])
        trail.set_data(weight[: frame + 1, 0].numpy(), weight[: frame + 1, 1].numpy())
        dot.set_data([float(weight[frame, 0])], [float(weight[frame, 1])])
        dro_line.set_data(mixing_np, test_risk[frame].numpy())
        pi_star = float(crossing[frame])
        if frame > 0 and 0.0 < pi_star < 1.0:
            erm_at_crossing = (1.0 - pi_star) * float(risks[0, 0]) + pi_star * float(
                risks[0, 1]
            )
            cross.set_data([pi_star], [erm_at_crossing])
            crossing_text = f"$\\pi^\\star$ = {pi_star:.3f}"
        else:
            cross.set_data([], [])
            crossing_text = "$\\pi^\\star$: none"
        mass_dot.set_data([radius], [float(minority_mass[frame])])
        note.set_text(
            f"$\\rho$ = {radius:.3f}\n"
            f"$R_A$ = {float(risks[frame, 0]):.3f}, "
            f"$R_B$ = {float(risks[frame, 1]):.3f}\n"
            f"{crossing_text}"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "shift-radius", fps=8)


FIGURES = (
    FigureSpec(
        name="shift-radius",
        page="examples/subpopulation_shift_training.md",
        outputs=("shift-radius-{theme}.gif",),
        render=render_shift_radius,
    ),
)
