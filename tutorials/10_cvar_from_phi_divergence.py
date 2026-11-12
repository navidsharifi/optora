r"""Recovering CVaR from a phi-divergence ambiguity set.

Conditional value-at-risk at level alpha, the mean of the worst alpha-fraction
of outcomes, is itself a distributionally robust objective. It is the worst
expected loss over every distribution q whose likelihood ratio to the nominal
p never exceeds 1/alpha (Artzner et al. 1999; Rockafellar and Uryasev 2002).
That constraint is a phi-divergence ball (Ben-Tal et al. 2013) for the
indicator generator

    phi_alpha(t) = 0 for 0 <= t <= 1/alpha, and +inf otherwise,

whose convex conjugate is phi_alpha^*(s) = max(s, 0) / alpha. With that
conjugate in the general phi-DRO dual the multiplier eta cancels, which
leaves the Rockafellar-Uryasev formula

    CVaR_alpha(loss) = min over lam of  lam + E_p[(loss - lam)_+] / alpha.

No new solver is needed. This script builds the ball with the general
`PhiAmbiguitySet`, solves its dual with `GradientDescent`, and checks the
value against CVaR computed directly from its definition, the mean of the
upper alpha-tail. It checks the dedicated `CVaRAmbiguitySet` as well.

Two numerical details matter. First, the dual is piecewise linear in lam,
so a fixed step size only reaches the optimum to within a multiple of that
step. The script therefore shrinks the step in stages and lets each stage
warm-start from the last, which is a diminishing-step subgradient method.
By weak duality every stage returns an upper bound on CVaR, and the bound
closes from above. Second, the radius is zero, and the set still solves.
Most ambiguity sets return the nominal mean at radius 0.0 without solving,
which is right for a generator that vanishes only at t = 1. This generator
vanishes on all of [0, 1/alpha], so its ball of radius zero is the full
CVaR set, and for this reason `PhiAmbiguitySet` with a dual solver solves
at every radius.

The script also tests a tempting shortcut, a total-variation ball with
budget (1 - alpha)/alpha, and shows that it does not give CVaR. That
number bounds how far each likelihood ratio may rise above one, which is a
pointwise (sup-norm) constraint, while total variation limits the L1 mass
that moves. For alpha <= 1/2 the budget is at least one and the TV worst
case is the largest loss. The correct link is an identity. The TV worst
case at budget 1 - alpha equals alpha * CVaR_alpha + (1 - alpha) *
max(loss), because the TV adversary keeps the upper alpha-tail and moves
all remaining mass onto the single worst scenario. CVaR can therefore be
read off a TV ball, but only after that affine correction.

References:
    Philippe Artzner, Freddy Delbaen, Jean-Marc Eber and David Heath,
    "Coherent Measures of Risk", Mathematical Finance 9(3), 1999.
    Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg
    and Gijs Rennen, "Robust Solutions of Optimization Problems Affected by
    Uncertain Probabilities", Management Science 59(2), 2013.
    R. Tyrrell Rockafellar and Stanislav Uryasev, "Conditional Value-at-Risk
    for General Loss Distributions", Journal of Banking and Finance 26(7),
    2002.
    Carlo Acerbi and Dirk Tasche, "On the Coherence of Expected Shortfall",
    Journal of Banking and Finance 26(7), 2002.

Run:
    python tutorials/10_cvar_from_phi_divergence.py
"""

import functools

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.divergences import PhiDivergence, TotalVariationDivergence
from optora.dro import (
    CVaRAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
    cvar_conjugate,
)
from optora.solvers import GradientDescent

SEED = 20261102
NUM_SCENARIOS = 400
ALPHAS = torch.logspace(0.0, -2.0, 25, dtype=torch.float64)
# Step sizes in the standardized units of `DualAmbiguitySet`. Each stage
# warm-starts from the previous optimum.
STEP_SCHEDULE = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
STEPS_PER_STAGE = 1500
DUAL_TOLERANCE = 1e-6
EXACT_TOLERANCE = 1e-12
LANDSCAPE_ALPHAS = (0.5, 0.1, 0.02)
LANDSCAPE_ETAS = (1.0, 10.0)


def sample_problem() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a weighted sample of heavy-tailed (log-normal) losses.

    The weights are a uniform draw from the simplex, so no scenario boundary
    lines up with a round value of alpha, and the boundary scenario is only
    partly in the tail at almost every level.
    """
    generator = torch.Generator().manual_seed(SEED)
    loss = torch.exp(
        torch.randn(NUM_SCENARIOS, generator=generator, dtype=torch.float64)
    )
    weights = -torch.log(
        torch.rand(NUM_SCENARIOS, generator=generator, dtype=torch.float64)
    )
    return weights / weights.sum(), loss


def tail_mean(
    nominal: torch.Tensor, loss: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """Compute CVaR from its definition, the mean of the upper alpha-tail.

    Scenarios are taken in decreasing order of loss with their full nominal
    mass until the cumulative mass reaches alpha, and the boundary scenario
    contributes the remainder (Acerbi and Tasche 2002).
    """
    order = torch.argsort(loss, descending=True)
    sorted_loss, sorted_mass = loss[order], nominal[order]
    mass_before = torch.cumsum(sorted_mass, dim=0) - sorted_mass
    taken = torch.clamp(
        torch.minimum(sorted_mass, alphas.unsqueeze(-1) - mass_before), min=0.0
    )
    return taken @ sorted_loss / alphas


def cvar_generator(ratio: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Return the indicator generator, zero on `[0, 1/alpha]` and `+inf` beyond."""
    return torch.zeros_like(ratio).masked_fill(ratio > 1.0 / alpha, torch.inf)


def phi_divergence_cvar(
    nominal: torch.Tensor, loss: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """Solve the CVaR phi-ball dual at every alpha, one stage per step size.

    Returns:
        The dual value after each stage, of shape `(stages, len(alphas))`.
    """
    alpha = alphas.unsqueeze(-1)
    ambiguity_set = PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(functools.partial(cvar_generator, alpha=alpha)),
        radius=0.0,
        phi_conjugate=functools.partial(cvar_conjugate, alpha=alpha),
    )
    batched = loss.expand(alphas.shape[0], -1)
    stages = []
    for step_size in STEP_SCHEDULE:
        ambiguity_set.dual_solver = GradientDescent(
            step_size=step_size, max_iter=STEPS_PER_STAGE, tol=1e-12
        )
        stages.append(ambiguity_set.worst_case_expectation(batched).detach())
    return torch.stack(stages)


def cvar_maximizers(
    nominal: torch.Tensor, loss: torch.Tensor, alphas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CVaR from `CVaRAmbiguitySet` and its maximizers.

    By Danskin's theorem the gradient of the worst case with respect to the
    loss is the worst-case distribution itself.
    """
    batched = loss.expand(alphas.shape[0], -1).clone().requires_grad_()
    values = CVaRAmbiguitySet(nominal, alphas).worst_case_expectation(batched)
    (maximizers,) = torch.autograd.grad(values.sum(), batched)
    return values.detach(), maximizers


def total_variation_value(
    nominal: torch.Tensor, loss: torch.Tensor, radii: torch.Tensor
) -> torch.Tensor:
    """Return the total-variation worst case at every radius in `radii`."""
    ambiguity_set = TotalVariationAmbiguitySet(nominal, radius=radii)
    return ambiguity_set.worst_case_expectation(loss.expand(radii.shape[0], -1))


def dual_landscape(
    nominal: torch.Tensor,
    loss: torch.Tensor,
    lams: torch.Tensor,
    alpha: float,
    eta: float,
) -> torch.Tensor:
    """Evaluate the phi dual at radius zero over a grid of `lam` for fixed `eta`."""
    shift = (loss - lams.unsqueeze(-1)) / eta
    return lams + eta * cvar_conjugate(shift, alpha) @ nominal


def relative_gap(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the elementwise relative gap `(value - reference) / reference`."""
    return (value - reference) / reference


def main() -> None:
    nominal, loss = sample_problem()
    max_loss = loss.max()
    direct = tail_mean(nominal, loss, ALPHAS)

    dual_stages = phi_divergence_cvar(nominal, loss, ALPHAS)
    dual_gaps = relative_gap(dual_stages, direct)
    closed_form, maximizers = cvar_maximizers(nominal, loss, ALPHAS)

    print("route                                   max relative gap to tail mean")
    for stage, step_size in enumerate(STEP_SCHEDULE):
        gap = dual_gaps[stage].abs().max().item()
        print(f"  PhiAmbiguitySet dual, step {step_size:.0e}       {gap:.2e}")
    closed_gap = relative_gap(closed_form, direct).abs().max().item()
    print(f"  CVaRAmbiguitySet closed form             {closed_gap:.2e}")
    assert dual_gaps.min().item() > -EXACT_TOLERANCE, "dual fell below CVaR"
    assert dual_gaps[-1].abs().max().item() < DUAL_TOLERANCE
    assert closed_gap < EXACT_TOLERANCE

    ratio_excess = (maximizers / nominal * ALPHAS.unsqueeze(-1)).amax(dim=-1) - 1.0
    tv_of_maximizer = TotalVariationDivergence()(maximizers, nominal)
    print(f"max_i q*_i / (p_i / alpha) - 1             {ratio_excess.max().item():.2e}")
    print(
        "max TV(q*, p) - (1 - alpha)               "
        f"{(tv_of_maximizer - (1.0 - ALPHAS)).max().item():.2e}"
    )
    assert ratio_excess.max().item() < EXACT_TOLERANCE
    assert (tv_of_maximizer <= 1.0 - ALPHAS + EXACT_TOLERANCE).all()

    tv_suggested = total_variation_value(nominal, loss, (1.0 - ALPHAS) / ALPHAS)
    tv_matched = total_variation_value(nominal, loss, 1.0 - ALPHAS)
    mixture = ALPHAS * direct + (1.0 - ALPHAS) * max_loss
    recovered = (tv_matched - (1.0 - ALPHAS) * max_loss) / ALPHAS
    saturated = ALPHAS <= 0.5
    print(
        "TV budget (1 - alpha)/alpha, max relative gap to CVaR       "
        f"{relative_gap(tv_suggested, direct).abs().max().item():.2e}"
    )
    print(
        "TV budget 1 - alpha vs alpha CVaR + (1 - alpha) max loss    "
        f"{relative_gap(tv_matched, mixture).abs().max().item():.2e}"
    )
    print(
        "CVaR recovered from the TV ball by the affine correction    "
        f"{relative_gap(recovered, direct).abs().max().item():.2e}"
    )
    assert torch.equal(tv_suggested[saturated], max_loss.expand(int(saturated.sum())))
    assert relative_gap(tv_matched, mixture).abs().max().item() < EXACT_TOLERANCE
    assert relative_gap(recovered, direct).abs().max().item() < 1e-10

    fig, (values_axes, gap_axes, dual_axes) = plt.subplots(
        1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]}
    )

    alphas = ALPHAS.numpy()
    values_axes.axhline(max_loss.item(), color="0.6", linewidth=1.0)
    values_axes.text(
        alphas[-1], max_loss.item(), "max loss ", va="bottom", ha="right", fontsize=8
    )
    values_axes.plot(
        alphas,
        tv_suggested.numpy(),
        "--",
        color="C2",
        label="TV, budget $(1-\\alpha)/\\alpha$",
    )
    values_axes.plot(
        alphas, tv_matched.numpy(), color="C2", label="TV, budget $1-\\alpha$"
    )
    values_axes.plot(
        alphas,
        direct.numpy(),
        color="C3",
        linewidth=3.0,
        label="tail mean (definition)",
    )
    values_axes.plot(
        alphas,
        closed_form.numpy(),
        ":",
        color="C0",
        linewidth=1.6,
        label="CVaRAmbiguitySet",
    )
    values_axes.plot(
        alphas,
        dual_stages[-1].numpy(),
        "o",
        color="C7",
        markerfacecolor="none",
        markeredgecolor="C7",
        markersize=7,
        label="PhiAmbiguitySet + cvar_conjugate",
    )
    values_axes.set_xscale("log")
    values_axes.invert_xaxis()
    values_axes.set_yscale("log")
    values_axes.set_xlabel("tail probability $\\alpha$ (log, decreasing)")
    values_axes.set_ylabel("worst-case expected loss")
    values_axes.set_title("three routes to CVaR, and the TV shortcut")
    values_axes.legend(loc="lower right", fontsize=7.5)

    for stage, step_size in enumerate(STEP_SCHEDULE):
        gap_axes.plot(
            alphas,
            dual_gaps[stage].clamp(min=1e-16).numpy(),
            "o-",
            color="C7",
            alpha=0.25 + 0.75 * stage / (len(STEP_SCHEDULE) - 1),
            markersize=3.5,
            markeredgewidth=0.0,
            linewidth=1.4,
            label=f"step {step_size:.0e}",
        )
    gap_axes.axhline(DUAL_TOLERANCE, color="0.6", linewidth=1.0, linestyle="--")
    gap_axes.set_xscale("log")
    gap_axes.invert_xaxis()
    gap_axes.set_yscale("log")
    gap_axes.set_xlabel("tail probability $\\alpha$ (log, decreasing)")
    gap_axes.set_ylabel("(dual value - CVaR) / CVaR")
    gap_axes.set_title("the dual bound closes from above")
    gap_axes.set_ylim(1e-17, 1.0)
    gap_axes.legend(loc="upper left", fontsize=7.5, ncol=3)

    lams = torch.linspace(
        0.0, float(torch.quantile(loss, 0.995)), 600, dtype=torch.float64
    )
    for index, alpha in enumerate(LANDSCAPE_ALPHAS):
        color = f"C{index * 2 + 1}"
        level = torch.tensor([alpha], dtype=torch.float64)
        cvar = tail_mean(nominal, loss, level).item()
        dual_axes.axhline(cvar, color=color, linewidth=0.8, alpha=0.6)
        small_eta, large_eta = (
            dual_landscape(nominal, loss, lams, alpha, eta) for eta in LANDSCAPE_ETAS
        )
        dual_axes.plot(
            lams.numpy(), small_eta.numpy(), color=color, label=f"$\\alpha$ = {alpha}"
        )
        dual_axes.plot(
            lams.numpy(),
            large_eta.numpy(),
            "o",
            color=color,
            markevery=40,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=6,
        )
        value_at_risk = lams[torch.argmin(small_eta)].item()
        dual_axes.plot([value_at_risk], [cvar], "D", color=color, markersize=7)
    dual_axes.set_xlabel("dual variable $\\lambda$")
    dual_axes.set_ylabel("dual objective")
    dual_axes.set_title(
        "$\\eta$ drops out: line $\\eta$ = 1, circles $\\eta$ = 10", fontsize=10.5
    )
    dual_axes.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "10_cvar_from_phi_divergence")


if __name__ == "__main__":
    main()
