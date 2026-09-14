"""Wasserstein-DRO as Lipschitz regularization.

Wasserstein-DRO is often justified by an equivalence: on a type-1
Wasserstein ball built from a ground metric, robustifying an expected loss
is the same as penalizing that loss's Lipschitz modulus. The linear
surrogate this suggests is

    sup_{q : W(q, nominal) <= radius} E_q[loss]  ~  E_nominal[loss]
                                                    + radius * Lip(loss)

which is exactly the "robustness = regularization" reading of the ball
radius: `radius` buys nothing but a penalty proportional to how fast the
loss can change per unit of transport.

On a finite support the identity is sharper than an approximation. The
exact discrete dual solved by `WassersteinAmbiguitySet` is minimized at
`gamma = L_n` for every sufficiently small radius, where

    L_n = max_{i != j} |loss_i - loss_j| / cost_ij

is the Lipschitz modulus of the loss *restricted to the sample*. The dual
value is then exactly `E_nominal[loss] + radius * L_n`, so the gap against
the surrogate built from the true constant `Lip(loss)` is

    gap(radius) = radius * (Lip(loss) - L_n) + (concavity correction)

The concavity correction vanishes as the radius shrinks; the first term
vanishes as the sample refines and `L_n -> Lip(loss)`. This script measures
both limits, using `softplus` as a loss whose Lipschitz constant is exactly
1 (its derivative is the sigmoid) but is only attained asymptotically, so
`L_n < 1` strictly at every finite sample size.

Reference:
    Qinyu Wu, Jonathan Yu-Meng Li, Tiantian Mao, "On Generalization and
    Regularization via Wasserstein Distributionally Robust Optimization",
    Management Science (2025). arXiv:2212.05716.

Run:
    python examples/05_wasserstein_lipschitz_equivalence.py
"""

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import WassersteinAmbiguitySet
from optora.solvers import GradientDescent

LIPSCHITZ_CONSTANT = 1.0
DUAL_SOLVER = GradientDescent(step_size=0.05, max_iter=800, tol=1e-12)
RADII = torch.logspace(-4.0, 0.5, steps=12, dtype=torch.float64)
RADIUS_SWEEP_SIZES = (16, 64, 256)
SAMPLE_SIZES = (8, 16, 32, 64, 128, 256)
SMALL_RADIUS = 1e-3


def empirical_measure(
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic `num_samples`-point discretization of N(0, 1).

    Support points are the midpoint quantiles of the standard normal, so
    the sample refines deterministically as `num_samples` grows and no seed
    is involved. The ground cost is the Euclidean metric on the support,
    which makes the ambiguity set a type-1 Wasserstein ball.

    Args:
        num_samples: Number of support points.

    Returns:
        A tuple `(nominal, loss, cost)` holding the uniform empirical
        distribution, the softplus loss evaluated on the support, and the
        pairwise ground cost matrix.
    """
    quantiles = (torch.arange(num_samples, dtype=torch.float64) + 0.5) / num_samples
    support = torch.special.ndtri(quantiles)
    loss = torch.nn.functional.softplus(support)
    cost = torch.abs(support.unsqueeze(-1) - support.unsqueeze(-2))
    nominal = torch.full_like(support, 1.0 / num_samples)
    return nominal, loss, cost


def empirical_lipschitz_constant(loss: torch.Tensor, cost: torch.Tensor) -> float:
    """Return the largest loss increment per unit of transport on the sample."""
    increments = torch.abs(loss.unsqueeze(-1) - loss.unsqueeze(-2))
    # The diagonal has a zero increment; the added identity only keeps 0/0 out.
    separation = cost + torch.eye(cost.shape[-1], dtype=cost.dtype)
    return torch.max(increments / separation).item()


def normalized_gap(
    nominal: torch.Tensor, loss: torch.Tensor, cost: torch.Tensor, radius: float
) -> float:
    """Return `(surrogate - exact dual value) / radius` at `radius`.

    Dividing by the radius removes the trivial part of the convergence: the
    absolute gap shrinks with the radius no matter what, whereas the
    normalized gap isolates the Lipschitz deficiency `Lip(loss) - L_n` that
    the small-radius limit should expose.
    """
    ambiguity_set = WassersteinAmbiguitySet(
        nominal=nominal, cost=cost, radius=radius, dual_solver=DUAL_SOLVER
    )
    exact = ambiguity_set.worst_case_expectation(loss).item()
    surrogate = torch.sum(nominal * loss).item() + radius * LIPSCHITZ_CONSTANT
    return (surrogate - exact) / radius


def main() -> None:
    fig, (ax_radius, ax_samples) = plt.subplots(1, 2, figsize=(11, 4))

    print("normalized gap (surrogate - exact) / radius, by sample size")
    for num_samples in RADIUS_SWEEP_SIZES:
        nominal, loss, cost = empirical_measure(num_samples)
        lipschitz = empirical_lipschitz_constant(loss, cost)
        gaps = [normalized_gap(nominal, loss, cost, radius.item()) for radius in RADII]
        deficiency = LIPSCHITZ_CONSTANT - lipschitz
        print(
            f"  n={num_samples:<4} L_n={lipschitz:.6f} "
            f"Lip - L_n={deficiency:.6f} "
            f"gap(radius->0)={gaps[0]:.6f} gap(radius={RADII[-1]:.3f})={gaps[-1]:.6f}"
        )
        assert min(gaps) > -1e-3
        assert abs(gaps[0] - deficiency) < 5e-3
        line = ax_radius.semilogx(RADII, gaps, marker="o", label=f"n = {num_samples}")
        ax_radius.axhline(deficiency, color=line[0].get_color(), linestyle=":")

    ax_radius.set_xlabel("Wasserstein radius")
    ax_radius.set_ylabel("(surrogate - exact) / radius")
    ax_radius.set_title("small radius: gap flattens at Lip - L_n")
    ax_radius.legend()

    print(f"\nsmall-radius limit at radius={SMALL_RADIUS}, by sample size")
    measured_gaps = []
    deficiencies = []
    for num_samples in SAMPLE_SIZES:
        nominal, loss, cost = empirical_measure(num_samples)
        lipschitz = empirical_lipschitz_constant(loss, cost)
        gap = normalized_gap(nominal, loss, cost, SMALL_RADIUS)
        measured_gaps.append(gap)
        deficiencies.append(LIPSCHITZ_CONSTANT - lipschitz)
        print(
            f"  n={num_samples:<4} L_n={lipschitz:.6f} "
            f"measured gap={gap:.6f} Lip - L_n={LIPSCHITZ_CONSTANT - lipschitz:.6f}"
        )

    decrements = torch.diff(torch.tensor(measured_gaps, dtype=torch.float64))
    is_decreasing = bool(torch.all(decrements < 0.0))
    print(f"gap decreases as the sample refines: {is_decreasing}")
    assert is_decreasing

    ax_samples.loglog(
        SAMPLE_SIZES, measured_gaps, marker="o", label="measured gap / radius"
    )
    ax_samples.loglog(
        SAMPLE_SIZES,
        deficiencies,
        marker="x",
        linestyle="--",
        label="Lip - L_n (predicted)",
    )
    ax_samples.set_xlabel("sample size n")
    ax_samples.set_ylabel("(surrogate - exact) / radius")
    ax_samples.set_title("growing sample: Lipschitz deficiency vanishes")
    ax_samples.legend()

    fig.tight_layout()
    save_figure(fig, "05_wasserstein_lipschitz_equivalence")


if __name__ == "__main__":
    main()
