"""KL-DRO worst-case expectation across a growing ambiguity radius.

The script isolates the core DRO primitive. For a fixed nominal
distribution and a fixed per-scenario loss, it measures how the
worst-case expected loss

    sup_{q: D_KL(q || nominal) <= radius} E_q[loss]

grows with the radius of the KL ball. Two closed-form limits bound the
curve, and the script checks both numerically.

- At radius 0 the ball collapses to `{nominal}`, so the worst case is
  exactly the nominal expectation `sum(nominal * loss)`.
- As the radius grows, the ball eventually contains the point mass on
  the highest-loss scenario, so the worst case saturates at
  `loss.max()`. This happens at a finite radius. From
  `radius = -log(nominal mass of that scenario)` on, the worst case is
  `loss.max()`. `KLAmbiguitySet` reaches it because it bisects a bounded
  parameterization of the tilt whose upper end is that point mass. A
  solve of the dual would instead have to approach its infimum as
  `eta -> 0`.

Run:
    python tutorials/01_kl_dro_radius_sweep.py
"""

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import KLAmbiguitySet

NOMINAL = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
LOSS = torch.tensor([0.5, 1.5, 3.0, 8.0], dtype=torch.float64)


def main() -> None:
    nominal_expectation = torch.sum(NOMINAL * LOSS).item()
    exact_zero_radius = KLAmbiguitySet(nominal=NOMINAL, radius=0.0)
    zero_radius_value = exact_zero_radius.worst_case_expectation(LOSS).item()
    print(f"E_nominal[loss]                    = {nominal_expectation:.6f}")
    print(f"worst_case_expectation(radius=0.0) = {zero_radius_value:.6f}")
    assert abs(nominal_expectation - zero_radius_value) < 1e-12

    radii = torch.logspace(-3, 1, steps=25, dtype=torch.float64)
    sweep = KLAmbiguitySet(nominal=NOMINAL, radius=radii)
    worst_case = sweep.worst_case_expectation(LOSS)
    worst_possible = LOSS.max().item()
    print(f"worst case at radius={radii[0].item():.4f}  -> {worst_case[0].item():.6f}")
    print(f"worst case at radius={radii[-1].item():.4f} -> {worst_case[-1].item():.6f}")
    print(f"loss.max() (saturation limit)      = {worst_possible:.6f}")

    increments = torch.diff(worst_case)
    is_nondecreasing = bool(torch.all(increments >= -1e-9))
    print(f"monotonically nondecreasing in radius: {is_nondecreasing}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(radii, worst_case, marker="o", label="worst-case E_q[loss]")
    ax.axhline(
        nominal_expectation, color="gray", linestyle="--", label="E_nominal[loss]"
    )
    ax.axhline(worst_possible, color="firebrick", linestyle=":", label="max_i loss_i")
    ax.set_xscale("log")
    ax.set_xlabel("KL ambiguity radius")
    ax.set_ylabel("worst-case expected loss")
    ax.set_title("KL-DRO worst-case expectation vs. ambiguity radius")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "01_kl_dro_radius_sweep")


if __name__ == "__main__":
    main()
