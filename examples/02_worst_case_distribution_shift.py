"""Tracking convergence across probability space: the TV-DRO worst-case
candidate distribution as the ambiguity radius grows.

`TotalVariationAmbiguitySet.worst_case_expectation` only returns the scalar
`E_q*[loss]` at the worst-case candidate distribution `q*`, not `q*`
itself -- the `AmbiguitySet` API is deliberately narrow (see
`optora/core/dro_base.py`). This example reconstructs `q*` directly from
the closed-form combinatorial solution the library documents internally
(sort scenarios by loss, then reallocate probability mass from the
cheapest scenarios to the single most expensive one, up to the radius
budget -- see `optora/dro/phi_dro.py` and `progress/decisions.md`), then
cross-checks `E_q*[loss]` against the library's own `worst_case_expectation`
to confirm the reconstruction is exact, not just plausible.

Run:
    python examples/02_worst_case_distribution_shift.py
"""

import matplotlib.pyplot as plt
import torch

from optora.dro import TotalVariationAmbiguitySet

from _plotting import save_figure

NOMINAL = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
LOSS = torch.tensor([0.5, 1.5, 3.0, 8.0], dtype=torch.float64)


def worst_case_distribution(
    nominal: torch.Tensor, loss: torch.Tensor, radius: float
) -> torch.Tensor:
    """Reconstruct the TV-DRO worst-case candidate distribution `q*`."""
    order = torch.argsort(loss)
    rest_idx, worst_idx = order[:-1], order[-1]
    rest_nominal = nominal[rest_idx]

    mass_available_before = torch.cumsum(rest_nominal, dim=0) - rest_nominal
    remaining_budget = torch.clamp(radius - mass_available_before, min=0.0)
    moved = torch.minimum(remaining_budget, rest_nominal)

    q = nominal.clone()
    q[rest_idx] = nominal[rest_idx] - moved
    q[worst_idx] = nominal[worst_idx] + moved.sum()
    return q


def main() -> None:
    max_movable_mass = (torch.sum(NOMINAL) - NOMINAL[torch.argmax(LOSS)]).item()
    radii = torch.linspace(0.0, max_movable_mass, steps=40, dtype=torch.float64)

    reconstructed = torch.stack(
        [worst_case_distribution(NOMINAL, LOSS, radius.item()) for radius in radii]
    )
    library_values = torch.stack(
        [
            TotalVariationAmbiguitySet(nominal=NOMINAL, radius=radius.item())
            .worst_case_expectation(LOSS)
            .detach()
            for radius in radii
        ]
    )
    reconstructed_values = reconstructed @ LOSS
    max_abs_diff = torch.max(torch.abs(reconstructed_values - library_values)).item()
    print(f"max |reconstructed E_q*[loss] - library worst_case_expectation| = {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-9

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(
        radii,
        *reconstructed.T,
        labels=[f"scenario {i} (loss={value:.1f})" for i, value in enumerate(LOSS)],
    )
    ax.set_xlabel("total-variation ambiguity radius")
    ax.set_ylabel("candidate probability mass q*_i(radius)")
    ax.set_title("worst-case distribution shift as the TV ball grows")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save_figure(fig, "02_worst_case_distribution_shift")


if __name__ == "__main__":
    main()
