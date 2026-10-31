"""The ambiguity-set families the figures compare, with their saturation radii.

Every family is built from the library's own classes; the only extra
knowledge here is the radius at which each ball first contains the point
mass on the highest-loss scenario, so that different geometries can be
shown at the same *fraction* of their own saturation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from optora.core import AmbiguitySet
from optora.divergences import EntropicTransportDivergence, MMDDivergence
from optora.dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    KLAmbiguitySet,
    MMDAmbiguitySet,
    SinkhornAmbiguitySet,
    TotalVariationAmbiguitySet,
    WassersteinAmbiguitySet,
)
from tools.figures.problems import Instance

Radius = float | torch.Tensor

EPSILON = 0.08
BANDWIDTH = 0.35


def line_transport_cost(
    q: torch.Tensor, p: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Return the type-1 Wasserstein distance between `q` and `p` on a line.

    With the ground cost `|x_i - x_j|` on sorted support points the optimal
    transport cost is the area between the two cumulative distributions.

    Args:
        q: Candidate distributions of shape `(..., n)`.
        p: Reference distribution of shape `(n,)`.
        points: Sorted support points of shape `(n,)`.

    Returns:
        The distances, of shape `(...)`.
    """
    gaps = points[1:] - points[:-1]
    cumulative_gap = torch.cumsum(q - p, dim=-1)[..., :-1]
    return (cumulative_gap.abs() * gaps).sum(dim=-1)


def point_mass_on_maximum(instance: Instance) -> torch.Tensor:
    """Return the distribution concentrated on the highest-loss scenario."""
    mass = torch.zeros_like(instance.nominal)
    mass[int(torch.argmax(instance.loss))] = 1.0
    return mass


@dataclass(frozen=True, slots=True)
class SetFamily:
    """One ambiguity-set geometry on a finite support.

    Attributes:
        key: Palette key, see `tools.figures._base.SET_SLOTS`.
        build: Constructor taking an instance and a radius (a float or a
            tensor broadcasting against the loss batch).
        divergence: The family's divergence of a batch of candidates of
            shape `(..., n)` from the instance's nominal.
        saturation: Radius at which the ball first contains the point mass
            on the highest-loss scenario, so the worst case is `max(loss)`.
    """

    key: str
    build: Callable[[Instance, Radius], AmbiguitySet]
    divergence: Callable[[Instance, torch.Tensor], torch.Tensor]
    saturation: Callable[[Instance], float]


def _top_mass(instance: Instance) -> float:
    return float(instance.nominal[instance.loss == instance.loss.max()].sum())


def _broadcast(instance: Instance, q: torch.Tensor) -> torch.Tensor:
    return instance.nominal.expand_as(q)


# The library's KL, chi-square, total-variation and max divergences reduce a
# whole tensor to one scalar; the simplex grids need one value per candidate,
# so these are their closed forms over the last dimension.
def _kl(instance: Instance, q: torch.Tensor) -> torch.Tensor:
    return torch.special.xlogy(q, q / instance.nominal).sum(dim=-1)


def _chi_square(instance: Instance, q: torch.Tensor) -> torch.Tensor:
    return ((q - instance.nominal) ** 2 / instance.nominal).sum(dim=-1)


def _total_variation(instance: Instance, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (q - instance.nominal).abs().sum(dim=-1)


def _max_divergence(instance: Instance, q: torch.Tensor) -> torch.Tensor:
    return torch.log((q / instance.nominal).max(dim=-1).values)


FAMILIES: tuple[SetFamily, ...] = (
    SetFamily(
        key="kl",
        build=lambda instance, radius: KLAmbiguitySet(instance.nominal, radius),
        divergence=_kl,
        saturation=lambda instance: (
            -float(torch.log(torch.tensor(_top_mass(instance))))
        ),
    ),
    SetFamily(
        key="chi_square",
        build=lambda instance, radius: ChiSquareAmbiguitySet(instance.nominal, radius),
        divergence=_chi_square,
        saturation=lambda instance: 1.0 / _top_mass(instance) - 1.0,
    ),
    SetFamily(
        key="total_variation",
        build=lambda instance, radius: TotalVariationAmbiguitySet(
            instance.nominal, radius
        ),
        divergence=_total_variation,
        saturation=lambda instance: 1.0 - _top_mass(instance),
    ),
    SetFamily(
        key="cvar",
        build=lambda instance, radius: CVaRAmbiguitySet(
            instance.nominal,
            torch.exp(-torch.as_tensor(radius, dtype=instance.nominal.dtype)),
        ),
        divergence=_max_divergence,
        saturation=lambda instance: (
            -float(torch.log(torch.tensor(_top_mass(instance))))
        ),
    ),
    SetFamily(
        key="wasserstein",
        build=lambda instance, radius: WassersteinAmbiguitySet(
            instance.nominal, instance.cost, radius
        ),
        divergence=lambda instance, q: line_transport_cost(
            q, instance.nominal, instance.points
        ),
        saturation=lambda instance: float(
            line_transport_cost(
                point_mass_on_maximum(instance), instance.nominal, instance.points
            )
        ),
    ),
    SetFamily(
        key="sinkhorn",
        build=lambda instance, radius: SinkhornAmbiguitySet(
            instance.nominal, instance.cost, EPSILON, radius
        ),
        divergence=lambda instance, q: EntropicTransportDivergence(
            instance.cost, EPSILON
        )(q, _broadcast(instance, q)),
        saturation=lambda instance: float(
            EntropicTransportDivergence(instance.cost, EPSILON)(
                point_mass_on_maximum(instance), instance.nominal
            )
        ),
    ),
    SetFamily(
        key="mmd",
        build=lambda instance, radius: MMDAmbiguitySet(
            instance.nominal, instance.gram(BANDWIDTH), radius
        ),
        divergence=lambda instance, q: MMDDivergence(instance.gram(BANDWIDTH))(
            q, _broadcast(instance, q)
        ),
        saturation=lambda instance: float(
            MMDDivergence(instance.gram(BANDWIDTH))(
                point_mass_on_maximum(instance), instance.nominal
            )
        ),
    ),
)

FAMILY_BY_KEY: dict[str, SetFamily] = {family.key: family for family in FAMILIES}


def budget_schedule(frames: int, start: float = 0.02) -> torch.Tensor:
    """Return `frames` budget fractions easing from `start` to one."""
    steps = torch.linspace(0.0, 1.0, frames, dtype=torch.float64) ** 1.6
    return start + (1.0 - start) * steps
