"""The one toy problem every explanatory figure draws.

A single twelve-scenario instance keeps the formulation pages consistent: a
reader who has seen the nominal distribution and the loss once can follow
every geometry's worst case on the same support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Instance:
    """A finite-support DRO instance.

    Attributes:
        points: Support points on the unit interval, shape `(n,)`.
        nominal: Nominal distribution over the support, shape `(n,)`.
        loss: Per-scenario loss, shape `(n,)`, highest on the right tail.
        cost: Pairwise ground cost `|x_i - x_j|`, shape `(n, n)`.
    """

    points: torch.Tensor
    nominal: torch.Tensor
    loss: torch.Tensor
    cost: torch.Tensor

    @property
    def size(self) -> int:
        """Return the support size."""
        return int(self.points.shape[0])

    def gram(self, bandwidth: float = 0.35) -> torch.Tensor:
        """Return the Laplacian-kernel Gram matrix on the support."""
        return torch.exp(-self.cost / bandwidth)


def demo_instance(size: int = 12, dtype: torch.dtype = torch.float64) -> Instance:
    """Return the shared demo instance.

    The nominal concentrates on the middle of the support while the loss
    rises steeply on the right, so the scenarios that matter most to an
    adversary carry little nominal mass: the situation DRO is for.

    Args:
        size: Support size.
        dtype: Floating-point dtype of every tensor.

    Returns:
        The instance.
    """
    points = torch.linspace(0.0, 1.0, size, dtype=dtype)
    weights = torch.exp(-(((points - 0.38) / 0.26) ** 2)) + 0.06
    nominal = weights / weights.sum()
    loss = (
        0.35
        + 1.6 * torch.sigmoid((points - 0.72) / 0.09)
        + 0.12 * torch.sin(9.0 * points)
    )
    cost = torch.abs(points[:, None] - points[None, :])
    return Instance(points=points, nominal=nominal, loss=loss, cost=cost)


def random_instance(
    size: int, seed: int = 0, dtype: torch.dtype = torch.float64
) -> Instance:
    """Return a random instance of support `size` on the unit interval.

    The support is an even grid rather than random points: random points
    can fall within $10^{-5}$ of each other, which makes a kernel Gram matrix
    on them numerically singular and the transport cost degenerate, so the
    benchmarks and the dtype comparison would measure that conditioning
    instead of the solvers.

    Args:
        size: Support size.
        seed: Generator seed.
        dtype: Floating-point dtype of every tensor.

    Returns:
        The instance: evenly spaced support points, a random nominal with
        every scenario reachable, a standard normal loss, and the line cost.
    """
    generator = torch.Generator().manual_seed(seed)
    points = torch.linspace(0.0, 1.0, size, dtype=dtype)
    weights = torch.rand(size, generator=generator, dtype=dtype) + 0.5
    nominal = weights / weights.sum()
    loss = torch.randn(size, generator=generator, dtype=dtype)
    cost = torch.abs(points[:, None] - points[None, :])
    return Instance(points=points, nominal=nominal, loss=loss, cost=cost)


def cast(instance: Instance, dtype: torch.dtype) -> Instance:
    """Return `instance` with every tensor cast to `dtype`."""
    return Instance(
        points=instance.points.to(dtype),
        nominal=instance.nominal.to(dtype),
        loss=instance.loss.to(dtype),
        cost=instance.cost.to(dtype),
    )
