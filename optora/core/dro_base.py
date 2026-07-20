"""Shared contract for ambiguity sets used by DRO formulations."""

from abc import ABC, abstractmethod

import torch

from optora.core.divergence_base import Divergence


class AmbiguitySet(ABC):
    """Set of distributions within a bounded divergence of a nominal distribution.

    An ambiguity set pairs a `Divergence` with a radius: every distribution
    `q` inside the set satisfies `divergence(q, nominal) <= radius`. Modules
    in `optora.dro` subclass `AmbiguitySet` to implement the inner
    maximization of the DRO minimax problem for a specific divergence
    geometry (for example KL, a general phi-divergence, or Wasserstein).

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on, a
            nonnegative tensor that sums to one along its last dimension.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative scalar bounding the divergence of any
            distribution inside the ambiguity set from `nominal`.
    """

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float,
    ) -> None:
        """Initialize the ambiguity set.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on.
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative scalar bounding the divergence of any
                distribution inside the ambiguity set from `nominal`.

        Raises:
            ValueError: If `radius` is negative.
        """
        if radius < 0:
            raise ValueError(f"radius must be nonnegative, got {radius}.")
        self.nominal = nominal
        self.divergence = divergence
        self.radius = radius

    def contains(self, candidate: torch.Tensor) -> bool:
        """Check whether a candidate distribution lies inside the ambiguity set.

        Args:
            candidate: Candidate distribution with the same shape as
                `nominal`.

        Returns:
            `True` if the divergence of `candidate` from `nominal` does not
            exceed `radius`, `False` otherwise.
        """
        return bool(self.divergence(candidate, self.nominal) <= self.radius)

    @abstractmethod
    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the ambiguity set.

        Args:
            loss: Per-scenario loss values, one entry per element of
                `nominal`'s support.

        Returns:
            A scalar tensor holding the worst-case expected loss attainable
            by any distribution inside the ambiguity set.
        """
        raise NotImplementedError
