"""Shared contract for divergences used to define DRO ambiguity sets."""

from abc import ABC, abstractmethod

import torch


class Divergence(ABC):
    """Nonnegative discrepancy between two probability distributions.

    Subclasses implement a specific divergence (for example
    Kullback-Leibler, a general phi-divergence, or an entropy-regularized
    Wasserstein discrepancy) that `optora.dro` ambiguity sets use to bound
    how far a candidate distribution may lie from a nominal distribution.
    """

    @abstractmethod
    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute the divergence of `p` from `q`.

        Args:
            p: Candidate distribution, a nonnegative tensor that sums to one
                along its last dimension.
            q: Reference distribution with the same shape as `p`.

        Returns:
            A scalar tensor holding the divergence value. Implementations
            must return zero when `p` equals `q` and a nonnegative value
            otherwise.
        """
        raise NotImplementedError
