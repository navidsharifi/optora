"""Kullback-Leibler divergence between discrete probability distributions."""

import torch

from optora.core.divergence_base import Divergence


class KLDivergence(Divergence):
    r"""Kullback-Leibler divergence of a candidate distribution from a reference.

    For discrete distributions represented as nonnegative tensors that sum to
    one along their last dimension, computes
    `D_KL(p || q) = sum_i p_i * log(p_i / q_i)`. `optora.dro.kl_dro` uses this
    divergence to define KL-based ambiguity sets.

    Attributes:
        eps: Small positive constant used to clamp `p` and `q` away from zero
            before taking the logarithm, avoiding division by zero and
            `log(0)` without branching on masked entries.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize the KL divergence.

        Args:
            eps: Small positive constant used to clamp `p` and `q` away from
                zero before taking the logarithm.

        Raises:
            ValueError: If `eps` is not positive.
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute the KL divergence of `p` from `q`.

        Args:
            p: Candidate distribution, a nonnegative tensor that sums to one
                along its last dimension.
            q: Reference distribution with the same shape as `p`.

        Returns:
            A scalar tensor holding `D_KL(p || q)`, clamped to be
            nonnegative to absorb floating-point error near zero.
        """
        p_safe = torch.clamp(p, min=self.eps)
        q_safe = torch.clamp(q, min=self.eps)
        divergence = torch.sum(p_safe * torch.log(p_safe / q_safe))
        return torch.clamp(divergence, min=0.0)
