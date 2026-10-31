"""Kullback-Leibler divergence between discrete probability distributions."""

import torch

from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype


class KLDivergence(Divergence):
    r"""Kullback-Leibler divergence of a candidate distribution from a reference.

    For discrete distributions represented as nonnegative tensors that sum to
    one along their last dimension, computes

    $$
    D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.
    $$

    `optora.dro.kl_dro` uses this divergence to define KL-based ambiguity
    sets.

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
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the KL divergence of `p` from `q`.

        Args:
            p: Candidate distribution of shape `(..., n)`, nonnegative and
                summing to one along its last dimension.
            q: Reference distribution of shape `(..., n)`, broadcasting
                against `p`.

        Returns:
            $D_{\mathrm{KL}}(p \,\|\, q)$ over the broadcast batch, of shape
            `(...)`, clamped to be nonnegative to absorb floating-point
            error near zero, computed in `optora.core.working_dtype(p, q)`
            and returned in the promoted dtype of `p` and `q`.
        """
        dtype = working_dtype(p, q)
        p_safe = torch.clamp(p.to(dtype), min=self.eps)
        q_safe = torch.clamp(q.to(dtype), min=self.eps)
        divergence = torch.sum(p_safe * torch.log(p_safe / q_safe), dim=-1)
        return torch.clamp(divergence, min=0.0).to(
            torch.promote_types(p.dtype, q.dtype)
        )
