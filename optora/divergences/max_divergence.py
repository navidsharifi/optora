"""Max-divergence (Renyi divergence of order infinity) between distributions."""

import torch

from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype


class MaxDivergence(Divergence):
    r"""Max-divergence of a candidate distribution from a reference.

    For discrete distributions represented as nonnegative tensors that sum to
    one along their last dimension, computes the logarithm of the largest
    likelihood ratio,

    $$
    D_\infty(p \,\|\, q) = \log \max_i \frac{p_i}{q_i},
    $$

    the Renyi divergence of order $\infty$ (van Erven and Harremoes 2014),
    also called the max-divergence (Dwork and Roth 2014). It is infinite
    when `p` puts mass where `q` has none. Its ball of radius
    $\log(1/\alpha)$ is exactly the set of distributions with likelihood
    ratio at most $1/\alpha$, which `optora.dro.phi_dro.CVaRAmbiguitySet`
    uses to express conditional value-at-risk as an ambiguity set.

    Attributes:
        eps: Small positive constant used to clamp `q` away from zero before
            dividing, so a candidate leaving the reference's support yields
            a large finite value rather than `inf` or `nan`.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize the max-divergence.

        Args:
            eps: Small positive constant used to clamp `q` away from zero
                before dividing.

        Raises:
            ValueError: If `eps` is not positive.
        """
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the max-divergence of `p` from `q`.

        Args:
            p: Candidate distribution, a nonnegative tensor that sums to one
                along its last dimension.
            q: Reference distribution with the same shape as `p`.

        Returns:
            A scalar tensor holding $D_\infty(p \,\|\, q)$, clamped to be
            nonnegative to absorb floating-point error near zero, computed
            in `optora.core.working_dtype(p, q)` and returned in the
            promoted dtype of `p` and `q`.
        """
        dtype = working_dtype(p, q)
        q_safe = torch.clamp(q.to(dtype), min=self.eps)
        divergence = torch.log(torch.amax(p.to(dtype) / q_safe))
        return torch.clamp(divergence, min=0.0).to(
            torch.promote_types(p.dtype, q.dtype)
        )
