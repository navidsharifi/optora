"""Kernel maximum mean discrepancy between distributions on a finite support."""

import torch

from optora.core.divergence_base import Divergence
from optora.core.precision import working_dtype


class MMDDivergence(Divergence):
    r"""Maximum mean discrepancy of a candidate distribution from a reference.

    For distributions `p` and `q` on a shared finite support
    $\{x_1, \dots, x_n\}$ and a positive semidefinite kernel $k$ with Gram
    matrix $K_{ij} = k(x_i, x_j)$, the maximum mean discrepancy is the
    distance between their kernel mean embeddings in the reproducing kernel
    Hilbert space $\mathcal{H}$ of $k$ (Gretton et al. 2012),

    $$
    \mathrm{MMD}_k(p, q)
        = \Big\lVert \sum_i (p_i - q_i)\, k(x_i, \cdot) \Big\rVert_{\mathcal{H}}
        = \sqrt{(p - q)^\top K (p - q)}.
    $$

    It is symmetric and satisfies the triangle inequality, and it is a
    metric on distributions over the support whenever $K$ is positive
    definite, as it is for a characteristic kernel such as the Laplacian
    $e^{-\lVert x - y \rVert}$ on distinct points. Unlike a
    $\varphi$-divergence it stays finite between distributions with
    different supports. `optora.dro.MMDAmbiguitySet` bounds it by its
    radius.

    Attributes:
        gram: Symmetric positive semidefinite Gram matrix of shape `(n, n)`.
    """

    gram: torch.Tensor

    def __init__(self, gram: torch.Tensor, validate: bool = False) -> None:
        """Initialize the maximum mean discrepancy.

        Args:
            gram: Symmetric positive semidefinite Gram matrix of shape
                `(n, n)` of the kernel on the support points.
            validate: Whether to check that `gram` is finite, symmetric, and
                positive semidefinite up to rounding. The check reads an
                eigenvalue decomposition on the host, so it is opt-in.

        Raises:
            ValueError: If `gram` is not a square 2D tensor, or `validate` is
                set and `gram` is not a finite symmetric positive
                semidefinite matrix.
        """
        super().__init__()
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise ValueError(
                f"gram must be a square 2D tensor, got shape {tuple(gram.shape)}."
            )
        if validate:
            _validate_gram(gram)
        self.register_buffer("gram", gram)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        r"""Compute the maximum mean discrepancy between `p` and `q`.

        Args:
            p: Candidate distribution of shape `(..., n)`.
            q: Reference distribution of shape `(..., n)`, broadcasting
                against `p`.

        Returns:
            $\mathrm{MMD}_k(p, q)$ over the broadcast batch, computed in
            `optora.core.working_dtype(p, q, gram)` and returned in the
            promoted dtype of `p` and `q`. The quadratic form is clamped at
            zero before the square root, so rounding near `p == q` never
            produces `nan`.

        Raises:
            ValueError: If the last dimension of `p` or `q` does not match
                `gram`.
        """
        size = self.gram.shape[0]
        for name, tensor in (("p", p), ("q", q)):
            if tensor.shape[-1] != size:
                raise ValueError(
                    f"{name} must have shape (..., {size}) to index gram, "
                    f"got {tuple(tensor.shape)}."
                )
        dtype = working_dtype(p, q, self.gram)
        difference = p.to(dtype) - q.to(dtype)
        squared = torch.sum(difference * (difference @ self.gram.to(dtype)), dim=-1)
        return torch.sqrt(torch.clamp(squared, min=0.0)).to(
            torch.promote_types(p.dtype, q.dtype)
        )


def _validate_gram(gram: torch.Tensor) -> None:
    """Raise unless `gram` is finite, symmetric, and positive semidefinite.

    Symmetry and the smallest eigenvalue are compared against a tolerance of
    `n` ulps of the largest absolute entry, the rounding of a Gram matrix
    evaluated in `gram`'s dtype.

    Args:
        gram: Square matrix to check.

    Raises:
        ValueError: If `gram` fails any of the checks.
    """
    working = gram.to(torch.promote_types(gram.dtype, torch.float32))
    if not bool(torch.all(torch.isfinite(working))):
        raise ValueError("gram must be finite.")
    tolerance = (
        working.shape[0]
        * torch.finfo(working.dtype).eps
        * float(torch.amax(torch.abs(working)))
    )
    if float(torch.amax(torch.abs(working - working.mT))) > tolerance:
        raise ValueError("gram must be symmetric.")
    if float(torch.linalg.eigvalsh(working)[0]) < -tolerance:
        raise ValueError("gram must be positive semidefinite.")
