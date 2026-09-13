"""Shared contract for ambiguity sets used by DRO formulations."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from optora.core.divergence_base import Divergence


class AmbiguitySet(nn.Module, ABC):
    """Set of distributions within a bounded divergence of a nominal distribution.

    An ambiguity set pairs a `Divergence` with a radius: every distribution
    `q` inside the set satisfies `divergence(q, nominal) <= radius`. Modules
    in `optora.dro` subclass `AmbiguitySet` to implement the inner
    maximization of the DRO minimax problem for a specific divergence
    geometry (for example KL, a general phi-divergence, or Wasserstein).

    Inherits from `torch.nn.Module` (rather than a plain ABC) so `nominal`
    is registered as a buffer and `divergence` as a submodule: a single
    `.to(device)`/`.cuda()` call then moves the nominal distribution and any
    tensor state the divergence holds (for example `SinkhornDivergence`'s
    ground-cost matrix) together, and both surface through `state_dict()`.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on, a
            finite, nonnegative tensor that sums to one along its last
            dimension within an absolute tolerance of `1e-6`.
        divergence: Divergence used to measure distance from `nominal`.
        radius: Nonnegative scalar bounding the divergence of any
            distribution inside the ambiguity set from `nominal`.
    """

    nominal: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        divergence: Divergence,
        radius: float,
        validate: bool = False,
    ) -> None:
        """Initialize the ambiguity set.

        With `validate=True`, the nominal tensor is checked without
        normalizing or detaching it. Validation reads tensor values on the
        host and may synchronize the device, so it is off by default.

        Args:
            nominal: Reference distribution the ambiguity set is centered
                on. Entries must be finite and nonnegative, and each sum
                along the last dimension must be within `1e-6` of one
                (absolute tolerance only).
            divergence: Divergence used to measure distance from `nominal`.
            radius: Nonnegative scalar bounding the divergence of any
                distribution inside the ambiguity set from `nominal`.
            validate: Whether to check the nominal probability values.
                Defaults to `False` to keep construction asynchronous.

        Raises:
            ValueError: If `radius` is negative, or if `validate` is set and
                `nominal` is invalid within the stated tolerance.
        """
        super().__init__()
        if radius < 0:
            raise ValueError(f"radius must be nonnegative, got {radius}.")
        if validate:
            if not torch.all(torch.isfinite(nominal) & (nominal >= 0)):
                raise ValueError(
                    "nominal must contain only finite, nonnegative entries."
                )
            mass = nominal.sum(dim=-1)
            if not torch.allclose(mass, torch.ones_like(mass), atol=1e-6, rtol=0.0):
                raise ValueError(
                    "nominal must sum to one along its last dimension "
                    "within an absolute tolerance of 1e-6."
                )
        self.register_buffer("nominal", nominal)
        self.divergence = divergence
        self.radius = radius

    def contains(self, candidate: torch.Tensor) -> torch.Tensor:
        """Check whether a candidate distribution lies inside the ambiguity set.

        The answer is returned as a boolean tensor on `candidate`'s device
        rather than as a Python `bool`, so membership can be used as a mask
        or composed with further tensor work without forcing a
        device-to-host synchronization. Call `bool(...)` on the result only
        where a host-side branch is genuinely needed.

        Args:
            candidate: Candidate distribution with the same shape as
                `nominal`.

        Returns:
            A boolean tensor that is `True` where the divergence of
            `candidate` from `nominal` does not exceed `radius`.
        """
        divergence: torch.Tensor = self.divergence(candidate, self.nominal)
        return divergence <= self.radius

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
