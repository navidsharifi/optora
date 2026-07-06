"""Optimization result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

Tensor = torch.Tensor


@dataclass(slots=True)
class OptimizationResult:
    """Result returned by Optora solvers.

    Attributes:
        x: Final iterate.
        fun: Objective value at ``x`` as a scalar tensor.
        jac: Gradient at ``x`` when available.
        nit: Number of solver iterations.
        nfev: Number of objective evaluations.
        njev: Number of gradient evaluations.
        success: Whether the solver reached its convergence criterion.
        status: Solver-specific integer status code.
        message: Human-readable termination message.
        history: Optional per-iteration diagnostics.
    """

    x: Tensor
    fun: Tensor
    jac: Tensor | None
    nit: int
    nfev: int
    njev: int
    success: bool
    status: int
    message: str
    history: list[dict[str, Any]] = field(default_factory=list)
