"""Convergence tracking for iterative loops without per-iteration host syncs."""

import torch

DEFAULT_CHECK_INTERVAL = 10


class ConvergenceTracker:
    """Convergence state of an iterative loop, held on the iterate's device.

    A textbook iterative method tests `residual < tol` with a Python `if`,
    which forces a device-to-host copy of the residual on every iteration.
    Optora composes such loops (an `optora.dro` ambiguity set's inner dual
    solve runs inside a `MinimaxSolver` outer solve, and a Wasserstein
    ambiguity set additionally runs Sinkhorn iterations), so a single solve
    would issue thousands of accelerator stalls.

    `ConvergenceTracker` keeps the convergence flag and the iteration count
    as tensors on the iterate's device, updates them with ordinary
    elementwise kernels, and copies them to the host at most once every
    `check_interval` iterations. Callers must freeze their iterates with
    `torch.where(tracker.converged, ...)`, which makes the iterations that
    run between two host reads exact no-ops: the returned solution is
    identical to one produced by testing convergence every iteration,
    whatever `check_interval` is. Larger values trade those redundant
    frozen iterations for fewer synchronizations.

    Attributes:
        converged: Zero-dimensional boolean tensor, `True` once the
            residual has fallen below the tolerance. Use it as the
            predicate of the `torch.where` that freezes the loop's
            iterates.
        check_interval: Number of iterations between host reads of
            `converged`.
    """

    def __init__(
        self,
        tol: float,
        check_interval: int,
        reference: torch.Tensor,
    ) -> None:
        """Initialize the tracker on the device and dtype of `reference`.

        Args:
            tol: Positive tolerance the residual is compared against.
            check_interval: Number of iterations between host reads of the
                convergence flag.
            reference: Iterate whose dtype and device the residual
                comparison is performed in.
        """
        self.check_interval = check_interval
        self._tol = torch.as_tensor(tol, dtype=reference.dtype, device=reference.device)
        self.converged = torch.zeros((), dtype=torch.bool, device=reference.device)
        self._num_iterations = torch.zeros(
            (), dtype=torch.long, device=reference.device
        )

    def update(self, residual: torch.Tensor) -> torch.Tensor:
        """Record one iteration's residual without synchronizing.

        Iterations performed after convergence do not advance the iteration
        count, so the reported count matches what a loop that exited the
        moment the residual dropped below the tolerance would report.

        Args:
            residual: Zero-dimensional nonnegative convergence residual,
                for example a gradient norm or the change in a dual
                potential.

        Returns:
            The updated `converged` flag, to be used as the predicate of a
            `torch.where` that freezes this iteration's update.
        """
        self._num_iterations = self._num_iterations + ~self.converged
        self.converged = self.converged | (residual.detach() < self._tol)
        return self.converged

    def should_stop(self, iteration: int) -> bool:
        """Check whether the loop may exit, synchronizing at most periodically.

        Args:
            iteration: Zero-based index of the iteration that just ran.

        Returns:
            `True` if this iteration is a checkpoint and the loop has
            converged, `False` otherwise.
        """
        if (iteration + 1) % self.check_interval != 0:
            return False
        return bool(self.converged)

    def to_host(self) -> tuple[bool, int]:
        """Copy the final convergence diagnostics back to the host.

        This is the one deliberate synchronization per solve, paid when the
        loop is already over.

        Returns:
            A tuple of the convergence flag and the number of iterations
            performed before convergence.
        """
        return bool(self.converged), int(self._num_iterations)


def validate_check_interval(check_interval: int) -> int:
    """Validate a `ConvergenceTracker` checkpoint interval.

    Args:
        check_interval: Number of iterations between host reads of the
            convergence flag.

    Returns:
        `check_interval`, guaranteed positive.

    Raises:
        ValueError: If `check_interval` is not positive.
    """
    if check_interval <= 0:
        raise ValueError(f"check_interval must be positive, got {check_interval}.")
    return check_interval
