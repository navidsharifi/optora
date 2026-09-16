"""Convergence tracking for iterative loops without per-iteration host syncs."""

from functools import cached_property

import torch

DEFAULT_CHECK_INTERVAL = 10


class ConvergenceStatus:
    """Convergence diagnostics of a finished solve, materialized on demand.

    A loop produces its convergence flag and iteration count on the
    iterate's device, so reporting them as a Python `bool` and `int` costs
    two synchronizations per solve: exactly the cost `ConvergenceTracker`
    removes from the loop, paid again once the loop is over. Under nesting
    that becomes the dominant cost, since every outer iteration of an
    `optora.dro.MinimaxSolver` runs an inner dual solve whose diagnostics
    nothing ever reads.

    `ConvergenceStatus` therefore keeps the diagnostics as the tensors the
    loop computed and converts them on first access, caching each
    conversion so repeated reads cost one synchronization at most. A solve
    whose diagnostics are never read never synchronizes for them.
    """

    def __init__(self, converged: torch.Tensor, num_iterations: torch.Tensor) -> None:
        """Hold a finished loop's convergence state on its device.

        Args:
            converged: Zero-dimensional boolean tensor recording whether
                the loop's residual fell below its tolerance.
            num_iterations: Zero-dimensional integer tensor holding the
                number of iterations performed before convergence.
        """
        self._converged = converged
        self._num_iterations = num_iterations

    @cached_property
    def converged(self) -> bool:
        """Whether the loop's convergence criterion was met, read on the host."""
        return bool(self._converged)

    @cached_property
    def num_iterations(self) -> int:
        """Number of iterations the loop performed, read on the host."""
        return int(self._num_iterations)


class ConvergenceDiagnostics:
    """Host-side view of the convergence diagnostics of a solver result.

    Mixed into every solver result so `result.converged` and
    `result.num_iterations` read as plain Python values while the solve
    itself stays asynchronous: `status` holds them as device tensors until
    one of these properties is read.

    Attributes:
        status: Device-side convergence diagnostics of the solve, converted
            to host values only when read.
    """

    status: ConvergenceStatus

    @property
    def converged(self) -> bool:
        """Whether the convergence criterion was met before the iteration budget."""
        return self.status.converged

    @property
    def num_iterations(self) -> int:
        """Number of iterations actually performed."""
        return self.status.num_iterations


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

    That trade is only favourable when an iteration is cheap relative to a
    synchronization, which holds for the inner loops this was written for
    (a few elementwise kernels per iteration against a host read costing
    microseconds). Raise `check_interval` only in that regime. Loops whose
    single iteration is expensive — notably an outer solve whose objective
    is an `optora.dro` worst-case expectation, where one frozen iteration
    is an entire inner dual solve — should use `check_interval=1` and pay
    the synchronization instead.

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
        """Initialize the tracker on the device of `reference`.

        Args:
            tol: Positive tolerance the residual is compared against. Kept
                as a Python float: comparing a tensor against a Python
                scalar passes it to the comparison kernel directly, whereas
                materializing it as a tensor would copy it to the device on
                every solve.
            check_interval: Number of iterations between host reads of the
                convergence flag.
            reference: Iterate whose device the convergence state is held
                on.
        """
        self.check_interval = check_interval
        self._tol = tol
        self.converged = torch.zeros((), dtype=torch.bool, device=reference.device)
        self._num_iterations = torch.zeros(
            (), dtype=torch.long, device=reference.device
        )

    def start(self, residual: torch.Tensor) -> torch.Tensor:
        """Record the residual of the starting iterate, before any iteration.

        A loop that evaluates its residual at the current iterate and steps
        with what it learned there tests the starting iterate once before
        iterating. That test is not an iteration, so it latches convergence
        without advancing the iteration count: a loop that starts at its
        own solution reports zero iterations.

        Args:
            residual: Zero-dimensional nonnegative convergence residual at
                the starting iterate.

        Returns:
            The updated `converged` flag, to be used as the predicate of a
            `torch.where` that freezes the first iteration's update.
        """
        self.converged = self.converged | (residual.detach() < self._tol)
        return self.converged

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
        return self.start(residual)

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

    def status(self) -> ConvergenceStatus:
        """Hand the loop's final diagnostics over without synchronizing.

        Returns:
            A `ConvergenceStatus` wrapping the convergence flag and the
            number of iterations performed before convergence, still as
            device tensors. Converting them to host values is deferred to
            whoever reads them, so an inner solve nobody inspects costs no
            synchronization at all.
        """
        return ConvergenceStatus(self.converged, self._num_iterations)


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
