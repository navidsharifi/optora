"""Tests for `ConvergenceTracker` and `validate_check_interval`."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from optora.core.convergence import ConvergenceTracker, validate_check_interval


def test_validate_check_interval_accepts_positive_values() -> None:
    assert validate_check_interval(7) == 7


@pytest.mark.parametrize("check_interval", [0, -1])
def test_validate_check_interval_rejects_non_positive_values(
    check_interval: int,
) -> None:
    with pytest.raises(ValueError, match="check_interval must be positive"):
        validate_check_interval(check_interval)


def test_tracker_state_lives_on_the_reference_device() -> None:
    reference = torch.zeros(3, dtype=torch.float64)
    tracker = ConvergenceTracker(tol=1e-6, check_interval=1, reference=reference)

    assert tracker.converged.dtype == torch.bool
    assert tracker.converged.device == reference.device
    assert not tracker.converged


def test_update_returns_a_device_side_flag_without_synchronizing(
    host_sync_counter: Callable[[], Any],
) -> None:
    tracker = ConvergenceTracker(
        tol=0.5, check_interval=1000, reference=torch.zeros(())
    )

    with host_sync_counter() as syncs:
        first = tracker.update(torch.tensor(1.0))
        second = tracker.update(torch.tensor(0.1))

    assert syncs == []
    assert isinstance(first, torch.Tensor)
    assert not first
    assert second


def test_convergence_is_latched_once_the_residual_drops() -> None:
    tracker = ConvergenceTracker(tol=0.5, check_interval=1, reference=torch.zeros(()))

    tracker.update(torch.tensor(0.1))
    tracker.update(torch.tensor(10.0))

    assert tracker.converged


def test_iteration_count_stops_advancing_after_convergence() -> None:
    tracker = ConvergenceTracker(
        tol=0.5, check_interval=1000, reference=torch.zeros(())
    )

    for residual in (1.0, 1.0, 0.1, 0.1, 0.1):
        tracker.update(torch.tensor(residual))

    status = tracker.status()
    assert (status.converged, status.num_iterations) == (True, 3)


def test_iteration_count_matches_the_loop_length_without_convergence() -> None:
    tracker = ConvergenceTracker(
        tol=0.5, check_interval=1000, reference=torch.zeros(())
    )

    for _ in range(4):
        tracker.update(torch.tensor(1.0))

    status = tracker.status()
    assert (status.converged, status.num_iterations) == (False, 4)


def test_should_stop_only_synchronizes_on_checkpoint_iterations(
    host_sync_counter: Callable[[], Any],
) -> None:
    tracker = ConvergenceTracker(tol=0.5, check_interval=3, reference=torch.zeros(()))
    tracker.update(torch.tensor(0.1))

    with host_sync_counter() as syncs:
        stops = [tracker.should_stop(iteration) for iteration in range(6)]

    assert stops == [False, False, True, False, False, True]
    assert len(syncs) == 2


def test_should_stop_is_false_while_the_residual_stays_above_tolerance() -> None:
    tracker = ConvergenceTracker(tol=0.5, check_interval=1, reference=torch.zeros(()))
    tracker.update(torch.tensor(1.0))

    assert not tracker.should_stop(0)


def test_start_latches_convergence_without_counting_an_iteration() -> None:
    tracker = ConvergenceTracker(tol=0.5, check_interval=1, reference=torch.zeros(()))

    frozen = tracker.start(torch.tensor(0.1))
    status = tracker.status()

    assert frozen
    assert status.converged
    assert status.num_iterations == 0


def test_status_does_not_synchronize_until_a_diagnostic_is_read(
    host_sync_counter: Callable[[], Any],
) -> None:
    # Regression test: a solve used to copy its convergence flag and
    # iteration count to the host unconditionally, so every inner dual solve
    # of a nested `MinimaxSolver` paid two synchronizations for diagnostics
    # nobody read.
    tracker = ConvergenceTracker(tol=0.5, check_interval=1, reference=torch.zeros(()))
    tracker.update(torch.tensor(0.1))

    with host_sync_counter() as syncs:
        status = tracker.status()

    assert syncs == []
    assert status.converged
    assert status.num_iterations == 1


def test_status_caches_the_diagnostics_it_reads_back(
    host_sync_counter: Callable[[], Any],
) -> None:
    tracker = ConvergenceTracker(tol=0.5, check_interval=1, reference=torch.zeros(()))
    tracker.update(torch.tensor(0.1))
    status = tracker.status()

    with host_sync_counter() as syncs:
        for _ in range(5):
            assert status.converged
            assert status.num_iterations == 1

    assert len(syncs) == 2
