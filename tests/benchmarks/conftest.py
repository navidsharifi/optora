"""Fixtures for the performance guard in `tests/benchmarks/`.

Each benchmark pins two things about one representative workload: its exact
host-synchronization count, a hard assertion that holds on every machine,
and its wall-clock time, measured by `pytest-benchmark` and compared against
the base commit in CI. The default test run passes `--benchmark-disable`
(see `pyproject.toml`), which still calls every benchmarked function once,
so the synchronization budgets are enforced by the ordinary test suite too.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


def pytest_benchmark_update_machine_info(machine_info: dict[str, Any]) -> None:
    """Drop the current CPU clock from the machine info saved with a run.

    On Linux `py-cpuinfo` reads the instantaneous frequency, so two runs on
    one machine record different values and every base-versus-head
    comparison would warn that the machine changed
    (ionelmc/pytest-benchmark#255).

    Args:
        machine_info: Machine info about to be saved or compared, updated in
            place.
    """
    for key in ("hz_actual", "hz_actual_friendly"):
        machine_info["cpu"].pop(key, None)


@pytest.fixture
def guarded_benchmark(
    benchmark: BenchmarkFixture, host_sync_counter: Callable[[], Any]
) -> Callable[..., Any]:
    """Return a benchmark runner that first asserts a host-sync budget.

    The budget is an exact count rather than an upper bound, so a change
    that adds a synchronization fails here, and a change that removes one
    has to lower the budget, keeping it a faithful record of the workload.
    The counted call runs before timing starts, so the patched tensor
    conversions never contaminate the measurement, and doubles as warm-up.

    Returns:
        A callable `(function, *, host_syncs)` that runs `function` once
        under `host_sync_counter`, asserts it synchronized exactly
        `host_syncs` times, and returns the result of benchmarking it.
    """

    def run(function: Callable[[], Any], *, host_syncs: int) -> Any:
        with host_sync_counter() as syncs:
            function()
        assert len(syncs) == host_syncs
        return benchmark(function)

    return run
