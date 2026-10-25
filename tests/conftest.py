"""Shared pytest fixtures for the Optora test suite."""

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import torch
from hypothesis import settings

FLOAT_DTYPES = (torch.float32, torch.float64)

# Property-based tests run small tensor solves whose first call pays PyTorch's
# one-off kernel warm-up, so a per-example deadline would report that latency
# as flakiness rather than measure anything about the code under test. The
# `thorough` profile is for an occasional deep local search, selected with
# `HYPOTHESIS_PROFILE=thorough`.
settings.register_profile("optora", deadline=None, print_blob=True)
settings.register_profile(
    "thorough", parent=settings.get_profile("optora"), max_examples=2000
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "optora"))


@pytest.fixture(
    params=FLOAT_DTYPES, ids=lambda dtype: str(dtype).removeprefix("torch.")
)
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Iterate over the floating-point dtypes Optora must preserve end to end.

    Returns:
        One of `torch.float32` or `torch.float64`.
    """
    parameter: torch.dtype = request.param
    return parameter


@pytest.fixture
def host_sync_counter() -> Callable[[], Any]:
    """Return a context manager that counts tensor-to-host synchronizations.

    Converting a tensor to a Python value — `bool(tensor)` in an iterative
    loop's stopping test, `int(tensor)` in its reported iteration count — is
    the synchronization every Optora loop must avoid paying per iteration:
    it blocks until the device has produced the value. Counting calls to
    those conversions therefore gives an exact, CPU-testable count of the
    host synchronizations a solve issues.

    Returns:
        A context manager yielding a list whose length is the number of
        synchronizations performed inside the block.
    """

    @contextmanager
    def counter() -> Iterator[list[None]]:
        syncs: list[None] = []
        conversions = {
            "__bool__": torch.Tensor.__bool__,
            "__int__": torch.Tensor.__int__,
            "__float__": torch.Tensor.__float__,
        }

        def counting(original: Callable[[torch.Tensor], Any]) -> Any:
            def wrapper(tensor: torch.Tensor) -> Any:
                syncs.append(None)
                return original(tensor)

            return wrapper

        with ExitStack() as stack:
            for name, original in conversions.items():
                stack.enter_context(
                    patch.object(torch.Tensor, name, counting(original))
                )
            yield syncs

    return counter
