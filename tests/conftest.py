"""Shared pytest fixtures for the Optora test suite."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def host_sync_counter() -> Callable[[], Any]:
    """Return a context manager that counts tensor-to-host synchronizations.

    `bool(tensor)` is the synchronization every Optora iterative loop must
    avoid paying per iteration: it blocks until the device has produced the
    value. Counting calls to `torch.Tensor.__bool__` therefore gives an
    exact, CPU-testable count of the host synchronizations a solve issues.

    Returns:
        A context manager yielding a list whose length is the number of
        synchronizations performed inside the block.
    """

    @contextmanager
    def counter() -> Iterator[list[None]]:
        syncs: list[None] = []
        original = torch.Tensor.__bool__

        def counting(tensor: torch.Tensor) -> bool:
            syncs.append(None)
            return original(tensor)

        with patch.object(torch.Tensor, "__bool__", counting):
            yield syncs

    return counter
