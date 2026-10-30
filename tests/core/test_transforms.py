"""Tests for the `torch.func` introspection in `optora.core._transforms`."""

from collections.abc import Callable

import pytest
import torch

from optora.core._transforms import transforms_active, vmap_active

Probe = Callable[[torch.Tensor], torch.Tensor]


def _probing(record: list[tuple[bool, bool]]) -> Probe:
    def function(point: torch.Tensor) -> torch.Tensor:
        record.append((transforms_active(), vmap_active()))
        return torch.sum(point**2)

    return function


def test_nothing_is_active_outside_a_transform() -> None:
    assert not transforms_active()
    assert not vmap_active()


@pytest.mark.parametrize(
    ("transform", "point", "expected"),
    [
        (torch.func.grad, torch.ones(3), (True, False)),
        (torch.func.vmap, torch.ones(2, 3), (True, True)),
        (
            lambda function: torch.func.vmap(torch.func.grad(function)),
            torch.ones(2, 3),
            (True, True),
        ),
        (torch.func.hessian, torch.ones(3), (True, True)),
    ],
    ids=["grad", "vmap", "vmap_of_grad", "hessian"],
)
def test_reports_the_active_transforms(
    transform: Callable[[Probe], Probe],
    point: torch.Tensor,
    expected: tuple[bool, bool],
) -> None:
    record: list[tuple[bool, bool]] = []

    transform(_probing(record))(point)

    assert record and all(entry == expected for entry in record)
