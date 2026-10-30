"""Tests for the `torch.func` introspection in `optora.core._transforms`."""

from collections.abc import Callable

import pytest
import torch

from optora.core import _transforms
from optora.core._transforms import transforms_active, vmap_active

Probe = Callable[[torch.Tensor], torch.Tensor]


@pytest.fixture(params=["stack", "walk"], autouse=True)
def interpreter_stack(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    # The walk is the only path on PyTorch releases that lack
    # `get_interpreter_stack`, so it runs on every release, not just those.
    if request.param == "walk":
        monkeypatch.delattr(
            _transforms._functorch, "get_interpreter_stack", raising=False
        )
    return str(request.param)


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


def test_lists_nested_transforms_outermost_first() -> None:
    record: list[list[torch._C._functorch.TransformType]] = []

    def function(point: torch.Tensor) -> torch.Tensor:
        record.append(_transforms._active_transforms())
        return torch.sum(point**2)

    torch.func.vmap(torch.func.grad(function))(torch.ones(2, 3))

    assert record == [[_transforms.TransformType.Vmap, _transforms.TransformType.Grad]]


def test_reading_the_stack_leaves_the_transforms_intact() -> None:
    # A walk that failed to restore a popped layer would detach the inner
    # computation from its transform and corrupt the derivative.
    def function(point: torch.Tensor) -> torch.Tensor:
        vmap_active()
        return torch.sum(point**3)

    point = torch.arange(6.0).reshape(2, 3)

    hessian = torch.func.vmap(torch.func.hessian(function))(point)

    assert torch.equal(hessian, torch.vmap(torch.diag)(6.0 * point))


def test_the_walk_restores_the_stack_when_reading_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second pop fails with the outer layer already popped, which the
    # walk must push back before the error reaches the caller.
    pop = _transforms._functorch.pop_dynamic_layer_stack
    pops: list[object] = []

    def failing_pop() -> object:
        if pops:
            raise RuntimeError("pop failed")
        pops.append(pop())
        return pops[-1]

    def function(point: torch.Tensor) -> torch.Tensor:
        with monkeypatch.context() as patch:
            patch.setattr(
                _transforms._functorch, "pop_dynamic_layer_stack", failing_pop
            )
            with pytest.raises(RuntimeError, match="pop failed"):
                _transforms._walk_interpreter_stack()
        assert _transforms._walk_interpreter_stack() == [
            _transforms.TransformType.Vmap,
            _transforms.TransformType.Grad,
        ]
        return torch.sum(point**2)

    gradient = torch.func.vmap(torch.func.grad(function))(torch.ones(2, 3))

    assert len(pops) == 1
    assert torch.equal(gradient, torch.full((2, 3), 2.0))
