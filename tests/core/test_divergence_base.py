"""Tests for the `Divergence` ABC contract."""

import pytest
import torch

from optora.core.divergence_base import Divergence


class _ZeroDivergence(Divergence):
    """Minimal concrete divergence used to exercise the ABC contract."""

    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.zeros(())


def test_divergence_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Divergence()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class _IncompleteDivergence(Divergence):
        pass

    with pytest.raises(TypeError):
        _IncompleteDivergence()  # type: ignore[abstract]


def test_concrete_divergence_is_callable() -> None:
    divergence = _ZeroDivergence()
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.25, 0.75])

    value = divergence(p, q)

    assert torch.equal(value, torch.zeros(()))
