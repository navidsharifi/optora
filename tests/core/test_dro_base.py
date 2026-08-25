"""Tests for the `AmbiguitySet` ABC contract."""

import pytest
import torch
from torch import nn

from optora.core.divergence_base import Divergence
from optora.core.dro_base import AmbiguitySet


class _AbsoluteDifferenceDivergence(Divergence):
    """Sum-of-absolute-differences divergence used to exercise the ABC."""

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.abs(p - q).sum()


class _MaxLossAmbiguitySet(AmbiguitySet):
    """Ambiguity set whose worst case is simply the maximum loss."""

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.max()


@pytest.fixture
def nominal() -> torch.Tensor:
    return torch.tensor([0.5, 0.5])


def test_ambiguity_set_cannot_be_instantiated_directly(nominal: torch.Tensor) -> None:
    with pytest.raises(TypeError):
        AmbiguitySet(  # type: ignore[abstract]
            nominal, _AbsoluteDifferenceDivergence(), radius=0.1
        )


def test_negative_radius_raises_value_error(nominal: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _MaxLossAmbiguitySet(nominal, _AbsoluteDifferenceDivergence(), radius=-1.0)


def test_contains_uses_divergence_and_radius(nominal: torch.Tensor) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )

    assert ambiguity_set.contains(torch.tensor([0.6, 0.4]))
    assert not ambiguity_set.contains(torch.tensor([0.9, 0.1]))


def test_worst_case_expectation_is_delegated_to_subclass(
    nominal: torch.Tensor,
) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )
    loss = torch.tensor([1.0, 3.0, 2.0])

    result = ambiguity_set.worst_case_expectation(loss)

    assert torch.equal(result, torch.tensor(3.0))


def test_ambiguity_set_is_an_nn_module_with_registered_buffer_and_submodule(
    nominal: torch.Tensor,
) -> None:
    divergence = _AbsoluteDifferenceDivergence()
    ambiguity_set = _MaxLossAmbiguitySet(nominal, divergence, radius=0.5)

    assert isinstance(ambiguity_set, nn.Module)
    assert "nominal" in dict(ambiguity_set.named_buffers())
    assert dict(ambiguity_set.named_modules())["divergence"] is divergence


def test_to_moves_nominal_buffer_and_divergence_submodule_together(
    nominal: torch.Tensor,
) -> None:
    ambiguity_set = _MaxLossAmbiguitySet(
        nominal, _AbsoluteDifferenceDivergence(), radius=0.5
    )

    ambiguity_set = ambiguity_set.to(dtype=torch.float64)

    assert ambiguity_set.nominal.dtype == torch.float64
