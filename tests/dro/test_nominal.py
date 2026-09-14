"""Shared nominal-distribution contract across public ambiguity sets."""

from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.divergences.f_divergence import PhiDivergence
from optora.dro import (
    ChiSquareAmbiguitySet,
    KLAmbiguitySet,
    PhiAmbiguitySet,
    TotalVariationAmbiguitySet,
    WassersteinAmbiguitySet,
)


def _phi_ambiguity_set(nominal: torch.Tensor, **kwargs: bool) -> AmbiguitySet:
    return PhiAmbiguitySet(
        nominal,
        divergence=PhiDivergence(
            phi=lambda ratio: ratio * torch.log(ratio) - ratio + 1
        ),
        radius=0.0,
        phi_conjugate=torch.expm1,
        **kwargs,
    )


def _wasserstein_ambiguity_set(nominal: torch.Tensor, **kwargs: bool) -> AmbiguitySet:
    cost = nominal.new_tensor([[0.0, 1.0], [1.0, 0.0]])
    return WassersteinAmbiguitySet(nominal, cost=cost, radius=0.0, **kwargs)


@pytest.fixture(
    params=[
        pytest.param(partial(KLAmbiguitySet, radius=0.0), id="kl"),
        pytest.param(_phi_ambiguity_set, id="phi"),
        pytest.param(partial(ChiSquareAmbiguitySet, radius=0.0), id="chi-square"),
        pytest.param(
            partial(TotalVariationAmbiguitySet, radius=0.0), id="total-variation"
        ),
        pytest.param(_wasserstein_ambiguity_set, id="wasserstein"),
    ]
)
def make_ambiguity_set(
    request: pytest.FixtureRequest,
) -> Callable[..., AmbiguitySet]:
    return request.param


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([-1e-8, 1.0 + 1e-8], id="negative-with-unit-sum"),
        pytest.param([0.0, 0.0], id="zero-mass"),
        pytest.param([0.4, 0.4], id="below-unit-mass"),
        pytest.param([0.6, 0.6], id="above-unit-mass"),
        pytest.param([float("nan"), 0.5], id="nan"),
        pytest.param([float("inf"), 0.5], id="positive-infinity"),
        pytest.param([float("-inf"), 0.5], id="negative-infinity"),
        pytest.param([[0.25, 0.25], [0.25, 0.25]], id="only-global-sum-is-one"),
        pytest.param([[0.5, 0.5], [0.1, 0.1]], id="invalid-second-row"),
        pytest.param([[0.75, 0.75], [0.25, 0.25]], id="only-mean-row-sum-is-one"),
    ],
)
def test_invalid_nominal_raises_value_error(
    make_ambiguity_set: Callable[..., AmbiguitySet],
    values: list[float] | list[list[float]],
) -> None:
    with pytest.raises(ValueError, match="nominal"):
        make_ambiguity_set(torch.tensor(values, dtype=torch.float64), validate=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("values", [[0.0, 1.0], [[0.25, 0.75], [1.0, 0.0]]])
def test_valid_nominal_is_preserved_as_the_registered_buffer(
    make_ambiguity_set: Callable[..., AmbiguitySet],
    dtype: torch.dtype,
    values: list[float] | list[list[float]],
) -> None:
    nominal = torch.tensor(values, dtype=dtype)
    original = nominal.clone()

    ambiguity_set = make_ambiguity_set(nominal, validate=True)

    assert ambiguity_set.nominal is nominal
    assert dict(ambiguity_set.named_buffers())["nominal"] is nominal
    assert torch.equal(nominal, original)


@pytest.mark.parametrize("offset", [-5e-7, 5e-7])
def test_unit_mass_within_absolute_tolerance_is_not_normalized(
    make_ambiguity_set: Callable[..., AmbiguitySet], offset: float
) -> None:
    nominal = torch.tensor([0.5, 0.5 + offset], dtype=torch.float64)
    original = nominal.clone()

    ambiguity_set = make_ambiguity_set(nominal, validate=True)

    assert ambiguity_set.nominal is nominal
    assert torch.equal(nominal, original)


@pytest.mark.parametrize("offset", [-2e-6, 2e-6])
def test_unit_mass_outside_absolute_tolerance_raises_value_error(
    make_ambiguity_set: Callable[..., AmbiguitySet], offset: float
) -> None:
    nominal = torch.tensor([0.5, 0.5 + offset], dtype=torch.float64)

    with pytest.raises(ValueError, match="nominal"):
        make_ambiguity_set(nominal, validate=True)


def test_nominal_keeps_autograd_connection_to_its_input(
    make_ambiguity_set: Callable[..., AmbiguitySet],
) -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    nominal = logits.softmax(dim=-1)
    ambiguity_set = make_ambiguity_set(nominal, validate=True)

    expectation = ambiguity_set.worst_case_expectation(nominal.new_tensor([1.0, 3.0]))
    (gradient,) = torch.autograd.grad(expectation, logits)

    assert torch.equal(gradient, nominal.new_tensor([-0.5, 0.5]))


@pytest.mark.parametrize("kwargs", [{}, {"validate": False}], ids=["default", "false"])
@pytest.mark.parametrize("values", [[0.5, 0.5], [-0.1, 0.2]])
def test_unchecked_construction_does_not_read_nominal_values(
    make_ambiguity_set: Callable[..., AmbiguitySet],
    host_sync_counter: Callable[[], Any],
    kwargs: dict[str, bool],
    values: list[float],
) -> None:
    nominal = torch.tensor(values)

    with host_sync_counter() as syncs:
        ambiguity_set = make_ambiguity_set(nominal, **kwargs)

    assert syncs == []
    assert ambiguity_set.nominal is nominal


@pytest.mark.parametrize("kwargs", [{}, {"validate": False}], ids=["default", "false"])
def test_unchecked_construction_supports_meta_tensors(
    make_ambiguity_set: Callable[..., AmbiguitySet], kwargs: dict[str, bool]
) -> None:
    # Meta tensors cannot supply values to host-side reductions. This covers
    # value reads beyond the Tensor.__bool__ calls counted by the fixture.
    nominal = torch.empty(2, device="meta")

    ambiguity_set = make_ambiguity_set(nominal, **kwargs)

    assert ambiguity_set.nominal is nominal
    assert all(buffer.device.type == "meta" for buffer in ambiguity_set.buffers())
