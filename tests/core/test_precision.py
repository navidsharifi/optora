"""Tests for the working precision shared by divergences and ambiguity sets."""

import pytest
import torch

from optora.core.precision import working_dtype


def _tensor(dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(2, dtype=dtype)


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ((torch.float16,), torch.float32),
        ((torch.bfloat16,), torch.float32),
        ((torch.float32,), torch.float32),
        ((torch.float64,), torch.float64),
        ((torch.float16, torch.bfloat16), torch.float32),
        ((torch.bfloat16, torch.float32), torch.float32),
        ((torch.bfloat16, torch.float64), torch.float64),
        ((torch.float32, torch.float64), torch.float64),
    ],
    ids=lambda value: str(value).replace("torch.", ""),
)
def test_working_dtype_promotes_and_raises_half_precision_to_float32(
    dtypes: tuple[torch.dtype, ...], expected: torch.dtype
) -> None:
    assert working_dtype(*(_tensor(dtype) for dtype in dtypes)) == expected


def test_working_dtype_of_no_tensors_is_float32() -> None:
    assert working_dtype() == torch.float32


def test_casting_to_the_working_dtype_copies_nothing_in_full_precision(
    dtype: torch.dtype,
) -> None:
    tensor = _tensor(dtype)

    assert tensor.to(working_dtype(tensor)) is tensor
