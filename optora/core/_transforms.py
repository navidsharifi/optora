"""Introspection of the active `torch.func` transforms.

PyTorch has no public query for whether code runs inside a `torch.func`
transform, which a solver needs to know: `Tensor.requires_grad_` and
`torch.autograd.grad` are unsupported there, and under `vmap` a tensor has
no single value to read on the host. This module is the one place optora
reads PyTorch's private interpreter stack, so a change to it is a one-line
fix here.
"""

import torch
from torch._C._functorch import TransformType


def _active_transforms() -> list[TransformType]:
    """Return the kinds of the active `torch.func` transforms, outermost first."""
    stack = torch._C._functorch.get_interpreter_stack()
    return [] if stack is None else [interpreter.key() for interpreter in stack]


def transforms_active() -> bool:
    """Return whether the caller runs inside any `torch.func` transform."""
    return bool(_active_transforms())


def vmap_active() -> bool:
    """Return whether the caller runs inside a `torch.func.vmap`, at any level."""
    return TransformType.Vmap in _active_transforms()
