"""Introspection of the active `torch.func` transforms.

PyTorch has no public query for whether code runs inside a `torch.func`
transform, which a solver needs to know: `Tensor.requires_grad_` and
`torch.autograd.grad` are unsupported there, and under `vmap` a tensor has
no single value to read on the host. This module is the one place optora
reads PyTorch's private interpreter stack, so a change to it is a one-line
fix here.
"""

from torch._C import _functorch
from torch._C._functorch import TransformType


def _active_transforms() -> list[TransformType]:
    """Return the kinds of the active `torch.func` transforms, outermost first."""
    get_interpreter_stack = getattr(_functorch, "get_interpreter_stack", None)
    if get_interpreter_stack is None:
        return _walk_interpreter_stack()
    stack = get_interpreter_stack()
    return [] if stack is None else [interpreter.key() for interpreter in stack]


def _walk_interpreter_stack() -> list[TransformType]:
    """Read the interpreter stack where PyTorch exposes only its top.

    Older PyTorch releases, down to the supported minimum, lack
    `get_interpreter_stack`. The stack is then read one level at a time by
    popping each layer after peeking at it, the mechanism behind PyTorch's
    own `temporarily_pop_interpreter_stack`, and every popped layer is
    pushed back in order, so the caller's transforms are left as found.

    Returns:
        The kinds of the active transforms, outermost first.
    """
    kinds: list[TransformType] = []
    layers = []
    try:
        while (interpreter := _functorch.peek_interpreter_stack()) is not None:
            kinds.append(interpreter.key())
            layers.append(_functorch.pop_dynamic_layer_stack())
    finally:
        for layer in reversed(layers):
            _functorch.push_dynamic_layer_stack(layer)
    return kinds[::-1]


def transforms_active() -> bool:
    """Return whether the caller runs inside any `torch.func` transform."""
    return bool(_active_transforms())


def vmap_active() -> bool:
    """Return whether the caller runs inside a `torch.func.vmap`, at any level."""
    return TransformType.Vmap in _active_transforms()
