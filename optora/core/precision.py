"""Working precision shared by every divergence and ambiguity set."""

import torch


def working_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Return the dtype a numerical routine over `tensors` computes in.

    This is the type promotion of the inputs' dtypes, raised to at least
    `torch.float32`. Half-precision inputs (`torch.float16`,
    `torch.bfloat16`) are therefore solved in float32 and only the result
    is rounded back, the same policy `torch.autocast` applies to reductions
    such as `logsumexp`, `softmax` and `cumsum`. The worst cases are long
    reductions, bisections whose trip count is read off the dtype, and
    differences of nearly equal sums, which lose every significant digit
    of an 8-bit (`bfloat16`) or 11-bit (`float16`) significand. Computing
    them in float32 leaves the final rounding to the input dtype as the
    only visible error. Float32 and float64 inputs are
    computed in their own precision, so the cast is a no-op that copies
    nothing.

    Args:
        *tensors: Floating-point tensors entering the computation.

    Returns:
        The promoted dtype of `tensors`, at least `torch.float32`.
    """
    dtype = torch.float32
    for tensor in tensors:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype
