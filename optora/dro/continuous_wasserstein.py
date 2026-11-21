r"""Wasserstein adversary over a continuous input space (Sinha et al. 2018).

The sets in `optora.dro` move probability mass between the points of a
finite support. Sinha, Namkoong and Duchi (2018) instead let the adversary
move every input $z_0$ to any point $z$ of the input space, at the transport
cost $c(z, z_0) = \lVert z - z_0 \rVert_2^2$, and replace the Wasserstein
ball by a penalty on that cost. For a loss $\ell(\theta; z)$ and a penalty
$\gamma > 0$, the robust surrogate of their Eq. (2b) is

$$
\phi_\gamma(\theta; z_0)
= \sup_{z} \big\{ \ell(\theta; z) - \gamma \lVert z - z_0 \rVert_2^2 \big\},
$$

and by their Proposition 1 the worst case over the ball is the dual

$$
\sup_{P:\, W_c(P, P_0) \le \rho} \mathbb{E}_P[\ell(\theta; Z)]
= \inf_{\gamma \ge 0} \big\{ \gamma \rho
    + \mathbb{E}_{P_0}[\phi_\gamma(\theta; Z)] \big\}.
$$

Training at a fixed $\gamma$ minimizes $\mathbb{E}_{P_0}[\phi_\gamma]$, the
Lagrangian relaxation of the constrained problem. If $z \mapsto
\ell(\theta; z)$ has $L$-Lipschitz gradients, the inner objective is
$(2\gamma - L)$-strongly concave for $\gamma > L / 2$, so it has a unique
maximizer $T_\gamma(\theta; z_0)$, the transport map of their Eq. (9), and
gradient ascent converges to it. By Danskin's theorem (their Lemma 1), the
gradient of the surrogate is
$\nabla_\theta \phi_\gamma(\theta; z_0) = \nabla_\theta \ell(\theta;
T_\gamma(\theta; z_0))$, so a training step evaluates the loss at the
transported inputs and descends it, with the map held constant.

`transport_map` approximates $T_\gamma$ for a batch of examples, and
`squared_distance` returns the transport cost of each example. Their means
over a sample give the certificate of Sinha et al. The coupling of $Z$ with
$T_\gamma(\theta; Z)$ costs
$\hat\rho_n = \mathbb{E}_{\hat P_n} \lVert T_\gamma(\theta; Z) - Z
\rVert_2^2$ (their Eq. 14), so the distribution of $T_\gamma(\theta; Z)$
lies in the empirical ball of that radius. The dual bound at the same
$\gamma$, $\gamma \hat\rho_n + \mathbb{E}_{\hat P_n}[\phi_\gamma(\theta;
Z)]$, equals its expected loss, so the worst case over that ball is
exactly $\mathbb{E}_{\hat P_n}[\ell(\theta; T_\gamma(\theta; Z))]$. The
map that `transport_map` returns is approximate. Its points still form a
distribution in the ball of radius $\hat\rho_n$, so their mean loss is a
lower bound on the worst case over that ball, and the gap is the error of
the ascent. In the same way, the value of the inner objective at any point
is a lower bound on $\phi_\gamma$.
"""

import math
from collections.abc import Callable

import torch

from optora.core._transforms import transforms_active
from optora.core.precision import working_dtype
from optora.core.solver_base import require_gradient


def squared_distance(points: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    r"""Return the transport cost $\lVert z - z_0 \rVert_2^2$ of every example.

    Args:
        points: Transported points $z$ of shape `(n, ...)`, with the
            examples along the first dimension.
        inputs: Original points $z_0$, of the same shape as `points`.

    Returns:
        Tensor of shape `(n,)` holding the squared Euclidean distance of
        each example, summed over every dimension but the first. It is
        computed in `optora.core.working_dtype(points, inputs)` and returned
        in the promoted dtype of `points` and `inputs`.

    Raises:
        ValueError: If `points` and `inputs` differ in shape or are
            zero-dimensional.
    """
    if points.shape != inputs.shape:
        raise ValueError(
            f"points of shape {tuple(points.shape)} and inputs of shape "
            f"{tuple(inputs.shape)} must have the same shape."
        )
    if points.ndim == 0:
        raise ValueError(
            "points must have shape (n, ...) with the examples along the first "
            "dimension, got a scalar tensor."
        )
    dtype = working_dtype(points, inputs)
    squares = (points.to(dtype) - inputs.to(dtype)).square()
    if squares.ndim > 1:
        squares = squares.flatten(1).sum(1)
    return squares.to(torch.promote_types(points.dtype, inputs.dtype))


def transport_map(
    loss: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    penalty: float | torch.Tensor,
    num_steps: int = 16,
) -> torch.Tensor:
    r"""Approximate the transport map $T_\gamma(\theta; z_0)$ by gradient ascent.

    The map maximizes $\ell(\theta; z) - \gamma \lVert z - z_0 \rVert_2^2$
    (Eq. 9 of Sinha et al. 2018). The ascent runs on the same objective
    divided by $2\gamma$,

    $$
    f(z) = \frac{\ell(\theta; z)}{2\gamma} - \frac{1}{2} \lVert z - z_0 \rVert_2^2,
    $$

    from $z_1 = z_0$, with the update
    $z_{t+1} = z_t + \nabla f(z_t) / \sqrt{t}$ for $t = 1, \dots, T$. This is
    the loop of the `wrm` function in the authors' code with
    `eps` $= 1 / (2\gamma)$, which takes a first step of size one and then
    `steps` $= T - 1$ steps of sizes $1 / \sqrt{2}, \dots, 1 / \sqrt{T}$.
    After the rescaling the penalty term is 1-strongly concave, and its
    gradient vanishes at $z_0$, so the first step moves by
    $\nabla_z \ell(\theta; z_0) / (2\gamma)$. That step is already the
    exact maximizer when the loss is linear in $z$. In general the ascent
    stops after $T$ steps whether or not it has reached the maximizer, and
    the decaying step makes it slow on examples where the inner objective
    is only weakly concave. Comparing the result with a run of more steps
    shows how far it is from convergence.

    All examples are solved together by differentiating the sum of their
    losses. The loss of each example must therefore depend only on its own
    point, which rules out batch normalization in training mode. The ascent
    leaves no gradient on the parameters of the model, runs under
    `torch.enable_grad()` so that it also works inside `torch.no_grad()`,
    and returns detached points, since the gradient of the surrogate with
    respect to $\theta$ is $\nabla_\theta \ell(\theta; T_\gamma(\theta;
    z_0))$ at the fixed map (Lemma 1 of Sinha et al.). A training step
    therefore calls `loss` once more on the returned points and descends
    that loss.

    The ascent has a fixed trip count and reads no value back to the host,
    so it can be captured in a CUDA graph. Inside a `torch.func` transform,
    where `torch.autograd.grad` is unsupported, the gradient is taken with
    `torch.func.grad` instead. For example, `torch.func.vmap` over a tensor
    of penalties solves one problem per penalty in a single call.

    Args:
        loss: Maps a batch of points of the shape and dtype of `inputs` to
            the loss of each example, a tensor of shape `(n,)`.
        inputs: Original points $z_0$ of shape `(n, ...)`, with the examples
            along the first dimension, in a floating-point dtype.
        penalty: The penalty $\gamma > 0$, either a float or a tensor of
            shape `()`, `(1,)` or `(n,)`, for example one value per example.
            The values of a tensor penalty are not checked, since the check
            would read them on the host. A value of `math.inf` leaves the
            inputs unchanged.
        num_steps: Number of ascent steps $T$, at least one.

    Returns:
        The transported points, detached and of the shape and dtype of
        `inputs`. The ascent runs in `optora.core.working_dtype(inputs)`, so
        half-precision inputs are transported in float32, passed to `loss`
        rounded to their own dtype, and rounded once at the end.

    Raises:
        ValueError: If `num_steps` is less than one, `penalty` is a float
            that is not positive, a tensor `penalty` has the wrong shape,
            `inputs` is zero-dimensional or not floating point, `loss`
            does not return one value per example, or the loss does not
            depend on the points through autograd.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least one, got {num_steps}.")
    if inputs.ndim == 0:
        raise ValueError(
            "inputs must have shape (n, ...) with the examples along the first "
            "dimension, got a scalar tensor."
        )
    if not inputs.is_floating_point():
        raise ValueError(f"inputs must be floating point, got {inputs.dtype}.")
    dtype = working_dtype(inputs)
    origin = inputs.detach().to(dtype)
    scale = _half_inverse_penalty(penalty, origin)
    num_examples = inputs.shape[0]

    def total_loss(points: torch.Tensor) -> torch.Tensor:
        values = loss(points.to(inputs.dtype))
        if values.shape != (num_examples,):
            raise ValueError(
                f"loss must return one value per example, of shape "
                f"({num_examples},), got {tuple(values.shape)}."
            )
        return values.sum()

    functional = transforms_active()
    points = origin
    with torch.enable_grad():
        for step in range(1, num_steps + 1):
            gradient = _gradient(total_loss, points, functional)
            ascent = scale * gradient - (points - origin)
            points = points + ascent / math.sqrt(step)
    return points.detach().to(inputs.dtype)


def _half_inverse_penalty(
    penalty: float | torch.Tensor, origin: torch.Tensor
) -> float | torch.Tensor:
    r"""Return $1 / (2\gamma)$, shaped to broadcast against `origin`.

    Args:
        penalty: The penalty $\gamma$, a float or a tensor of shape `()`,
            `(1,)` or `(n,)`.
        origin: Detached original points of shape `(n, ...)`, whose device
            and dtype a tensor penalty is moved to.

    Returns:
        A float for a float penalty, otherwise a detached tensor of shape
        `()` or `(n, 1, ..., 1)` with one trailing singleton per dimension
        of an example.

    Raises:
        ValueError: If a float `penalty` is not positive, or a tensor
            `penalty` is not of shape `()`, `(1,)` or `(n,)`.
    """
    if not isinstance(penalty, torch.Tensor):
        if not penalty > 0:
            raise ValueError(f"penalty must be positive, got {penalty}.")
        return 0.5 / penalty
    if penalty.ndim > 1 or (
        penalty.ndim == 1 and penalty.shape[0] not in (1, origin.shape[0])
    ):
        raise ValueError(
            f"penalty must have shape (), (1,) or ({origin.shape[0]},), got "
            f"{tuple(penalty.shape)}."
        )
    scale = 0.5 / penalty.detach().to(device=origin.device, dtype=origin.dtype)
    if scale.ndim == 0:
        return scale
    return scale.reshape((-1,) + (1,) * (origin.ndim - 1))


def _gradient(
    objective: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    functional: bool,
) -> torch.Tensor:
    """Return the gradient of a scalar objective at detached points.

    Args:
        objective: Scalar-valued function of the points.
        points: Points the gradient is taken at, left unchanged.
        functional: Whether to differentiate with `torch.func.grad`, as a
            `torch.func` transform of the caller requires.

    Returns:
        The gradient of `objective` with respect to `points`.

    Raises:
        ValueError: If `functional` is unset and `objective` does not depend
            on the points through autograd.
    """
    if functional:
        gradient: torch.Tensor = torch.func.grad(objective)(points)
        return gradient
    variable = points.detach().requires_grad_(True)
    (raw_gradient,) = torch.autograd.grad(
        objective(variable), variable, allow_unused=True
    )
    return require_gradient(raw_gradient, "the points")
