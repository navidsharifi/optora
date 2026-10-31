r"""Group distributionally robust optimization (Sagawa et al. 2020).

The training population is a mixture of $G$ known groups, for example the
combinations of a label and a spurious attribute, and group DRO minimizes the
risk of the worst group rather than the average one:

$$
\min_\theta \max_{g = 1, \dots, G} R_g(\theta)
= \min_\theta \sup_{q \in \Delta_G} \sum_{g=1}^{G} q_g R_g(\theta),
\qquad R_g(\theta) = \mathbb{E}_{Z \sim P_g}[\ell(\theta; Z)].
$$

Over the empirical group risks, the inner supremum is a worst case over
distributions on the groups, so it is an existing ambiguity set applied to
the vector of group losses that `group_losses` computes, centred on the
group proportions $p$:

- `CVaRAmbiguitySet(p, alpha=0.0)` is the maximum over the groups with
  $p_g > 0$; its level $\alpha$ interpolates between ERM at $\alpha = 1$ and
  group DRO as $\alpha \to 0$.
- `TotalVariationAmbiguitySet(p, radius=1.0)` is the maximum over all
  groups.
- `KLAmbiguitySet(p, radius)` and `ChiSquareAmbiguitySet(p, radius)` are
  balls around the group mix, and equal the worst group exactly once the
  radius reaches $\log(1/p_{g^\star})$, respectively $1/p_{g^\star} - 1$,
  for the worst group $g^\star$.

On a minibatch, the maximum of the minibatch group means overestimates the
worst group risk ($\mathbb{E}[\max] \ge \max \mathbb{E}$), and the worst
group can change from one minibatch to the next. `OnlineGroupDRO` is the
online algorithm of Sagawa et al. instead: it keeps the group weights as
state and moves them by exponentiated-gradient ascent, one step per
minibatch.
"""

import torch
from torch import nn

from optora.core._transforms import transforms_active
from optora.core.dro_base import _broadcast_batch_shapes
from optora.core.precision import working_dtype


def _broadcasts_to(shape: torch.Size, target: torch.Size) -> bool:
    """Return whether `shape` broadcasts to exactly `target`."""
    try:
        return _broadcast_batch_shapes(shape=shape, target=target) == target
    except ValueError:
        return False


def group_losses(
    loss: torch.Tensor, groups: torch.Tensor, num_groups: int
) -> torch.Tensor:
    r"""Average per-sample losses within each group.

    The group loss of group $g$ is the empirical group risk

    $$
    \hat R_g = \frac{1}{n_g} \sum_{i:\, g_i = g} \ell_i,
    \qquad n_g = |\{i : g_i = g\}|,
    $$

    and is zero for a group without samples, whose sum is empty. Weighting
    the group losses by the group proportions $n_g / n$ recovers the mean
    of `loss`. The reduction is a scatter-add, so it never synchronizes with
    the host and its gradient with respect to $\ell_i$ is $1 / n_{g_i}$.

    Args:
        loss: Per-sample losses of shape `(..., n)`.
        groups: Integer group index of each sample, in `[0, num_groups)`,
            of a shape broadcasting against `loss`: `(n,)` for one group
            assignment shared by every batch element. Indices out of range
            are not checked, since the check would read the data on the
            host: a CPU tensor raises an indexing error and a CUDA tensor
            triggers a device-side assertion.
        num_groups: Number of groups $G$, at least one.

    Returns:
        The group losses of shape `(..., num_groups)` over the broadcast
        batch, computed in `optora.core.working_dtype(loss)` and returned in
        `loss`'s dtype.

    Raises:
        ValueError: If `num_groups` is less than one, `loss` is
            zero-dimensional, or `groups` is not an integer tensor or does
            not broadcast against `loss`.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be at least one, got {num_groups}.")
    if loss.ndim == 0:
        raise ValueError(
            "loss must have shape (..., n) with a trailing sample dimension, "
            "got a scalar tensor."
        )
    if groups.is_floating_point() or groups.is_complex() or groups.dtype == torch.bool:
        raise ValueError(f"groups must be an integer tensor, got {groups.dtype}.")
    shape = _broadcast_batch_shapes(loss=loss.shape, groups=groups.shape)
    dtype = working_dtype(loss)
    index = groups.to(torch.int64).expand(shape)
    values = loss.to(dtype).expand(shape)
    totals = torch.zeros(
        (*shape[:-1], num_groups), dtype=dtype, device=loss.device
    ).scatter_add(-1, index, values)
    counts = torch.zeros(
        (*shape[:-1], num_groups), dtype=dtype, device=loss.device
    ).scatter_add(-1, index, torch.ones(shape, dtype=dtype, device=loss.device))
    return (totals / torch.clamp(counts, min=1.0)).to(loss.dtype)


class OnlineGroupDRO(nn.Module):
    r"""Online group DRO by exponentiated-gradient ascent on the group weights.

    Group DRO is the minimax problem
    $\min_\theta \max_{q \in \Delta_G} \sum_g q_g R_g(\theta)$. The online
    algorithm of Sagawa et al. (2020, Algorithm 1) plays stochastic mirror
    ascent on $q$ with the entropic mirror map, that is exponentiated
    gradient (Nemirovski et al. 2009), against stochastic gradient descent
    on $\theta$. On a minibatch with group losses $\hat\ell_g$ it first
    moves the weights,

    $$
    q_g \leftarrow \frac{q_g\, e^{\eta \hat\ell_g}}
        {\sum_h q_h\, e^{\eta \hat\ell_h}},
    $$

    and then descends the weighted loss $\sum_g q_g \hat\ell_g$ in
    $\theta$. `step` takes the ascent step and `forward` evaluates the
    weighted loss, so one training step reads

    ```python
    group_loss = group_losses(per_sample_loss, groups, num_groups)
    adversary.step(group_loss)
    adversary(group_loss).backward()
    optimizer.step()
    ```

    in that order, as in Algorithm 1: the model step uses the weights
    already updated on the same minibatch. `forward` alone never changes
    the weights; without `step` it is a fixed reweighting.

    A group absent from the minibatch has group loss zero (see
    `group_losses`), so `step` leaves its unnormalized weight unchanged,
    which is how Algorithm 1 treats every group it did not sample. Draw
    minibatches with every group present, for example by sampling a fixed
    number of examples per group, and each group loss is an unbiased
    estimate of that group's risk given the minibatch's group counts.

    After $T$ steps on the same group losses $\ell$ the weights are the
    exponential tilt $q_0\, e^{T \eta \ell} / Z$ of the initial weights,
    which is the worst case of the KL ball around $q_0$ whose radius is the
    tilt's KL divergence from $q_0$ (see `optora.dro.KLAmbiguitySet`). The
    iterates follow the KL-DRO worst-case path and approach the worst group
    as $T \eta$ grows.

    The weights are state, kept as normalized log-weights in a buffer, so
    `.to(...)` moves them, `state_dict` checkpoints them, and a group with
    zero initial weight keeps weight zero. `step` runs without autograd and
    is not part of the objective; `forward` is a pure function of the
    weights and the group losses, so it composes with `torch.func`
    transforms, while `step` must run outside them. Neither synchronizes
    with the host.

    Attributes:
        log_weights: Normalized log group weights of shape `(..., G)`,
            `-inf` for a group with zero weight. Leading dimensions index
            independent adversaries, one per batch element of the losses.
        step_size: Nonnegative ascent step size $\eta$ in inverse units of
            the loss, a float or a tensor broadcasting against the leading
            dimensions of `log_weights`, giving each adversary its own.
    """

    log_weights: torch.Tensor
    step_size: float | torch.Tensor

    def __init__(
        self,
        initial_weights: torch.Tensor,
        step_size: float | torch.Tensor,
        validate: bool = False,
    ) -> None:
        r"""Initialize the group weights.

        Args:
            initial_weights: Starting group weights $q_0$, a nonnegative
                floating-point tensor of shape `(..., G)` that sums to one
                along its last dimension. Sagawa et al. start from the
                uniform distribution; starting from the group proportions
                makes a zero step size exactly ERM. Leading dimensions give
                independent adversaries, for example one per jointly
                trained model.
            step_size: Nonnegative ascent step size $\eta$. A tensor is
                registered as a buffer and must broadcast against the
                leading dimensions of `initial_weights`.
            validate: Whether to check that `initial_weights` is a
                distribution and a tensor `step_size` is finite and
                nonnegative. The check reads reductions on the host, so it
                is opt-in.

        Raises:
            ValueError: If `initial_weights` is not a floating-point tensor
                with at least one dimension and one group, a float
                `step_size` is negative or not finite, a tensor
                `step_size` does not broadcast against the leading
                dimensions, or `validate` is set and a check fails.
        """
        super().__init__()
        if not initial_weights.is_floating_point() or initial_weights.ndim == 0:
            raise ValueError(
                "initial_weights must be a floating-point tensor of shape (..., G), "
                f"got {initial_weights.dtype} of shape {tuple(initial_weights.shape)}."
            )
        if initial_weights.shape[-1] == 0:
            raise ValueError("initial_weights must have at least one group.")
        batch_shape = initial_weights.shape[:-1]
        if isinstance(step_size, torch.Tensor):
            if not _broadcasts_to(step_size.shape, batch_shape):
                raise ValueError(
                    f"step_size of shape {tuple(step_size.shape)} must broadcast "
                    f"against the batch shape {tuple(batch_shape)} of "
                    "initial_weights."
                )
            self.register_buffer("step_size", step_size)
        else:
            if not 0.0 <= step_size < float("inf"):
                raise ValueError(
                    f"step_size must be finite and nonnegative, got {step_size}."
                )
            self.step_size = step_size
        if validate:
            self._validate(initial_weights)
        positive = initial_weights > 0.0
        log_weights = torch.where(
            positive,
            torch.log(torch.where(positive, initial_weights, 1.0)),
            -torch.inf,
        )
        self.register_buffer(
            "log_weights",
            log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True),
        )

    def _validate(self, initial_weights: torch.Tensor) -> None:
        """Check the initial weights and a tensor step size on the host."""
        checks = [
            torch.all(torch.isfinite(initial_weights) & (initial_weights >= 0.0)),
            torch.all(
                torch.abs(torch.sum(initial_weights, dim=-1) - 1.0)
                <= initial_weights.shape[-1] * torch.finfo(initial_weights.dtype).eps
            ),
        ]
        if isinstance(self.step_size, torch.Tensor):
            checks.append(
                torch.all(torch.isfinite(self.step_size) & (self.step_size >= 0.0))
            )
        valid = torch.stack(checks).tolist()
        if not valid[0] or not valid[1]:
            raise ValueError(
                "initial_weights must be nonnegative and sum to one along its "
                "last dimension."
            )
        if len(valid) > 2 and not valid[2]:
            raise ValueError("step_size must be finite and nonnegative.")

    @property
    def num_groups(self) -> int:
        """Number of groups $G$."""
        return self.log_weights.shape[-1]

    @property
    def weights(self) -> torch.Tensor:
        """Current group weights $q$ of shape `(..., G)`, on the simplex."""
        return torch.exp(self.log_weights)

    def _check_group_loss(self, group_loss: torch.Tensor) -> None:
        """Raise unless `group_loss` has shape `(..., G)`."""
        if group_loss.ndim == 0 or group_loss.shape[-1] != self.num_groups:
            raise ValueError(
                f"group_loss must have shape (..., {self.num_groups}), "
                f"got {tuple(group_loss.shape)}."
            )

    def _scaled(self, group_loss: torch.Tensor) -> torch.Tensor:
        r"""Return $\eta \hat\ell$ with the step size aligned to the groups."""
        if isinstance(self.step_size, torch.Tensor):
            return self.step_size.to(group_loss.dtype).unsqueeze(-1) * group_loss
        return self.step_size * group_loss

    def step(self, group_loss: torch.Tensor) -> None:
        r"""Take one exponentiated-gradient ascent step on the group weights.

        Sets $q \leftarrow q\, e^{\eta \hat\ell} / \sum_h q_h e^{\eta
        \hat\ell_h}$ in place, computed in log space without autograd.

        Args:
            group_loss: Group losses of shape `(..., G)`, broadcasting to
                the shape of `log_weights`. Only its value is used.

        Raises:
            ValueError: If `group_loss` does not have shape `(..., G)` or
                does not broadcast to the shape of `log_weights`.
            RuntimeError: If called inside a `torch.func` transform, which
                cannot mutate the weights.
        """
        self._check_group_loss(group_loss)
        if not _broadcasts_to(group_loss.shape, self.log_weights.shape):
            raise ValueError(
                f"group_loss of shape {tuple(group_loss.shape)} must broadcast to "
                f"the weights' shape {tuple(self.log_weights.shape)}; build the "
                "adversary with one row of initial weights per batch element."
            )
        if transforms_active():
            raise RuntimeError(
                "OnlineGroupDRO.step updates the group weights in place and "
                "cannot run inside a torch.func transform; call it outside the "
                "transformed function."
            )
        with torch.no_grad():
            dtype = working_dtype(group_loss, self.log_weights)
            ascent = self.log_weights.to(dtype) + self._scaled(
                group_loss.detach().to(dtype)
            )
            self.log_weights.copy_(
                ascent - torch.logsumexp(ascent, dim=-1, keepdim=True)
            )

    def forward(self, group_loss: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the weighted loss $\sum_g q_g \hat\ell_g$ at the current weights.

        Args:
            group_loss: Group losses of shape `(..., G)`, broadcasting
                against the leading dimensions of `log_weights`.

        Returns:
            The weighted loss over the broadcast batch, computed in
            `optora.core.working_dtype(group_loss, log_weights)` and
            returned in the promoted dtype of `group_loss` and
            `log_weights`. Its gradient with respect to `group_loss` is the
            weights.

        Raises:
            ValueError: If `group_loss` does not have shape `(..., G)`.
        """
        self._check_group_loss(group_loss)
        dtype = working_dtype(group_loss, self.log_weights)
        weights = torch.exp(self.log_weights.to(dtype))
        value = torch.sum(weights * group_loss.to(dtype), dim=-1)
        return value.to(torch.promote_types(group_loss.dtype, self.log_weights.dtype))
