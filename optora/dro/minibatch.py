r"""Minibatch estimators of the worst-case expected loss under a population nominal.

Every `optora.core.AmbiguitySet` evaluates the worst case over a ball around a
*fixed, finite* nominal. In stochastic training the nominal is instead an
unknown population distribution $P_0$ seen only through minibatches of
i.i.d. samples, and the plug-in estimate, the worst case over the ball around
the empirical distribution $\hat P_m$ of one minibatch,

$$
\hat V_m = \sup_{q:\, D(q \,\|\, \hat P_m) \,\le\, \rho} \mathbb{E}_q[\ell],
$$

is a biased estimate of the population worst case $V$: the supremum is a
nonlinear functional of the sample, so $\mathbb{E}[\hat V_m] \neq V$ and the
gradient of $\hat V_m$ is not an unbiased stochastic gradient of $V$
(Levy, Carmon, Duchi, and Sidford 2020). For KL-DRO the plug-in is the
log-sum-exp of a subsample.

This module provides the two standard remedies:

- `MultilevelMonteCarloEstimator` keeps the exact plug-in solve of any
  ambiguity set and removes its bias up to the largest batch size by
  randomized multilevel Monte Carlo (Blanchet and Glynn 2015; Levy et al.
  2020), at a logarithmic increase in the expected batch size.
- `StochasticPhiDual` keeps the dual variables of a phi-divergence ball as
  learnable parameters, which turns the worst case into a plain expectation
  over $P_0$ whose minibatch mean is exactly unbiased (Ben-Tal et al. 2013;
  Levy et al. 2020; Jin, Zhang, Wang, and Wang 2021).
"""

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from optora.core.dro_base import AmbiguitySet
from optora.core.precision import working_dtype


class MultilevelMonteCarloEstimator(nn.Module):
    r"""Randomized multilevel Monte Carlo estimator of a minibatch worst case.

    Let $\hat V_m$ be the plug-in worst case of an ambiguity set centered on
    the uniform distribution over $m$ i.i.d. samples. Its bias shrinks as
    $m$ grows, but evaluating a large $m$ on every step is what minibatching
    avoids. With a base batch size $n_0$ and a maximum level $L$, draw a
    level $J \in \{1, \dots, L\}$ with probability
    $p_J \propto 2^{-J}$, draw $n_0 2^J$ samples $S$, split them into halves
    $S^{(1)}, S^{(2)}$, and report

    $$
    \hat V = \hat V_{n_0}(S_{1:n_0})
        + \frac{1}{p_J}\Big(\hat V_{n_0 2^J}(S)
        - \tfrac{1}{2}\big(\hat V_{n_0 2^{J-1}}(S^{(1)})
        + \hat V_{n_0 2^{J-1}}(S^{(2)})\big)\Big).
    $$

    Each term has the distribution of a plug-in estimate on that many
    i.i.d. samples, so the expectation telescopes:

    $$
    \mathbb{E}[\hat V] = \mathbb{E}[\hat V_{n_0}]
        + \sum_{j=1}^{L} \big(\mathbb{E}[\hat V_{n_0 2^j}]
        - \mathbb{E}[\hat V_{n_0 2^{j-1}}]\big)
        = \mathbb{E}[\hat V_{n_0 2^L}],
    $$

    and the same holds for the gradient, since every term is differentiated
    at its own maximizer (Danskin). The estimator therefore has the bias of
    a batch of $n_0 2^L$ samples at an expected batch size of
    $n_0 L / (1 - 2^{-L})$ (Blanchet and Glynn 2015; Levy et al. 2020). The
    antithetic halves keep its variance bounded as $L$ grows: the
    correction is the gap between one solve and the mean of two solves on
    the same samples, which vanishes as the plug-in converges. With
    `max_level=0` there is no correction and the estimator is the plain
    plug-in on $n_0$ samples.

    Usage is a training loop whose batch size is drawn per step:

    ```python
    estimator = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=0.1),
        base_batch_size=32,
        max_level=4,
    )
    batch_size = estimator.sample_batch_size()
    loss = per_sample_loss(model, next_batch(batch_size))
    estimator(loss).backward()
    ```

    Attributes:
        ambiguity_sets: One ambiguity set per level `j = 0, ..., max_level`,
            centered on the uniform distribution over `base_batch_size *
            2 ** j` samples. Registered as submodules, so `.to(...)` moves
            and casts every level together.
        base_batch_size: Number of samples $n_0$ of the base level.
        max_level: Largest level $L$; the bias is that of a batch of
            `base_batch_size * 2 ** max_level` samples.
    """

    def __init__(
        self,
        ambiguity_set_factory: Callable[[torch.Tensor], AmbiguitySet],
        base_batch_size: int,
        max_level: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the estimator and build one ambiguity set per level.

        Args:
            ambiguity_set_factory: Map from a nominal distribution to the
                ambiguity set centered on it, for example
                `functools.partial(KLAmbiguitySet, radius=0.1)`. It is called
                once per level with a uniform nominal on `device` in
                `dtype`.
                Any set whose construction depends only on the nominal
                qualifies (KL, chi-square, total variation, general phi); a
                Wasserstein set also needs the samples' ground cost and does
                not.
            base_batch_size: Number of samples $n_0$ of the base level, at
                least one.
            max_level: Largest level $L$, at least zero.
            device: Device of the uniform nominals, the current default
                device when `None`.
            dtype: Dtype of the uniform nominals, the default dtype when
                `None`. Build in the dtype the set will run in rather than
                casting afterwards: `1 / m` rounded to float32 and cast to
                float64 is not the float64 uniform distribution.

        Raises:
            ValueError: If `base_batch_size` is less than one or `max_level`
                is negative.
        """
        super().__init__()
        if base_batch_size < 1:
            raise ValueError(
                f"base_batch_size must be at least one, got {base_batch_size}."
            )
        if max_level < 0:
            raise ValueError(f"max_level must be nonnegative, got {max_level}.")
        self.base_batch_size = base_batch_size
        self.max_level = max_level
        sizes: list[int] = [
            base_batch_size * 2**level for level in range(max_level + 1)
        ]
        self.ambiguity_sets = nn.ModuleList(
            ambiguity_set_factory(
                torch.full((size,), 1.0 / size, device=device, dtype=dtype)
            )
            for size in sizes
        )
        weights = [2.0**-level for level in range(1, max_level + 1)]
        self._level_probabilities = [weight / sum(weights) for weight in weights]
        self._sampled_sizes = sizes[1:]
        sampled_levels = range(1, max_level + 1) if max_level > 0 else (0,)
        self._level_of_size = {sizes[level]: level for level in sampled_levels}

    @property
    def expected_batch_size(self) -> float:
        r"""Mean number of samples a step draws, $n_0 L / (1 - 2^{-L})$ for $L > 0$."""
        if self.max_level == 0:
            return float(self.base_batch_size)
        return sum(
            probability * size
            for probability, size in zip(
                self._level_probabilities, self._sampled_sizes, strict=True
            )
        )

    def sample_batch_size(self, generator: torch.Generator | None = None) -> int:
        r"""Draw the batch size $n_0 2^J$ of the next step.

        The level is drawn on the CPU, so drawing it never waits on an
        accelerator; the returned size is what the data pipeline needs on
        the host anyway.

        Args:
            generator: CPU random number generator to draw the level with,
                for reproducible runs. Defaults to PyTorch's global one.

        Returns:
            `base_batch_size * 2 ** J` for a level `J` drawn from
            `{1, ..., max_level}` with probability proportional to
            `2 ** -J`, or `base_batch_size` when `max_level` is zero.
        """
        if self.max_level == 0:
            return self.base_batch_size
        level = torch.multinomial(
            torch.tensor(self._level_probabilities, dtype=torch.float64),
            num_samples=1,
            generator=generator,
        )
        return self._sampled_sizes[int(level)]

    def _ambiguity_set(self, level: int) -> AmbiguitySet:
        """Return the ambiguity set of `level`, typed as an `AmbiguitySet`."""
        return cast(AmbiguitySet, self.ambiguity_sets[level])

    def _plug_in(self, level: int, loss: torch.Tensor) -> torch.Tensor:
        """Return the plug-in worst case of `level` on its `(..., m)` loss."""
        return self._ambiguity_set(level).worst_case_expectation(loss)

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        """Estimate the worst-case expected loss from one multilevel batch.

        Args:
            loss: Per-sample losses of shape `(..., m)` for `m` i.i.d.
                samples, where `m` is a size `sample_batch_size` can return.
                The level is read off `m`. Leading dimensions are
                independent batches, broadcast against a tensor radius as in
                `AmbiguitySet.worst_case_expectation`.

        Returns:
            The multilevel estimate, of the batch shape the ambiguity sets
            return for a `(..., m)` loss. Its expectation and the
            expectation of its gradient are those of the plug-in worst case
            on `base_batch_size * 2 ** max_level` samples.

        Raises:
            ValueError: If `loss` is zero-dimensional or its trailing size is
                not one `sample_batch_size` can return.
        """
        size = loss.shape[-1] if loss.ndim > 0 else 0
        level = self._level_of_size.get(size)
        if level is None:
            raise ValueError(
                "loss must have shape (..., m) with m in "
                f"{sorted(self._level_of_size)}, got {tuple(loss.shape)}."
            )
        base = self._plug_in(0, loss[..., : self.base_batch_size])
        if level == 0:
            return base
        full = self._plug_in(level, loss)
        radius = self._ambiguity_set(0).radius
        radius_rank = radius.ndim if isinstance(radius, torch.Tensor) else 0
        # The halves index goes in front of every batch dimension, padded so a
        # tensor radius still aligns with the loss's batch dimensions.
        halves = loss.reshape(*loss.shape[:-1], 2, size // 2).movedim(-2, 0)
        halves = halves[(slice(None),) + (None,) * max(0, radius_rank - loss.ndim + 1)]
        correction = full - torch.mean(self._plug_in(level - 1, halves), dim=0)
        return base + correction / self._level_probabilities[level - 1]


class StochasticPhiDual(nn.Module):
    r"""Phi-DRO dual objective with learnable dual variables, unbiased on minibatches.

    The worst case over a phi-divergence ball around a population $P_0$ has
    the convex dual (Ben-Tal et al. 2013; Duchi and Namkoong 2021)

    $$
    \sup_{q:\, D_\phi(q \,\|\, P_0) \,\le\, \rho} \mathbb{E}_q[\ell]
    = \inf_{\eta > 0,\, \lambda \in \mathbb{R}}
        \mathbb{E}_{X \sim P_0}\!\left[\eta \rho + \lambda
        + \eta\, \phi^*\!\left(\frac{\ell(X) - \lambda}{\eta}\right)\right],
    $$

    an expectation over $P_0$ for fixed $(\eta, \lambda)$. Minimizing it
    *jointly* with the model parameters $\theta$ therefore needs only
    unbiased estimates of that expectation, and the minibatch mean is one:
    `forward` is linear in the empirical distribution, so its expectation
    over a minibatch, and that of its gradient with respect to $\theta$,
    $\log \eta$, and $\lambda$, is the population dual exactly. This is the
    dual stochastic-gradient approach to large-scale phi-DRO (Levy et al.
    2020; Jin et al. 2021, who analyze it for non-convex losses).

    At any $(\eta, \lambda)$ the population dual bounds the worst case from
    above (weak duality); it is tight at the joint minimizer. The dual
    variables are `nn.Parameter`s, so they train with the model:

    ```python
    dual = StochasticPhiDual(radius=0.1, phi_conjugate=torch.expm1)  # KL
    optimizer = torch.optim.Adam([*model.parameters(), *dual.parameters()])
    dual(per_sample_loss(model, batch)).backward()
    optimizer.step()
    ```

    The KL conjugate is $\phi^*(s) = e^s - 1$, `torch.expm1`; the chi-square
    conjugate is `optora.dro.chi_square_conjugate`. The price of
    unbiasedness is that $(\eta, \lambda)$ are reached by stochastic
    optimization rather than solved per batch: a conjugate that grows
    exponentially, as KL's does, makes the objective's variance sensitive to
    $\lambda$ trailing the loss scale, and nothing here standardizes the
    loss, since a per-batch shift or scale would reintroduce the bias.
    Initialize `initial_lam` near the typical loss and `initial_log_eta`
    near the log of its spread.

    Attributes:
        radius: Nonnegative phi-divergence radius $\rho$, a float or a
            tensor of radii, each with its own dual variables.
        phi_conjugate: Convex conjugate $\phi^*$ of the generator restricted
            to $t \ge 0$, finite on the real line.
        log_eta: Learnable $\log \eta$, of `radius`'s shape.
        lam: Learnable $\lambda$, in the units of the loss, of `radius`'s
            shape.
    """

    radius: float | torch.Tensor

    def __init__(
        self,
        radius: float | torch.Tensor,
        phi_conjugate: Callable[[torch.Tensor], torch.Tensor],
        initial_log_eta: float = 0.0,
        initial_lam: float = 0.0,
    ) -> None:
        r"""Initialize the stochastic dual and its dual variables.

        Args:
            radius: Nonnegative phi-divergence radius. A tensor radius is
                registered as a buffer and gives every entry its own
                `log_eta` and `lam`, in its dtype and on its device, so a
                radius sweep trains in one module.
            phi_conjugate: Convex conjugate of the generator restricted to
                $t \ge 0$, finite on the real line, for example
                `torch.expm1` for KL.
            initial_log_eta: Starting value of `log_eta`.
            initial_lam: Starting value of `lam`, in the units of the loss.

        Raises:
            ValueError: If `radius` is a negative or `nan` float, or either
                initial value is not finite.
        """
        super().__init__()
        if not (math.isfinite(initial_log_eta) and math.isfinite(initial_lam)):
            raise ValueError(
                "initial_log_eta and initial_lam must be finite, got "
                f"{initial_log_eta} and {initial_lam}."
            )
        if isinstance(radius, torch.Tensor):
            self.register_buffer("radius", radius)
            reference = radius
        else:
            if not radius >= 0:
                raise ValueError(f"radius must be nonnegative, got {radius}.")
            self.radius = radius
            reference = torch.tensor(0.0)
        self.phi_conjugate = phi_conjugate
        self.log_eta = nn.Parameter(torch.full_like(reference, initial_log_eta))
        self.lam = nn.Parameter(torch.full_like(reference, initial_lam))

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the dual objective on a minibatch of per-sample losses.

        Args:
            loss: Per-sample losses of shape `(..., m)` for `m` i.i.d.
                samples. Leading dimensions are independent batches,
                broadcast against `radius`'s shape.

        Returns:
            $\eta \rho + \lambda + \eta \cdot \frac{1}{m} \sum_i
            \phi^*((\ell_i - \lambda) / \eta)$ per batch element, computed in
            `optora.core.working_dtype(loss, log_eta)` and returned in the
            promoted dtype of `loss` and `log_eta`.

        Raises:
            ValueError: If `loss` is zero-dimensional.
        """
        if loss.ndim == 0:
            raise ValueError(
                "loss must have shape (..., m) with a trailing sample dimension, "
                "got a scalar tensor."
            )
        dtype = working_dtype(loss, self.log_eta)
        log_eta = self.log_eta.to(dtype)
        lam = self.lam.to(dtype)
        eta = torch.exp(log_eta)
        scaled_shift = (loss.to(dtype) - lam.unsqueeze(-1)) / eta.unsqueeze(-1)
        conjugate_mean = torch.mean(self.phi_conjugate(scaled_shift), dim=-1)
        value = eta * self.radius + lam + eta * conjugate_mean
        return value.to(torch.promote_types(loss.dtype, self.log_eta.dtype))
