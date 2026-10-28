r"""Shared tolerance for checking that a distribution lies inside a divergence ball.

A worst-case distribution $q^\star$ of a ball $\{q : D(q \,\|\, p) \le \rho\}$
sits on its boundary below saturation, so a computed $D(q^\star \,\|\, p)$
may exceed $\rho$ by the rounding of its own evaluation. That rounding has
two parts:

- a relative part, proportional to $\rho$;
- an absolute floor of about $n \varepsilon$, with $n$ the support size and
  $\varepsilon$ the machine epsilon, because the per-scenario terms of $D$
  (for example $\log(q_i / p_i)$) cancel to roughly zero when $q$ is close
  to $p$.

At a vanishing radius the relative slack alone is sub-ulp, so a check
without the floor passes or fails depending on the kernels of the platform
it runs on. Every feasibility check in `tests/dro/` uses `feasibility_bound`
so that the floor cannot be forgotten.
"""

from typing import Final

import torch

# Relative slack on the radius: the 1e-9 float64 accuracy target of plan
# milestone 8a.
FEASIBILITY_RTOL: Final = 1e-9


def feasibility_bound(
    radius: float,
    support_size: int,
    dtype: torch.dtype = torch.float64,
    rtol: float = FEASIBILITY_RTOL,
) -> float:
    r"""Return the largest computed divergence a feasible distribution may reach.

    The bound is $\rho (1 + \mathrm{rtol}) + n \varepsilon$, where $\rho$ is
    the radius, $n$ the support size, and $\varepsilon$ the machine epsilon
    of `dtype`.

    Args:
        radius: Nonnegative radius of the divergence ball.
        support_size: Number of scenarios $n$ the divergence sums over.
        dtype: Floating dtype the divergence was evaluated in.
        rtol: Relative slack on `radius`.

    Returns:
        The upper bound a computed divergence of a distribution inside the
        ball must not exceed.
    """
    return radius * (1.0 + rtol) + support_size * torch.finfo(dtype).eps
