"""Phi-divergences.

Evaluate the KL, chi-square and total-variation divergences, define a new
one (the squared Hellinger distance) from its convex generator with
`PhiDivergence`, and compare a whole batch of candidates in one call.
"""

import torch

from optora import (
    ChiSquareDivergence,
    KLDivergence,
    PhiDivergence,
    TotalVariationDivergence,
)

torch.set_default_dtype(torch.float64)

# Two distributions on the same four-point support.
p = torch.tensor([0.1, 0.2, 0.3, 0.4])
q = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Each divergence is an `nn.Module` called as D(p, q) = D(p || q).
divergences = {
    "KL": KLDivergence(),
    "chi-square": ChiSquareDivergence(),
    "total variation": TotalVariationDivergence(),
}
for name, divergence in divergences.items():
    print(f"{name:>16}: {divergence(p, q):.6f}")

# The same numbers from the definitions, sum_i q_i phi(p_i / q_i).
ratio = p / q
torch.testing.assert_close(KLDivergence()(p, q), (p * ratio.log()).sum())
torch.testing.assert_close(ChiSquareDivergence()(p, q), (q * (ratio - 1) ** 2).sum())
torch.testing.assert_close(
    TotalVariationDivergence()(p, q), 0.5 * (q * (ratio - 1).abs()).sum()
)

# Any convex generator with phi(1) = 0 defines a new divergence. The
# generator (sqrt(t) - 1)^2 gives the squared Hellinger distance.
hellinger = PhiDivergence(phi=lambda t: (t.sqrt() - 1) ** 2)
print(f"{'Hellinger^2':>16}: {hellinger(p, q):.6f}")
torch.testing.assert_close(hellinger(p, q), ((p.sqrt() - q.sqrt()) ** 2).sum())

# Leading dimensions are batch dimensions: here, 1000 random candidates
# against the same reference, in one vectorized call.
candidates = torch.distributions.Dirichlet(torch.ones(4)).sample((1000,))
values = KLDivergence()(candidates, q)
print(
    f"KL of 1000 random candidates: shape {tuple(values.shape)}, "
    f"min {values.min():.4f}, max {values.max():.4f}"
)
assert (values >= 0).all()
