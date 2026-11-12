"""Sinkhorn divergence.

Build a ground cost on a one-dimensional grid, evaluate the debiased
`SinkhornDivergence` for decreasing entropic strength, and watch it
approach the exact type-1 Wasserstein distance, which on the real line is
the area between the two cumulative distribution functions.
"""

import torch

from optora import SinkhornDivergence

torch.set_default_dtype(torch.float64)

# Two bumps on a grid of 50 points in [0, 1].
x = torch.linspace(0.0, 1.0, 50)
p = torch.exp(-((x - 0.3) ** 2) / 0.01)
q = torch.exp(-((x - 0.6) ** 2) / 0.02)
p, q = p / p.sum(), q / q.sum()

# Pairwise ground cost |x_i - x_j|, shared by both distributions.
cost = (x[:, None] - x[None, :]).abs()

# On the real line W_1(p, q) = integral of |F_p - F_q|.
spacing = x[1] - x[0]
exact = (p.cumsum(0) - q.cumsum(0)).abs().sum() * spacing
print(f"exact W_1            : {exact:.6f}")

for epsilon in (1e-1, 1e-2, 1e-3):
    divergence = SinkhornDivergence(cost, epsilon=epsilon, max_iter=2000, tol=1e-9)
    value = divergence(p, q)
    print(f"Sinkhorn, eps = {epsilon:.0e}: {value:.6f}")

# Debiasing makes the divergence vanish between identical distributions.
divergence = SinkhornDivergence(cost, epsilon=1e-2, max_iter=2000, tol=1e-9)
print(f"S(p, p)              : {divergence(p, p):.2e}")
torch.testing.assert_close(divergence(p, q), divergence(q, p))
torch.testing.assert_close(
    SinkhornDivergence(cost, epsilon=1e-3, max_iter=2000, tol=1e-9)(p, q),
    exact,
    rtol=2e-2,
    atol=0.0,
)
