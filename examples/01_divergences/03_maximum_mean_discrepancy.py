r"""Maximum mean discrepancy.

Build a Gaussian-kernel Gram matrix on a finite support, evaluate
`MMDDivergence`, and check it against the closed form
$\sqrt{(p - q)^\top K (p - q)}$. Unlike a phi-divergence, the MMD stays
finite between distributions with disjoint supports.
"""

import torch

from optora import KLDivergence, MMDDivergence

torch.set_default_dtype(torch.float64)

# Ten support points in the plane and a Gaussian kernel of bandwidth 1.
generator = torch.Generator().manual_seed(0)
support = torch.randn(10, 2, generator=generator)
gram = torch.exp(-(torch.cdist(support, support) ** 2) / 2.0)

mmd = MMDDivergence(gram)

p = torch.full((10,), 0.1)
q = torch.softmax(torch.randn(10, generator=generator), dim=0)
difference = p - q
print(f"MMD(p, q)            : {mmd(p, q):.6f}")
torch.testing.assert_close(mmd(p, q), (difference @ gram @ difference).sqrt())

# Mass on the first five points against mass on the last five. The KL
# divergence is infinite in exact arithmetic, because `right` puts mass where
# `left` has none. `KLDivergence` clamps the reference at `eps = 1e-12`, so
# it reports about log(0.2 / 1e-12) = 26 instead. MMD needs no clamp.
left = torch.cat([torch.full((5,), 0.2), torch.zeros(5)])
right = left.flip(0)
print(f"MMD, disjoint support: {mmd(left, right):.6f}")
print(f"KL,  disjoint support: {KLDivergence()(right, left):.3g}")

# MMD is a metric when the Gram matrix is positive definite, so it is
# symmetric and satisfies the triangle inequality.
torch.testing.assert_close(mmd(p, q), mmd(q, p))
assert mmd(left, right) <= mmd(left, p) + mmd(p, right)
