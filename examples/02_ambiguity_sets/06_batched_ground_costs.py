"""Batched ground costs.

Solve the Wasserstein worst case of several datasets in one call. Each
dataset has its own support points, so each needs its own ground cost, and
`WassersteinAmbiguitySet` takes them stacked as one tensor of shape
`(..., n, n)`. The batched values are checked against one set per dataset
and against the closed form for a Lipschitz loss, and a batch of radii is
then solved for every dataset at once.
"""

import torch

from optora import WassersteinAmbiguitySet

torch.set_default_dtype(torch.float64)

# Five datasets of forty points on the real line. Dataset i has the ground
# cost |x_ik - x_il| between its own points, so the costs form a batch of
# shape (5, 40, 40).
generator = torch.Generator().manual_seed(0)
num_datasets, num_samples = 5, 40
x = torch.randn(num_datasets, num_samples, generator=generator)
cost = (x[:, :, None] - x[:, None, :]).abs()
nominal = torch.full((num_samples,), 1.0 / num_samples)
loss = 2.0 * x

# One set holds all five costs, and one call solves all five worst cases.
batched = WassersteinAmbiguitySet(nominal, cost, radius=0.1)
values = batched.worst_case_expectation(loss)
print(f"batched worst cases : {[round(v, 4) for v in values.tolist()]}")

# The same numbers come from one set per dataset.
for i in range(num_datasets):
    single = WassersteinAmbiguitySet(nominal, cost[i], radius=0.1)
    torch.testing.assert_close(values[i], single.worst_case_expectation(loss[i]))

# The loss 2 x has Lipschitz constant 2 in each ground metric, so every
# worst case is the mean of its own dataset plus 2 * radius.
torch.testing.assert_close(values, loss.mean(dim=-1) + 2.0 * 0.1)

# A radius sweep is one more batch dimension. The cost gets a singleton
# axis for it, and the result has one row per dataset and one column per
# radius.
radii = torch.tensor([0.0, 0.05, 0.1])
sweep = WassersteinAmbiguitySet(nominal, cost[:, None], radius=radii)
grid = sweep.worst_case_expectation(loss[:, None])
print(f"datasets x radii    : {tuple(grid.shape)}")
torch.testing.assert_close(grid, loss.mean(dim=-1, keepdim=True) + 2.0 * radii)
