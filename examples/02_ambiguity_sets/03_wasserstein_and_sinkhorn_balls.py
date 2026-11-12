"""Wasserstein and Sinkhorn balls.

Solve the worst case over a type-1 `WassersteinAmbiguitySet`, where the
adversary moves mass along a ground cost instead of reweighting it, and
check the closed form for a Lipschitz loss. Then solve the entropic
relaxation `SinkhornAmbiguitySet` and watch it approach the exact value as
the entropic strength shrinks.
"""

import torch

from optora import SinkhornAmbiguitySet, WassersteinAmbiguitySet

torch.set_default_dtype(torch.float64)

# A grid on [0, 1] with the cost |x_i - x_j|, and a nominal spread
# uniformly over the points between 0.2 and 0.4.
x = torch.linspace(0.0, 1.0, 41)
cost = (x[:, None] - x[None, :]).abs()
nominal = ((x >= 0.2) & (x <= 0.4)).to(x.dtype)
nominal = nominal / nominal.sum()

# The loss 2 x has Lipschitz constant 2 in the ground metric. Moving mass a
# distance d to the right raises the loss by 2 d, so as long as there is room
# to move, the worst case is the nominal mean plus 2 * radius.
loss = 2.0 * x
mean = (nominal * loss).sum()
for radius in (0.0, 0.05, 0.1):
    wasserstein = WassersteinAmbiguitySet(nominal, cost, radius=radius)
    value = wasserstein.worst_case_expectation(loss)
    print(f"Wasserstein, radius {radius:.2f}: {value:.4f}")
    torch.testing.assert_close(value, mean + 2.0 * radius)

# The Sinkhorn ball smooths the transport plan by an entropic term. Its
# worst case is smooth in the loss, and it tends to the Wasserstein value as
# the entropic strength epsilon goes to zero.
exact = WassersteinAmbiguitySet(nominal, cost, radius=0.1).worst_case_expectation(loss)
for epsilon in (1e-1, 1e-2, 1e-3):
    sinkhorn = SinkhornAmbiguitySet(nominal, cost, epsilon=epsilon, radius=0.1)
    value = sinkhorn.worst_case_expectation(loss)
    print(f"Sinkhorn, eps {epsilon:.0e}, radius 0.10: {value:.4f}")
torch.testing.assert_close(value, exact, rtol=1e-2, atol=0.0)
