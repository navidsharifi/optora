"""KL ball.

Evaluate `KLAmbiguitySet.worst_case_expectation` for a sweep of radii in
one batched call, check its two closed-form limits, and read the
worst-case distribution off the gradient with respect to the loss.
"""

import torch

from optora import KLAmbiguitySet, KLDivergence

torch.set_default_dtype(torch.float64)

nominal = torch.full((4,), 0.25)
loss = torch.tensor([1.0, 2.0, 3.0, 10.0])

# A tensor radius sweeps several balls at once. The loss is expanded so
# that each row is solved against its own radius.
radii = torch.tensor([0.0, 0.01, 0.1, 1.0, 10.0])
ambiguity_set = KLAmbiguitySet(nominal, radius=radii)
values = ambiguity_set.worst_case_expectation(loss.expand(len(radii), -1))
for radius, value in zip(radii, values, strict=True):
    print(f"radius {radius:5.2f}: worst case {value:.4f}")

# Radius zero gives the nominal mean, and a radius of at least
# log(1 / nominal of the worst scenario) gives the largest loss.
torch.testing.assert_close(values[0], (nominal * loss).sum())
torch.testing.assert_close(values[-1], loss.max())
assert (values.diff() >= 0).all()

# By Danskin's theorem the gradient of the worst case with respect to the
# loss is the worst-case distribution q*.
loss = loss.clone().requires_grad_()
KLAmbiguitySet(nominal, radius=0.1).worst_case_expectation(loss).backward()
worst_case = loss.grad
print(f"worst-case distribution at radius 0.1: {worst_case}")

# Below saturation the constraint is active, KL(q* || nominal) = radius.
torch.testing.assert_close(KLDivergence()(worst_case, nominal), torch.tensor(0.1))
