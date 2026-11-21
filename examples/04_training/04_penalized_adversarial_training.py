"""Penalized adversarial training.

Train a classifier on the penalized Wasserstein objective of Sinha,
Namkoong and Duchi (2018), in which an adversary may move every input at a
squared Euclidean cost. `transport_map` finds the moved inputs and
`squared_distance` measures how far they moved, which gives the radius of
the Wasserstein ball that the reported worst-case loss refers to.
"""

from collections.abc import Callable

import torch
from torch import nn

from optora.dro import squared_distance, transport_map

torch.set_default_dtype(torch.float64)

generator = torch.Generator().manual_seed(0)
penalty = 2.0


def draw(num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw points of the synthetic task of Sinha et al.

    The label is the side of the circle of radius sqrt(2) a Gaussian point
    lies on, and points within a margin around the circle are rejected.
    """
    points = torch.randn(4 * num_points, 2, generator=generator)
    norms = points.norm(dim=1)
    keep = (norms < 2**0.5 / 1.3) | (norms > 1.3 * 2**0.5)
    return points[keep][:num_points], (norms[keep][:num_points] > 2**0.5).long()


def per_example_loss(
    model: nn.Module, labels: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the cross-entropy of `model` on each example, as a function of inputs."""
    return lambda inputs: nn.functional.cross_entropy(
        model(inputs), labels, reduction="none"
    )


def certificate(
    loss: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the transport cost, the worst-case loss and the surrogate value."""
    with torch.no_grad():
        transported = transport_map(loss, inputs, penalty, num_steps)
        radius = squared_distance(transported, inputs).mean()
        worst_case = loss(transported).mean()
    return radius, worst_case, worst_case - penalty * radius


# A loss that is linear in the inputs is maximized by the first ascent step.
slope = torch.tensor([[1.0, -2.0]])
torch.testing.assert_close(
    transport_map(lambda z: (z * slope).sum(1), torch.zeros(1, 2), penalty),
    slope / (2 * penalty),
)

features, labels = draw(1000)
test_features, test_labels = draw(1000)
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 4), nn.ELU(), nn.Linear(4, 2), nn.ELU(), nn.Linear(2, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = per_example_loss(model, labels)
initial = certificate(loss, features, 16)[2]

for _ in range(300):
    transported = transport_map(loss, features, penalty)
    optimizer.zero_grad()
    loss(transported).mean().backward()
    optimizer.step()

# Any point gives a lower bound on the surrogate, since it is a supremum,
# so more ascent steps tighten the estimate from below.
print("data   steps   radius   worst-case loss   surrogate")
rows = [
    ("train", features, labels, 16),
    ("train", features, labels, 400),
    ("test", test_features, test_labels, 16),
]
for name, inputs, targets, num_steps in rows:
    radius, worst_case, value = certificate(
        per_example_loss(model, targets), inputs, num_steps
    )
    print(
        f"{name:<5}  {num_steps:5d}  {radius:7.4f}  {worst_case:16.4f}  {value:10.4f}"
    )

assert certificate(loss, features, 16)[2] < initial
