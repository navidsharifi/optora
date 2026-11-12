"""Training with torch.optim.

Fit an `nn.Linear` with Adam on the worst case over a KL ball by passing
the per-sample losses to `KLAmbiguitySet.worst_case_expectation`. Radius
zero is ordinary least squares. The data mix a large group with a small
one that follows a different slope, and a larger radius trades some
majority loss for a lower minority loss.
"""

import torch
from torch import nn

from optora import KLAmbiguitySet

torch.set_default_dtype(torch.float64)

# 180 points on slope +1 and 20 points on slope -1.
generator = torch.Generator().manual_seed(0)
features = torch.randn(200, 1, generator=generator)
minority = torch.arange(200) >= 180
slope = torch.where(minority, -1.0, 1.0)
targets = slope * features[:, 0] + 0.1 * torch.randn(200, generator=generator)
nominal = torch.full((200,), 1.0 / 200)


def fit(radius: float) -> nn.Linear:
    """Fit a linear model to the worst case over a KL ball of `radius`."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    ambiguity_set = KLAmbiguitySet(nominal, radius=radius)
    for _ in range(300):
        optimizer.zero_grad()
        # One loss per sample, not reduced: the ambiguity set reweights them.
        per_sample_loss = (model(features).squeeze(-1) - targets) ** 2
        ambiguity_set.worst_case_expectation(per_sample_loss).backward()
        optimizer.step()
    return model


print("radius   slope   majority loss   minority loss")
group_losses = []
for radius in (0.0, 0.1, 0.5):
    model = fit(radius)
    with torch.no_grad():
        loss = (model(features).squeeze(-1) - targets) ** 2
    majority_loss, minority_loss = loss[~minority].mean(), loss[minority].mean()
    group_losses.append((majority_loss, minority_loss))
    print(
        f"{radius:6.1f}  {model.weight.squeeze():6.3f}  "
        f"{majority_loss:14.4f}  {minority_loss:14.4f}"
    )

# Radius zero is ERM. A larger radius lowers the minority loss and raises
# the majority loss.
(erm_major, erm_minor), _, (dro_major, dro_minor) = group_losses
assert dro_minor < erm_minor and dro_major > erm_major
