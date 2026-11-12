"""Group DRO.

Average the per-sample losses within each group with `group_losses`, and
train on the worst group in two ways: exactly on the full batch with
`CVaRAmbiguitySet(alpha=0)`, whose worst case is the largest group loss,
and on minibatches with the exponentiated-gradient adversary
`OnlineGroupDRO`.
"""

import torch
from torch import nn

from optora import CVaRAmbiguitySet, OnlineGroupDRO
from optora.dro import group_losses

torch.set_default_dtype(torch.float64)

# Group 0 has 180 points on slope +1, group 1 has 20 points on slope -1.
generator = torch.Generator().manual_seed(0)
features = torch.randn(200, 1, generator=generator)
groups = (torch.arange(200) >= 180).long()
targets = torch.where(groups == 1, -1.0, 1.0) * features[:, 0]
targets = targets + 0.1 * torch.randn(200, generator=generator)
proportions = torch.bincount(groups, minlength=2) / 200


def squared_error(model: nn.Module, index: torch.Tensor) -> torch.Tensor:
    """Return the per-sample squared error of `model` on the rows `index`."""
    return (model(features[index]).squeeze(-1) - targets[index]) ** 2


def train(method: str) -> nn.Linear:
    """Train a linear model by ERM, exact group DRO or online group DRO."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    worst_group = CVaRAmbiguitySet(proportions, alpha=0.0)
    adversary = OnlineGroupDRO(torch.full((2,), 0.5), step_size=0.05)
    everything = torch.arange(200)
    for _ in range(400):
        optimizer.zero_grad()
        if method == "online":
            # 16 samples from each group, so both groups are in every batch.
            index = torch.cat(
                [
                    torch.randint(180, (16,), generator=generator),
                    180 + torch.randint(20, (16,), generator=generator),
                ]
            )
            group_loss = group_losses(squared_error(model, index), groups[index], 2)
            adversary.step(group_loss)
            objective = adversary(group_loss)
        else:
            group_loss = group_losses(squared_error(model, everything), groups, 2)
            if method == "erm":
                objective = proportions @ group_loss
            else:
                objective = worst_group.worst_case_expectation(group_loss)
        objective.backward()
        optimizer.step()
    return model


# The worst case of a CVaR ball at alpha = 0 is the largest group loss.
example_loss = torch.tensor([0.3, 1.7])
torch.testing.assert_close(
    CVaRAmbiguitySet(proportions, alpha=0.0).worst_case_expectation(example_loss),
    example_loss.max(),
)

print("method   group 0 loss   group 1 loss")
worst = {}
for method in ("erm", "exact", "online"):
    model = train(method)
    with torch.no_grad():
        loss = group_losses(squared_error(model, torch.arange(200)), groups, 2)
    worst[method] = loss.max()
    print(f"{method:<7}  {loss[0]:12.4f}  {loss[1]:13.4f}")

assert worst["exact"] < worst["erm"] and worst["online"] < worst["erm"]
