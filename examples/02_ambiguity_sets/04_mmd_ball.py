"""MMD ball.

Solve the worst case over an `MMDAmbiguitySet` built from a Gaussian
kernel. A KL ball can only reweight the scenarios the nominal already
contains. The MMD ball can also move probability onto a scenario the
nominal gives no mass, at a price set by the kernel.
"""

import torch

from optora import AmbiguitySet, KLAmbiguitySet, MMDAmbiguitySet, MMDDivergence

torch.set_default_dtype(torch.float64)

# Eleven scenarios on a line. The nominal only covers the first ten, and
# the loss is largest on the eleventh, which the sample never contains.
x = torch.linspace(0.0, 1.0, 11)
gram = torch.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.2**2))
nominal = torch.cat([torch.full((10,), 0.1), torch.zeros(1)])
loss = x**2


def worst_case_distribution(
    ambiguity_set: AmbiguitySet, loss: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the worst case and its maximizer, the gradient in `loss`."""
    loss = loss.clone().requires_grad_()
    value = ambiguity_set.worst_case_expectation(loss)
    (distribution,) = torch.autograd.grad(value, loss)
    return value.detach(), distribution


for radius in (0.05, 0.1, 0.2):
    _, kl_q = worst_case_distribution(KLAmbiguitySet(nominal, radius=radius), loss)
    mmd_value, mmd_q = worst_case_distribution(
        MMDAmbiguitySet(nominal, gram, radius=radius), loss
    )
    print(
        f"radius {radius:.2f}: MMD worst case {mmd_value:.4f}, mass on the "
        f"unseen scenario: KL {kl_q[-1]:.3f}, MMD {mmd_q[-1]:.3f}"
    )
    assert kl_q[-1] == 0
    assert mmd_q[-1] > 0

# The worst-case distribution sits on the boundary of the ball.
torch.testing.assert_close(
    MMDDivergence(gram)(mmd_q, nominal), torch.tensor(0.2), rtol=1e-6, atol=0.0
)
