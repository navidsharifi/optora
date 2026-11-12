"""Chi-square, total-variation and CVaR balls.

Solve the worst case over three phi-divergence balls on the same loss,
check each one against its closed form, and rebuild the KL ball from a
general generator with `PhiAmbiguitySet`.
"""

import torch

from optora import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    KLAmbiguitySet,
    PhiAmbiguitySet,
    PhiDivergence,
    TotalVariationAmbiguitySet,
)
from optora.dro import kl_conjugate

torch.set_default_dtype(torch.float64)

nominal = torch.full((4,), 0.25)
loss = torch.tensor([1.0, 2.0, 3.0, 10.0])
mean = (nominal * loss).sum()
variance = (nominal * (loss - mean) ** 2).sum()

# Chi-square ball: below saturation the worst case is mean + sqrt(rho Var).
chi_square = ChiSquareAmbiguitySet(nominal, radius=0.1).worst_case_expectation(loss)
print(f"chi-square, radius 0.1 : {chi_square:.4f}")
torch.testing.assert_close(chi_square, mean + (0.1 * variance).sqrt())

# Total-variation ball: the adversary moves mass `radius` from the
# smallest loss to the largest one.
total_variation = TotalVariationAmbiguitySet(nominal, radius=0.1)
tv = total_variation.worst_case_expectation(loss)
print(f"total variation, 0.1   : {tv:.4f}")
torch.testing.assert_close(tv, mean + 0.1 * (loss.max() - loss.min()))

# CVaR at level alpha: the mean of the worst alpha-fraction of outcomes.
cvar = CVaRAmbiguitySet(nominal, alpha=0.5).worst_case_expectation(loss)
print(f"CVaR, alpha 0.5        : {cvar:.4f}")
torch.testing.assert_close(cvar, loss.sort().values[-2:].mean())

# Any convex generator gives a ball, given the conjugate of the generator.
# With phi(t) = t log t - t + 1 and its conjugate e^s - 1 this is KL again.
kl_generator = PhiDivergence(phi=lambda t: torch.special.xlogy(t, t) - t + 1)
general = PhiAmbiguitySet(
    nominal, divergence=kl_generator, radius=0.1, phi_conjugate=kl_conjugate
)
value = general.worst_case_expectation(loss)
print(f"PhiAmbiguitySet as KL  : {value:.4f}")
torch.testing.assert_close(
    value, KLAmbiguitySet(nominal, radius=0.1).worst_case_expectation(loss)
)
