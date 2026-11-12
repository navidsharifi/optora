"""Choosing the radius.

Turn a sample size and a confidence level into a radius with
`chi_square_radius`, and check that the worst case over the resulting
`ChiSquareAmbiguitySet` is the upper end of the normal confidence interval
for the mean.
"""

import torch

from optora import ChiSquareAmbiguitySet
from optora.dro import chi_square_radius, kl_radius

torch.set_default_dtype(torch.float64)

generator = torch.Generator().manual_seed(0)
num_samples = 200
loss = torch.empty(num_samples).exponential_(generator=generator)
nominal = torch.full((num_samples,), 1.0 / num_samples)

# The calibrated radius is the chi-square quantile divided by n.
radius = chi_square_radius(num_samples, confidence=0.95)
print(f"chi-square radius for n = {num_samples}, 95%: {radius:.5f}")
print(f"KL radius for the same setting      : {kl_radius(num_samples, 0.95):.5f}")

upper = ChiSquareAmbiguitySet(nominal, radius=radius).worst_case_expectation(loss)

# The normal interval mean + z s / sqrt(n), with the plug-in standard
# deviation s of the sample.
mean = loss.mean()
std = loss.std(correction=0)
z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.975))
print(f"robust upper bound   : {upper:.5f}")
print(f"normal interval upper: {mean + z * std / num_samples**0.5:.5f}")
torch.testing.assert_close(upper, mean + z * std / num_samples**0.5)
