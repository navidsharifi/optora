"""Minibatch estimators.

The worst case of a minibatch is a biased estimate of the worst case of
the full dataset. Measure that bias for the plug-in estimate, remove most
of it with `MultilevelMonteCarloEstimator`, and recover the full-batch
value from minibatches alone with `StochasticPhiDual`.
"""

import functools

import torch

from optora import KLAmbiguitySet, MultilevelMonteCarloEstimator, StochasticPhiDual
from optora.dro import kl_conjugate

torch.set_default_dtype(torch.float64)

generator = torch.Generator().manual_seed(0)
num_points = 4096
population = torch.empty(num_points).exponential_(generator=generator)
radius = 0.1


def draw(batch_size: int) -> torch.Tensor:
    """Draw a minibatch of losses with replacement."""
    return population[torch.randint(num_points, (batch_size,), generator=generator)]


uniform = torch.full((num_points,), 1.0 / num_points)
full_batch = KLAmbiguitySet(uniform, radius=radius).worst_case_expectation(population)
print(f"full-batch worst case    : {full_batch:.4f}")

# Plug-in: the worst case of each minibatch of 32, all draws in one batched
# solve. Its mean is well below the full-batch value.
num_draws = 400
batches = torch.stack([draw(32) for _ in range(num_draws)])
plug_in = KLAmbiguitySet(torch.full((32,), 1 / 32), radius=radius)
plug_in_values = plug_in.worst_case_expectation(batches)
print(f"plug-in, batch 32        : {plug_in_values.mean():.4f}")

# Multilevel Monte Carlo: a random batch size 32 * 2^J and a telescoping
# correction give the bias of a batch of 32 * 2^5 = 1024 points.
estimator = MultilevelMonteCarloEstimator(
    functools.partial(KLAmbiguitySet, radius=radius), base_batch_size=32, max_level=5
)
mlmc_values = torch.stack(
    [estimator(draw(estimator.sample_batch_size(generator))) for _ in range(num_draws)]
)
mlmc = mlmc_values.mean()
standard_error = mlmc_values.std() / num_draws**0.5
print(f"multilevel Monte Carlo   : {mlmc:.4f} +- {standard_error:.4f}")

# Stochastic dual: the dual objective is an expectation for fixed dual
# variables, so minibatch means are unbiased. Training the dual variables on
# minibatches of 32 converges to the full-batch worst case.
dual = StochasticPhiDual(
    radius=radius, phi_conjugate=kl_conjugate, initial_lam=float(population.mean())
)
optimizer = torch.optim.Adam(dual.parameters(), lr=0.02)
for _ in range(1500):
    optimizer.zero_grad()
    dual(draw(32)).backward()
    optimizer.step()
with torch.no_grad():
    dual_value = dual(population)
print(f"stochastic dual, trained : {dual_value:.4f}")

assert plug_in_values.mean() < full_batch - 0.02
assert (mlmc - full_batch).abs() < 3 * standard_error
assert (dual_value - full_batch).abs() < 0.01
