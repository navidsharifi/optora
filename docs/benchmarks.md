---
icon: lucide/gauge
---

# Benchmarks

Every number on this page is measured when the documentation is built. The
deploy workflow installs this commit of Optora, times every ambiguity set on
the CPU of the GitHub-hosted runner, compares six of them with an
independent reference solver, and renders the charts below. No number is
typed in by hand, and the environment table records the machine and
software that produced them.

!!! note "CPU numbers, by design"

    The documentation runner has no accelerator, so these are CPU timings
    with the default PyTorch thread count. They show how the cost scales
    with the support size and the batch size and where host
    synchronizations occur, not the peak throughput of a device. To
    measure your own machine, including a GPU, see
    [Reproducing](#reproducing).

--8<-- "docs/assets/figures/benchmarks-highlights.md"

--8<-- "docs/assets/figures/benchmarks-summary.md"

## Latency by ambiguity set

The chart times one call of `worst_case_expectation` on a batch of float32
loss vectors, the shape one training step produces. It shows the forward
pass alone and together with the backward pass, which returns the
worst-case distribution. The KL, $\chi^2$, total variation, CVaR,
Wasserstein and Sinkhorn sets run a fixed sequence of kernels whose length
is set by the dtype. MMD runs an interior-point method to a duality gap of
$10^{-10}$, and the general $\varphi$ set runs an injected fixed-step dual
solver for its full iteration budget.

<figure class="figure" markdown="span">
  ![Median wall time of the worst-case expectation per ambiguity set, forward and with backward](assets/figures/benchmarks-latency-light.svg#only-light){ loading=lazy }
  ![Median wall time of the worst-case expectation per ambiguity set, forward and with backward](assets/figures/benchmarks-latency-dark.svg#only-dark){ loading=lazy }
  <figcaption>Median over repeated calls after warm-up. The backward pass
  adds little time because, by Danskin's theorem, the gradient is the
  maximizer that the forward pass has already found, so autograd does not
  differentiate through the solve.</figcaption>
</figure>

## Scaling with the support size

This chart times the same sets on a single loss vector as the support
grows. KL, $\chi^2$, total variation and CVaR read each scenario a fixed
number of times per solve. Wasserstein and Sinkhorn evaluate an
$n \times n$ cost or kernel matrix at every step. MMD factors an
$n \times n$ matrix at every interior-point iteration, which limits the
support sizes it is swept over.

<figure class="figure" markdown="span">
  ![Median wall time against the support size for every ambiguity set](assets/figures/benchmarks-scaling-light.svg#only-light){ loading=lazy }
  ![Median wall time against the support size for every ambiguity set](assets/figures/benchmarks-scaling-dark.svg#only-dark){ loading=lazy }
  <figcaption>Both axes are logarithmic. A flat start means that kernel
  launches dominate the cost, and the slope on the right reflects the
  arithmetic cost of the solve.</figcaption>
</figure>

## Batched throughput

`worst_case_expectation` takes a loss of shape `(..., n)` and solves every
leading element in one joint, vectorized call. The chart counts how many
loss vectors per second that call completes as the batch grows. The
per-call overhead is paid once per call, so throughput rises until the
arithmetic dominates.

<figure class="figure" markdown="span">
  ![Loss vectors solved per second against the batch size](assets/figures/benchmarks-throughput-light.svg#only-light){ loading=lazy }
  ![Loss vectors solved per second against the batch size](assets/figures/benchmarks-throughput-dark.svg#only-dark){ loading=lazy }
  <figcaption>Placing a radius sweep, a set of seeds or a minibatch of
  problems in the leading dimension amortizes the per-call overhead over
  the whole batch.</figcaption>
</figure>

## Against independent reference solvers

The test suite checks every ambiguity set against a reference that shares no
code with Optora. Total variation, CVaR and Wasserstein are compared with
exact linear programs solved by SciPy's HiGHS, KL with SciPy root finding on
its closed-form tilt, $\chi^2$ with an active-set closed form, and MMD with
Clarabel's interior-point method on the same cone program. These six
references are timed here on one float64 instance, and the difference of
each pair of values is reported relative to the loss spread. Sinkhorn and
the general $\varphi$ set are checked in the tests against a separately
minimized dual in NumPy and SciPy, which is not timed here.

<figure class="figure" markdown="span">
  ![Wall time of Optora and of the independent reference solver per ambiguity set](assets/figures/benchmarks-reference-light.svg#only-light){ loading=lazy }
  ![Wall time of Optora and of the independent reference solver per ambiguity set](assets/figures/benchmarks-reference-dark.svg#only-dark){ loading=lazy }
  <figcaption>On a single small problem a scalar closed form in NumPy can be
  faster than Optora, whose cost there is dominated by kernel launches. The
  ratio above each pair is the reference time divided by the Optora time,
  and the difference below it is limited by float64 rounding or by the
  tolerance of the reference solver.</figcaption>
</figure>

## Host synchronizations

A device-to-host synchronization happens when a tensor is read back as a
Python value. The host then waits until the device has finished all queued
work. The table above counts the conversions of a tensor to a Python `bool`,
`int` or `float`, which is how such a read happens in a solver loop. The
KL, $\chi^2$, total variation, CVaR, Wasserstein and Sinkhorn sets perform
none, forward or backward, because their trip count is fixed by the dtype
rather than by a convergence flag. The MMD interior-point method and the
general $\varphi$ dual check convergence every `check_interval` iterations,
and each check is one synchronization. The test suite asserts these counts
as exact budgets on every platform in CI, so a change that adds a
synchronization fails the build.

## Reproducing

The benchmarks and their charts are produced by one command from a
development install.

```bash
pip install -e ".[dev,docs,examples]"
python -m tools.figures --only benchmarks-data benchmarks-charts
```

Pass `--device cuda` to time an accelerator. The raw measurements are
written to `docs/assets/figures/benchmarks-results.json` next to the charts.
Separately, the pull-request guard in `tests/benchmarks/` compares the
timing of each commit against its base with `pytest-benchmark`, as
described in
[`CONTRIBUTING.md`](https://github.com/navidsharifi/optora/blob/main/CONTRIBUTING.md#performance-guard).

[Raw results of this build (JSON)](assets/figures/benchmarks-results.json){ .md-button }
