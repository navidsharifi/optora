---
icon: lucide/gauge
---

# Benchmarks

Every number on this page is measured when the documentation is built: the
deploy workflow installs this commit of Optora, times every ambiguity set on
the GitHub-hosted runner's CPU, checks each one against an independent
reference solver, and renders the charts below. Nothing here is typed in by
hand, and the environment table says exactly what produced the figures.

!!! note "CPU numbers, by design"

    The documentation runner has no accelerator, so these are CPU timings
    with the default PyTorch thread count. They measure the shape of the
    cost (how it scales with the support, how batching amortizes it, where
    host synchronizations occur) rather than peak throughput. To measure
    your own machine, including a GPU, see [Reproducing](#reproducing).

--8<-- "docs/assets/figures/benchmarks-summary.md"

## Latency by ambiguity set

One call of `worst_case_expectation` on a batch of float32 loss vectors, the
shape one training step produces, forward alone and with the backward pass
that yields the worst-case distribution. The exact sets (KL, $\chi^2$, total
variation, CVaR, Wasserstein, Sinkhorn) run a fixed, dtype-determined
sequence of kernels; MMD runs an interior-point method to a $10^{-10}$ gap;
the general $\varphi$ set runs an injected fixed-step dual solver to its
iteration budget.

<figure class="figure" markdown="span">
  ![Median wall time of the worst-case expectation per ambiguity set, forward and with backward](assets/figures/benchmarks-latency-light.svg#only-light){ loading=lazy }
  ![Median wall time of the worst-case expectation per ambiguity set, forward and with backward](assets/figures/benchmarks-latency-dark.svg#only-dark){ loading=lazy }
  <figcaption>Median over repeated calls after warm-up. The backward pass
  adds little because the gradient is the maximizer the forward pass already
  found (Danskin's theorem), not a differentiation through the solve.</figcaption>
</figure>

## Scaling with the support size

The same sets on a single loss vector as the support grows. KL, $\chi^2$,
total variation and CVaR touch each scenario a fixed number of times per
solve; Wasserstein and Sinkhorn evaluate an $n \times n$ cost or kernel per
step; MMD factors an $n \times n$ matrix per interior-point iteration, which
caps the sizes it is swept over.

<figure class="figure" markdown="span">
  ![Median wall time against the support size for every ambiguity set](assets/figures/benchmarks-scaling-light.svg#only-light){ loading=lazy }
  ![Median wall time against the support size for every ambiguity set](assets/figures/benchmarks-scaling-dark.svg#only-dark){ loading=lazy }
  <figcaption>Log-log axes. A flat start means the cost is kernel-launch
  bound; the slope at the right is the arithmetic cost of the solve.</figcaption>
</figure>

## Batched throughput

`worst_case_expectation` takes a loss of shape `(..., n)` and solves every
leading element in one joint, vectorized call. The chart counts how many
loss vectors per second that call completes as the batch grows: the
per-call overhead is paid once, so throughput rises until the arithmetic
dominates.

<figure class="figure" markdown="span">
  ![Loss vectors solved per second against the batch size](assets/figures/benchmarks-throughput-light.svg#only-light){ loading=lazy }
  ![Loss vectors solved per second against the batch size](assets/figures/benchmarks-throughput-dark.svg#only-dark){ loading=lazy }
  <figcaption>Batching a radius sweep, a set of seeds, or a minibatch of
  problems into the leading dimension is the single largest speed-up
  available on any device.</figcaption>
</figure>

## Against independent reference solvers

The test suite checks every ambiguity set against a reference that shares no
code with Optora: exact linear programs solved by SciPy's HiGHS for total
variation, CVaR and Wasserstein, SciPy root finding on the closed-form KL
tilt, an active-set closed form for $\chi^2$, and Clarabel's interior-point
method for the MMD cone program. The same references are timed here on one
float64 instance, and the largest difference is reported relative to the
loss spread.

<figure class="figure" markdown="span">
  ![Wall time of Optora and of the independent reference solver per ambiguity set](assets/figures/benchmarks-reference-light.svg#only-light){ loading=lazy }
  ![Wall time of Optora and of the independent reference solver per ambiguity set](assets/figures/benchmarks-reference-dark.svg#only-dark){ loading=lazy }
  <figcaption>A scalar closed form in NumPy can beat Optora on one small
  problem, where Optora's cost is kernel launches; the differences reported
  above each pair are rounding-level agreement, which is the point of the
  comparison.</figcaption>
</figure>

## Host synchronizations

A device-to-host synchronization, reading a tensor back as a Python value,
stalls an accelerator until the device has caught up. The table above counts
them exactly: the tilt-solved and closed-form sets perform none, forward or
backward, because their trip count is fixed by the dtype rather than by a
convergence flag. The MMD interior-point method and the general $\varphi$
dual check convergence every `check_interval` iterations, and each check is
one synchronization. The test suite pins these counts as exact budgets on
every platform in CI, so a change that adds a synchronization fails the
build.

## Reproducing

The benchmarks and every chart on this site are produced by one command
from a development install:

```bash
pip install -e ".[dev,docs,examples]"
python -m tools.figures --only benchmarks-data benchmarks-charts
```

Pass `--device cuda` to time an accelerator. The raw measurements are
written to `docs/assets/figures/benchmarks-results.json` next to the charts,
and the pull-request guard in `tests/benchmarks/` compares the timing of
each commit against its base with `pytest-benchmark`; see
[`CONTRIBUTING.md`](https://github.com/navidsharifi/optora/blob/main/CONTRIBUTING.md#performance-guard).

[Raw results of this build (JSON)](assets/figures/benchmarks-results.json){ .md-button }
