"""Calibrating the ambiguity radius from finite-sample theory.

Every DRO example has to pick a radius, and picking one by hand is the
weakest part of the whole pipeline: too small and the robust value is just
the empirical mean, too large and it saturates at the worst scenario.
Finite-sample theory removes the choice. For the chi-square ambiguity set
built on an empirical distribution of `n` points, the profile divergence

    T_n(mu) = min { D_chi2(q || nominal) : E_q[Z] = mu }

evaluated at the *true* mean satisfies `n * T_n(mu_true) -> chi2_1` in
distribution, which is the empirical-likelihood / DRO calibration result of
Duchi and Namkoong. So setting

    radius(n, alpha) = chi2_{1, 1 - alpha} / n

makes the ambiguity set an asymptotically valid `(1 - alpha)` confidence
region for the true distribution, and the interval it induces on the mean,

    [ -sup_q E_q[-Z] , sup_q E_q[Z] ]

an asymptotically valid `(1 - alpha)` confidence interval. Nothing here is
a new solver: `ChiSquareAmbiguitySet` is used exactly as it ships, and the
only new ingredient is the radius formula.

This script measures whether that promise holds at finite `n`. It draws
many independent samples from a standard exponential (mean 1, variance 1,
skewness 2 — deliberately far from Gaussian), builds the calibrated
interval for each one, and reports the fraction of intervals that actually
cover the true mean, against the nominal confidence and against `n`.

Two things are worth watching:

* Coverage is *below* nominal at small `n`. The calibration is asymptotic,
  and skewness is exactly what the chi-square limit ignores.
* The dual solve is not redundant. In the interior regime the worst-case
  expectation equals `mean + sqrt(radius * variance)`, but a large radius
  drives some `q_i` to zero, where that formula overstates the worst case
  and only the dual is correct. The printed `worst closed-form gap` column
  is the most negative such discrepancy over the replications: it is solver
  noise everywhere except at small `n` and high confidence, which is
  precisely the regime where the calibration is under the most strain.

References:
    John C. Duchi and Hongseok Namkoong, "Learning Models with Uniform
    Performance via Distributionally Robust Optimization", Annals of
    Statistics 49(3), 2021.
    Rui Gao, "Finite-Sample Guarantees for Wasserstein Distributionally
    Robust Optimization", Operations Research 73(4), 2025.

Run:
    python examples/06_calibrated_ambiguity_radius.py
"""

import math

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure

from optora.dro import ChiSquareAmbiguitySet
from optora.solvers import GradientDescent

CONFIDENCE_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
SAMPLE_SIZES = (10, 40, 160)
NUM_REPLICATIONS = 200
SEED = 20260915
POPULATION_MEAN = 1.0
POPULATION_VARIANCE = 1.0
DUAL_SOLVER = GradientDescent(step_size=0.2, max_iter=120, tol=1e-7)


def chi_square_quantile(confidence: float) -> float:
    """Return the `confidence` quantile of the chi-square law with one dof.

    A chi-square variable with one degree of freedom is the square of a
    standard normal, so its quantile function is the squared two-sided
    normal quantile and needs no special-function code beyond
    `torch.special.ndtri`.

    Args:
        confidence: Probability level in `(0, 1)`.

    Returns:
        The `confidence` quantile of `chi2_1`.
    """
    normal_quantile = torch.special.ndtri(
        torch.tensor((1.0 + confidence) / 2.0, dtype=torch.float64)
    )
    return float(normal_quantile**2)


def calibrated_radius(num_samples: int, confidence: float) -> float:
    """Return the chi-square ambiguity radius that targets `confidence` coverage.

    Args:
        num_samples: Number of observations in the empirical distribution.
        confidence: Target coverage level in `(0, 1)`.

    Returns:
        `chi2_{1, confidence} / num_samples`, the radius under which the
        ambiguity set is an asymptotically valid confidence region for the
        data-generating distribution.
    """
    return chi_square_quantile(confidence) / num_samples


def exponential_samples(num_samples: int, generator: torch.Generator) -> torch.Tensor:
    """Draw `NUM_REPLICATIONS` independent standard-exponential samples.

    Args:
        num_samples: Observations per replication.
        generator: Seeded generator, so the reported coverage is
            reproducible.

    Returns:
        A `(NUM_REPLICATIONS, num_samples)` tensor of nonnegative draws with
        unit mean and unit variance.
    """
    uniform = torch.rand(
        NUM_REPLICATIONS, num_samples, generator=generator, dtype=torch.float64
    )
    return -torch.log(uniform)


def build_ambiguity_sets(
    nominal: torch.Tensor, radius: float
) -> tuple[ChiSquareAmbiguitySet, ChiSquareAmbiguitySet]:
    """Build the two ambiguity sets that produce a two-sided mean interval.

    Both sets are identical as sets; they differ only in where their dual
    solve starts. Keeping them separate lets each one warm-start from its
    own previous optimum across replications instead of being dragged back
    and forth between the upper and the lower problem. The starting point
    is the asymptotic interior optimum implied by the *population* moments,
    which is a good guess without reading anything off the sample.

    Args:
        nominal: Uniform empirical distribution over the support.
        radius: Calibrated chi-square radius.

    Returns:
        The `(lower, upper)` ambiguity sets.
    """
    initial_log_eta = 0.5 * math.log(POPULATION_VARIANCE / (4.0 * radius))
    lower = ChiSquareAmbiguitySet(
        nominal=nominal,
        radius=radius,
        dual_solver=DUAL_SOLVER,
        initial_log_eta=initial_log_eta,
        initial_lam=-POPULATION_MEAN,
    )
    upper = ChiSquareAmbiguitySet(
        nominal=nominal,
        radius=radius,
        dual_solver=DUAL_SOLVER,
        initial_log_eta=initial_log_eta,
        initial_lam=POPULATION_MEAN,
    )
    return lower, upper


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    coverage_by_size: dict[int, list[float]] = {}
    closed_form_gap_by_setting: dict[tuple[int, float], float] = {}

    for num_samples in SAMPLE_SIZES:
        # Common random numbers across confidence levels: the calibrated sets
        # are then nested by construction, so measured coverage is exactly
        # monotone in the nominal level rather than monotone up to noise.
        samples = exponential_samples(num_samples, generator)
        nominal = torch.full((num_samples,), 1.0 / num_samples, dtype=torch.float64)
        sample_mean = torch.mean(samples, dim=-1)
        sample_variance = torch.var(samples, dim=-1, unbiased=False)

        print(f"\nn = {num_samples}, {NUM_REPLICATIONS} replications")
        print("  confidence   radius   coverage   worst closed-form gap")
        coverages = []
        for confidence in CONFIDENCE_LEVELS:
            radius = calibrated_radius(num_samples, confidence)
            lower_set, upper_set = build_ambiguity_sets(nominal, radius)
            lower = torch.stack(
                [-lower_set.worst_case_expectation(-sample) for sample in samples]
            )
            upper = torch.stack(
                [upper_set.worst_case_expectation(sample) for sample in samples]
            )

            covered = (lower <= POPULATION_MEAN) & (upper >= POPULATION_MEAN)
            coverage = float(torch.mean(covered.to(samples.dtype)))
            asymptotic = sample_mean + torch.sqrt(radius * sample_variance)
            closed_form_gap = float(torch.min(upper - asymptotic))

            coverages.append(coverage)
            closed_form_gap_by_setting[num_samples, confidence] = closed_form_gap
            print(
                f"  {confidence:>9.2f}   {radius:>6.4f}   {coverage:>8.3f}"
                f"   {closed_form_gap:>+21.2e}"
            )
        coverage_by_size[num_samples] = coverages

    nominal_levels = torch.tensor(CONFIDENCE_LEVELS, dtype=torch.float64)
    for num_samples, coverages in coverage_by_size.items():
        measured = torch.tensor(coverages, dtype=torch.float64)
        assert bool(torch.all(torch.diff(measured) >= 0.0)), num_samples

    errors = {
        num_samples: torch.tensor(coverages, dtype=torch.float64) - nominal_levels
        for num_samples, coverages in coverage_by_size.items()
    }
    worst_error = {
        num_samples: float(torch.max(torch.abs(error)))
        for num_samples, error in errors.items()
    }
    print("\nlargest |coverage - nominal| by sample size")
    for num_samples, error in worst_error.items():
        print(f"  n={num_samples:<5} {error:.3f}")
    assert worst_error[SAMPLE_SIZES[-1]] < worst_error[SAMPLE_SIZES[0]]

    # The boundary regime has to bind somewhere, or the dual solve would be
    # an expensive way of evaluating `mean + sqrt(radius * variance)`.
    assert closed_form_gap_by_setting[SAMPLE_SIZES[0], CONFIDENCE_LEVELS[-1]] < -1e-4

    _assert_calibration_matches_normal_interval()

    fig, (ax_coverage, ax_error) = plt.subplots(1, 2, figsize=(11, 4))
    ax_coverage.plot(
        CONFIDENCE_LEVELS,
        CONFIDENCE_LEVELS,
        color="black",
        linestyle="--",
        label="exact calibration",
    )
    for num_samples, coverages in coverage_by_size.items():
        ax_coverage.plot(
            CONFIDENCE_LEVELS, coverages, marker="o", label=f"n = {num_samples}"
        )
    ax_coverage.set_xlabel("nominal confidence 1 - alpha")
    ax_coverage.set_ylabel("empirical coverage")
    ax_coverage.set_title("calibrated radius: coverage vs. nominal level")
    ax_coverage.legend()

    ax_error.axhline(0.0, color="black", linestyle="--")
    for index, confidence in enumerate(CONFIDENCE_LEVELS):
        ax_error.plot(
            SAMPLE_SIZES,
            [float(errors[num_samples][index]) for num_samples in SAMPLE_SIZES],
            marker="o",
            label=f"1 - alpha = {confidence:.2f}",
        )
    ax_error.set_xscale("log")
    ax_error.set_xlabel("sample size n")
    ax_error.set_ylabel("coverage - nominal")
    ax_error.set_title("under-coverage shrinks as the sample grows")
    ax_error.legend(fontsize="small")

    fig.tight_layout()
    save_figure(fig, "06_calibrated_ambiguity_radius")


def _assert_calibration_matches_normal_interval() -> None:
    """Check the calibrated radius reproduces the textbook normal interval.

    In the interior regime the chi-square worst-case expectation is
    `mean + sqrt(radius * variance)`, so the calibrated radius
    `z_{1-alpha/2}^2 / n` must reproduce the half-width
    `z_{1-alpha/2} * std / sqrt(n)` exactly. This is the algebraic identity
    the whole experiment rests on, checked against the solver rather than
    against itself.
    """
    num_samples = 160
    confidence = 0.95
    generator = torch.Generator().manual_seed(SEED)
    sample = exponential_samples(num_samples, generator)[0]
    nominal = torch.full((num_samples,), 1.0 / num_samples, dtype=torch.float64)
    radius = calibrated_radius(num_samples, confidence)

    _, upper_set = build_ambiguity_sets(nominal, radius)
    solved = float(upper_set.worst_case_expectation(sample))

    normal_quantile = math.sqrt(chi_square_quantile(confidence))
    standard_error = float(torch.std(sample, unbiased=False)) / math.sqrt(num_samples)
    textbook = float(torch.mean(sample)) + normal_quantile * standard_error

    print(
        f"\ncalibrated upper bound at n={num_samples}, "
        f"1 - alpha={confidence}: {solved:.9f}"
    )
    print(f"normal-quantile upper bound:               {textbook:.9f}")
    assert abs(solved - textbook) < 1e-5


if __name__ == "__main__":
    main()
