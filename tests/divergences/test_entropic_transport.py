"""Tests for `EntropicTransportDivergence`."""

import math
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.autograd import gradcheck

from optora.core.divergence_base import Divergence
from optora.divergences.entropic_transport import EntropicTransportDivergence
from optora.dro.sinkhorn_dro import SinkhornAmbiguitySet
from tests.dro.duality import entropic_transport_divergence

POINTS = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
COST = torch.abs(POINTS.unsqueeze(-1) - POINTS.unsqueeze(-2))
NOMINAL = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
CANDIDATE = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
SPARSE_NOMINAL = torch.tensor([0.0, 0.4, 0.0, 0.6], dtype=torch.float64)
SPARSE_CANDIDATE = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=torch.float64)


def _smoothed(
    divergence: EntropicTransportDivergence, nominal: torch.Tensor
) -> torch.Tensor:
    kernel = torch.exp(divergence.log_kernel(nominal.dtype))
    return nominal @ kernel


def test_vanishes_at_the_kernel_smoothed_reference() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)

    value = divergence(_smoothed(divergence, NOMINAL), NOMINAL)

    assert float(value) == pytest.approx(0.0, abs=1e-12)


def test_does_not_vanish_at_the_reference_itself() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)

    assert float(divergence(NOMINAL, NOMINAL)) > 1e-3


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
def test_a_zero_cost_reduces_to_kl_from_the_normalized_reference(
    epsilon: float,
) -> None:
    # Every kernel row is nu, and the product coupling p x candidate attains
    # min KL(gamma || p x nu) = KL(candidate || nu).
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    divergence = EntropicTransportDivergence(
        torch.zeros_like(COST), epsilon=epsilon, reference=reference
    )
    nu = reference / reference.sum()

    value = divergence(CANDIDATE, NOMINAL)

    expected = epsilon * float(torch.sum(CANDIDATE * torch.log(CANDIDATE / nu)))
    assert float(value) == pytest.approx(expected, rel=1e-9)


def test_log_kernel_rows_are_distributions_with_reference_zeros() -> None:
    reference = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    divergence = EntropicTransportDivergence(COST, epsilon=1e-3, reference=reference)

    log_kernel = divergence.log_kernel(torch.float64)

    torch.testing.assert_close(
        torch.exp(log_kernel).sum(-1), torch.ones(4, dtype=torch.float64)
    )
    assert bool(torch.all(log_kernel[:, 1] == -torch.inf))
    assert bool(torch.all(torch.isfinite(log_kernel[:, [0, 2, 3]])))


def test_is_positive_away_from_the_centre_and_batches() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)
    candidates = torch.stack([CANDIDATE, CANDIDATE.flip(-1), NOMINAL])

    batched = divergence(candidates, NOMINAL)

    assert batched.shape == (3,)
    assert bool(torch.all(batched > 0))
    for index in range(3):
        assert float(batched[index]) == pytest.approx(
            float(divergence(candidates[index], NOMINAL)), rel=1e-9
        )


def _near_centre(divergence: EntropicTransportDivergence) -> torch.Tensor:
    # A candidate a small divergence from the centre, where the value is
    # tiny and a partly converged solve is relatively furthest off.
    return 0.999 * _smoothed(divergence, NOMINAL) + 0.001 * CANDIDATE


@pytest.mark.parametrize("epsilon", [0.02, 0.05, 0.3])
@pytest.mark.parametrize("near_centre", [False, True], ids=["far", "near_centre"])
def test_matches_the_exact_dual_at_a_sharp_kernel(
    epsilon: float, near_centre: bool
) -> None:
    # Issue #69: at max(C) / epsilon up to 150 Sinkhorn scaling stalls far
    # from the maximum, which near the centre left the value ~100% low.
    divergence = EntropicTransportDivergence(COST, epsilon=epsilon)
    candidate = _near_centre(divergence) if near_centre else CANDIDATE
    kernel = torch.exp(divergence.log_kernel(torch.float64))

    value = divergence(candidate, NOMINAL)

    expected = entropic_transport_divergence(
        candidate.numpy(), NOMINAL.numpy(), kernel.numpy(), epsilon
    )
    assert float(value) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("vanishing", [1e-18, 1e-16, 1e-12, 1e-8])
def test_matches_the_exact_dual_with_vanishing_mass_on_coincident_points(
    vanishing: float,
) -> None:
    # Issue #85: two points at one position give equal kernel columns. With a
    # vanishing share of the plan their curvature fell below the rounding of
    # the unscaled Newton system, which turned singular and stalled 2.5e-3 low.
    points = torch.tensor([1.0, 3.0, 0.0, 1.0], dtype=torch.float64)
    cost = torch.abs(points.unsqueeze(-1) - points.unsqueeze(-2))
    nominal = torch.tensor([0.0, 0.5, 0.0, 0.5], dtype=torch.float64)
    candidate = torch.tensor(
        [vanishing, 0.4978, 0.5022 - 2.0 * vanishing, vanishing], dtype=torch.float64
    )
    divergence = EntropicTransportDivergence(cost, epsilon=0.05)
    kernel = torch.exp(divergence.log_kernel(torch.float64))

    value = divergence(candidate, nominal)

    expected = entropic_transport_divergence(
        candidate.numpy(), nominal.numpy(), kernel.numpy(), 0.05
    )
    assert float(value) == pytest.approx(expected, rel=1e-9)


def test_a_candidate_with_mass_off_the_reference_support_is_infinite() -> None:
    # No coupling is absolutely continuous with diag(q) Q when the candidate
    # puts mass on a column the reference measure gives no weight.
    reference = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    divergence = EntropicTransportDivergence(COST, epsilon=0.5, reference=reference)
    on_support = torch.tensor([0.3, 0.0, 0.3, 0.4], dtype=torch.float64)

    values = divergence(torch.stack([CANDIDATE, on_support]), NOMINAL)

    assert float(values[0]) == math.inf
    assert math.isfinite(float(values[1]))


@pytest.mark.parametrize("epsilon", [0.05, 0.5])
def test_gradient_is_the_envelope_gradient_of_the_converged_dual(
    epsilon: float,
) -> None:
    # The potentials are held fixed under autograd, which by the envelope
    # theorem is the exact gradient once the solve has converged.
    divergence = EntropicTransportDivergence(COST, epsilon=epsilon)
    candidate = CANDIDATE.clone().requires_grad_(True)
    nominal = NOMINAL.clone().requires_grad_(True)

    assert gradcheck(divergence, (candidate, nominal), atol=1e-6)


def test_sinkhorn_honours_its_sync_budget(
    host_sync_counter: Callable[[], Any],
) -> None:
    # At epsilon = 0.02 Sinkhorn cannot converge in 40 iterations, so it
    # reads the flag at each of the 4 checkpoints.
    divergence = EntropicTransportDivergence(
        COST, epsilon=0.02, max_iter=40, newton_max_iter=0, check_interval=10
    )

    with host_sync_counter() as syncs:
        divergence(CANDIDATE, NOMINAL)

    assert len(syncs) == 4


def test_newton_reads_the_flag_once_per_iteration(
    host_sync_counter: Callable[[], Any],
) -> None:
    divergence = EntropicTransportDivergence(
        COST, epsilon=0.02, max_iter=40, newton_max_iter=3, check_interval=10
    )

    with host_sync_counter() as syncs:
        divergence(CANDIDATE, NOMINAL)

    assert len(syncs) == 4 + 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_converged_sinkhorn_stops_early_and_skips_newton(
    dtype: torch.dtype, host_sync_counter: Callable[[], Any]
) -> None:
    # The default tol of 1e-9 is below float32's rounding floor of the
    # marginal violation; flooring it at n * eps lets float32 stop too.
    divergence = EntropicTransportDivergence(COST.to(dtype), epsilon=1.0)

    with host_sync_counter() as syncs:
        divergence(CANDIDATE.to(dtype), NOMINAL.to(dtype))

    assert len(syncs) < divergence.max_iter // divergence.check_interval


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epsilon": 0.0}, "epsilon must be positive"),
        ({"epsilon": math.nan}, "epsilon must be positive"),
        ({"epsilon": 1.0, "tol": 0.0}, "tol must be positive"),
        ({"epsilon": 1.0, "max_iter": 0}, "max_iter must be positive"),
        ({"epsilon": 1.0, "newton_max_iter": -1}, "newton_max_iter must be"),
        ({"epsilon": 1.0, "reference": torch.ones(3)}, "reference must have shape"),
    ],
)
def test_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EntropicTransportDivergence(COST, **kwargs)


# --- Zero-mass marginals (issue #75) -------------------------------------


def _reference(
    divergence: EntropicTransportDivergence,
    candidate: torch.Tensor,
    nominal: torch.Tensor,
) -> float:
    kernel = torch.exp(divergence.log_kernel(torch.float64))
    return entropic_transport_divergence(
        candidate.numpy(), nominal.numpy(), kernel.numpy(), divergence.epsilon
    )


@pytest.mark.parametrize("epsilon", [0.02, 0.3, 1.0])
@pytest.mark.parametrize(
    ("candidate", "nominal"),
    [
        (CANDIDATE, SPARSE_NOMINAL),
        (SPARSE_CANDIDATE, NOMINAL),
        (SPARSE_CANDIDATE, SPARSE_NOMINAL),
    ],
    ids=["zero_rows", "zero_columns", "both"],
)
def test_zero_mass_entries_match_the_dual_restricted_to_the_supports(
    candidate: torch.Tensor, nominal: torch.Tensor, epsilon: float
) -> None:
    # The reference drops zero rows and columns by indexing; the divergence
    # masks them, so the two agree to the solve's tolerance. epsilon = 0.02
    # also runs the Newton phase on the masked problem.
    divergence = EntropicTransportDivergence(COST, epsilon=epsilon)

    value = divergence(candidate, nominal)

    assert float(value) == pytest.approx(
        _reference(divergence, candidate, nominal), rel=1e-9
    )


@pytest.mark.parametrize("weight", [1e-3, 1e-5, 1e-6])
def test_a_tiny_divergence_with_zero_mass_rows_is_not_floored(weight: float) -> None:
    # The former eps = 1e-12 clamp perturbed the semi-dual by eps times the
    # size of the potentials, an absolute floor of ~1e-10 that returned 0.0
    # for a divergence of that size.
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)
    candidate = (1.0 - weight) * _smoothed(divergence, SPARSE_NOMINAL) + weight * (
        CANDIDATE
    )

    value = divergence(candidate, SPARSE_NOMINAL)

    expected = _reference(divergence, candidate, SPARSE_NOMINAL)
    assert 0.0 < expected < 1e-6
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_vanishes_at_the_smoothed_reference_of_a_sparse_nominal() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.5)

    value = divergence(_smoothed(divergence, SPARSE_NOMINAL), SPARSE_NOMINAL)

    assert float(value) == pytest.approx(0.0, abs=1e-15)


def test_zero_patterns_differ_across_a_batch() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=0.05)
    candidates = torch.stack([CANDIDATE, SPARSE_CANDIDATE, SPARSE_CANDIDATE])
    nominals = torch.stack([SPARSE_NOMINAL, NOMINAL, SPARSE_NOMINAL])

    batched = divergence(candidates, nominals)

    for index in range(3):
        assert float(batched[index]) == pytest.approx(
            _reference(divergence, candidates[index], nominals[index]), rel=1e-9
        )


def test_gradient_with_zero_mass_entries_is_finite_and_exact_on_the_support() -> None:
    # The derivative in the mass of an empty column is -inf and is reported
    # as zero; on the supports the envelope gradient stays exact.
    divergence = EntropicTransportDivergence(COST, epsilon=0.3)
    support = SPARSE_CANDIDATE > 0.0

    def on_support(weights: torch.Tensor) -> torch.Tensor:
        candidate = torch.zeros(4, dtype=torch.float64).masked_scatter(support, weights)
        return divergence(candidate, SPARSE_NOMINAL)

    weights = SPARSE_CANDIDATE[support].clone().requires_grad_(True)
    assert gradcheck(on_support, (weights,), atol=1e-6)

    candidate = SPARSE_CANDIDATE.clone().requires_grad_(True)
    nominal = SPARSE_NOMINAL.clone().requires_grad_(True)
    divergence(candidate, nominal).backward()
    assert candidate.grad is not None and nominal.grad is not None
    assert bool(torch.all(torch.isfinite(candidate.grad)))
    assert bool(torch.all(torch.isfinite(nominal.grad)))
    assert bool(torch.all(candidate.grad[~support] == 0.0))


def test_eps_is_deprecated_and_ignored() -> None:
    with pytest.warns(DeprecationWarning, match="eps is deprecated"):
        deprecated = EntropicTransportDivergence(COST, epsilon=0.5, eps=1e-3)
    with pytest.warns(DeprecationWarning, match="eps is deprecated"):
        SinkhornAmbiguitySet(NOMINAL, COST, epsilon=0.5, radius=0.1, eps=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = EntropicTransportDivergence(COST, epsilon=0.5)
        SinkhornAmbiguitySet(NOMINAL, COST, epsilon=0.5, radius=0.1)

    assert float(deprecated(CANDIDATE, SPARSE_NOMINAL)) == float(
        default(CANDIDATE, SPARSE_NOMINAL)
    )


def test_rejects_a_non_square_cost() -> None:
    with pytest.raises(ValueError, match="square"):
        EntropicTransportDivergence(torch.zeros(3, 4), epsilon=1.0)


def test_validate_rejects_a_non_finite_cost() -> None:
    cost = COST.clone()
    cost[0, 1] = math.inf

    with pytest.raises(ValueError, match="cost must be finite"):
        EntropicTransportDivergence(cost, epsilon=1.0, validate=True)


def test_rejects_mismatched_distribution_sizes() -> None:
    divergence = EntropicTransportDivergence(COST, epsilon=1.0)

    with pytest.raises(ValueError, match="to index cost"):
        divergence(torch.ones(3) / 3, NOMINAL)


def test_is_a_divergence_instance() -> None:
    assert isinstance(EntropicTransportDivergence(COST, epsilon=1.0), Divergence)
