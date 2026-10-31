r"""Tests for group DRO: `group_losses` and `OnlineGroupDRO`.

`OBJECTIVES` is the registry of group-DRO objectives, each an existing
ambiguity set or the online estimator applied to the group losses that
`group_losses` computes, so every one runs through the same batching, dtype,
host-synchronization and `torch.func` checks, as `tests.dro.specs.SPECS` does
for the ambiguity sets. Four of them are exact group DRO, the maximum over
groups, and are checked against that maximum: CVaR at level zero, total
variation at radius one, and KL and chi-square balls around the group mix at
their saturation radii. `OnlineGroupDRO` is checked against Algorithm 1 of
Sagawa et al. (2020) written out, against the KL worst-case path its iterates
follow, and against the closed-form minimax solution of a training problem.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from optora.core.dro_base import AmbiguitySet
from optora.dro.group_dro import OnlineGroupDRO, group_losses
from optora.dro.kl_dro import KLAmbiguitySet
from optora.dro.phi_dro import (
    ChiSquareAmbiguitySet,
    CVaRAmbiguitySet,
    TotalVariationAmbiguitySet,
)

DTYPE = torch.float64
NUM_GROUPS = 4
GROUPS = torch.tensor([0, 1, 2, 3, 0, 0, 1, 2, 0, 3, 1, 0])
LOSS = torch.tensor(
    [0.3, 1.2, -0.4, 2.0, 0.1, 0.9, 0.7, -1.1, 0.4, 1.5, 0.2, -0.3], dtype=DTYPE
)
COUNTS = torch.bincount(GROUPS, minlength=NUM_GROUPS).to(DTYPE)
PROPORTIONS = COUNTS / COUNTS.sum()
STEP_SIZE = 0.5

GroupObjective = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ObjectiveSpec:
    """A group-DRO objective over the vector of group losses.

    Attributes:
        name: Identifier used as the pytest parameter id.
        build: Map from the group proportions to the objective, a function
            of group losses of shape `(..., G)` returning `(...)`.
        worst_group: Whether the objective is exactly the maximum over the
            groups, rather than a reweighting of them.
        ulps: Rounding the worst-group value may carry, in units of the
            dtype's epsilon times the largest absolute group loss.
    """

    name: str
    build: Callable[[torch.Tensor], GroupObjective]
    worst_group: bool
    ulps: float = 0.0


def _set_objective(ambiguity_set: AmbiguitySet) -> GroupObjective:
    return ambiguity_set.worst_case_expectation


# Saturation radii are mathematically tight boundaries: a radius computed
# there can round to either side of it, so these sit strictly beyond the
# saturation radius of the smallest group, where every group saturates.
SATURATION_MARGIN = 1.5


def _kl_saturated(proportions: torch.Tensor) -> GroupObjective:
    radius = -math.log(float(proportions.min())) * SATURATION_MARGIN
    return _set_objective(KLAmbiguitySet(proportions, radius=radius))


def _chi_square_saturated(proportions: torch.Tensor) -> GroupObjective:
    radius = (1.0 / float(proportions.min()) - 1.0) * SATURATION_MARGIN
    return _set_objective(ChiSquareAmbiguitySet(proportions, radius=radius))


OBJECTIVES = (
    ObjectiveSpec(
        "cvar",
        lambda p: _set_objective(CVaRAmbiguitySet(p, alpha=0.0)),
        worst_group=True,
    ),
    ObjectiveSpec(
        "total_variation",
        lambda p: _set_objective(TotalVariationAmbiguitySet(p, radius=1.0)),
        worst_group=True,
        ulps=4.0,
    ),
    ObjectiveSpec("kl", _kl_saturated, worst_group=True),
    ObjectiveSpec("chi_square", _chi_square_saturated, worst_group=True),
    ObjectiveSpec(
        "online",
        lambda p: OnlineGroupDRO(p, step_size=STEP_SIZE),
        worst_group=False,
    ),
)

WORST_GROUP = tuple(spec for spec in OBJECTIVES if spec.worst_group)


def objective_parameters(
    specs: tuple[ObjectiveSpec, ...],
) -> pytest.MarkDecorator:
    """Parametrize a test over `specs`, with each spec's name as its id."""
    return pytest.mark.parametrize("spec", specs, ids=[spec.name for spec in specs])


def _composed(spec: ObjectiveSpec) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the objective of the per-sample losses under `GROUPS`."""
    objective = spec.build(PROPORTIONS)
    return lambda loss: objective(group_losses(loss, GROUPS, NUM_GROUPS))


def _reference_group_losses(
    loss: torch.Tensor, groups: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Average `loss` within each group with an explicit Python loop."""
    values = [
        loss[groups == group].mean() if bool((groups == group).any()) else 0.0
        for group in range(num_groups)
    ]
    return torch.tensor(values, dtype=loss.dtype)


def _batch(generator_seed: int, rows: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(generator_seed)
    return torch.randn(rows, len(GROUPS), generator=generator, dtype=DTYPE)


def _gradient(
    function: Callable[[torch.Tensor], torch.Tensor], point: torch.Tensor
) -> torch.Tensor:
    leaf = point.detach().clone().requires_grad_(True)
    (result,) = torch.autograd.grad(function(leaf), leaf)
    return result


# --- group_losses ------------------------------------------------------------


class TestGroupLosses:
    """The scatter-mean helper against an explicit loop and its identities."""

    def test_matches_an_explicit_loop(self) -> None:
        result = group_losses(LOSS, GROUPS, NUM_GROUPS)

        expected = _reference_group_losses(LOSS, GROUPS, NUM_GROUPS)
        torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)

    def test_an_empty_group_has_loss_zero(self) -> None:
        result = group_losses(LOSS, GROUPS, NUM_GROUPS + 2)

        assert result.shape == (NUM_GROUPS + 2,)
        assert torch.equal(result[NUM_GROUPS:], torch.zeros(2, dtype=DTYPE))
        torch.testing.assert_close(
            result[:NUM_GROUPS], group_losses(LOSS, GROUPS, NUM_GROUPS)
        )

    def test_proportion_weighted_group_losses_are_the_mean_loss(self) -> None:
        result = group_losses(LOSS, GROUPS, NUM_GROUPS) @ PROPORTIONS

        torch.testing.assert_close(result, LOSS.mean(), rtol=1e-15, atol=1e-16)

    def test_gradient_is_the_inverse_group_count(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DTYPE)

        result = _gradient(
            lambda loss: group_losses(loss, GROUPS, NUM_GROUPS) @ weights, LOSS
        )

        assert torch.equal(result, weights[GROUPS] / COUNTS[GROUPS])

    def test_batched_loss_with_shared_groups_matches_rows(self) -> None:
        loss = _batch(0).reshape(5, 1, len(GROUPS))

        result = group_losses(loss, GROUPS, NUM_GROUPS)

        expected = torch.stack(
            [group_losses(row, GROUPS, NUM_GROUPS) for row in loss.reshape(5, -1)]
        )
        assert result.shape == (5, 1, NUM_GROUPS)
        assert torch.equal(result.reshape(5, NUM_GROUPS), expected)

    def test_per_row_group_assignments_match_rows(self) -> None:
        loss = _batch(1)
        groups = torch.randint(
            NUM_GROUPS, loss.shape, generator=torch.Generator().manual_seed(1)
        )

        result = group_losses(loss, groups, NUM_GROUPS)

        expected = torch.stack(
            [
                _reference_group_losses(row, assignment, NUM_GROUPS)
                for row, assignment in zip(loss, groups, strict=True)
            ]
        )
        torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)

    def test_one_loss_against_several_group_assignments(self) -> None:
        groups = torch.stack([GROUPS, torch.flip(GROUPS, (0,))])

        result = group_losses(LOSS, groups, NUM_GROUPS)

        assert result.shape == (2, NUM_GROUPS)
        assert torch.equal(result[0], group_losses(LOSS, GROUPS, NUM_GROUPS))

    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint8])
    def test_accepts_any_integer_index_dtype(self, index_dtype: torch.dtype) -> None:
        result = group_losses(LOSS, GROUPS.to(index_dtype), NUM_GROUPS)

        assert torch.equal(result, group_losses(LOSS, GROUPS, NUM_GROUPS))

    def test_dtype_follows_the_loss(self, dtype: torch.dtype) -> None:
        assert group_losses(LOSS.to(dtype), GROUPS, NUM_GROUPS).dtype == dtype

    def test_half_precision_is_the_float32_reduction_rounded_once(
        self, half_dtype: torch.dtype
    ) -> None:
        loss = (LOSS * 100.0).to(half_dtype)

        result = group_losses(loss, GROUPS, NUM_GROUPS)

        assert result.dtype == half_dtype
        expected = group_losses(loss.float(), GROUPS, NUM_GROUPS).to(half_dtype)
        assert torch.equal(result, expected)

    def test_non_finite_losses_stay_in_their_group(self) -> None:
        loss = LOSS.clone()
        loss[1] = math.inf
        loss[2] = math.nan

        result = group_losses(loss, GROUPS, NUM_GROUPS)

        assert result[1] == math.inf
        assert torch.isnan(result[2])
        assert torch.isfinite(result[[0, 3]]).all()

    def test_out_of_range_index_raises_on_cpu(self) -> None:
        with pytest.raises(RuntimeError):
            group_losses(LOSS, GROUPS, NUM_GROUPS - 1)

    @pytest.mark.parametrize(
        ("loss", "groups", "num_groups", "match"),
        [
            (LOSS, GROUPS, 0, "num_groups"),
            (torch.tensor(1.0), torch.tensor(0), 1, "scalar"),
            (LOSS, GROUPS.to(DTYPE), NUM_GROUPS, "integer"),
            (LOSS, GROUPS > 0, NUM_GROUPS, "integer"),
            (LOSS, GROUPS[:5], NUM_GROUPS, "broadcast"),
        ],
        ids=["no_groups", "scalar_loss", "float_groups", "bool_groups", "shapes"],
    )
    def test_rejects_invalid_arguments(
        self, loss: torch.Tensor, groups: torch.Tensor, num_groups: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            group_losses(loss, groups, num_groups)

    def test_never_synchronizes(self, host_sync_counter: Callable[[], Any]) -> None:
        loss = LOSS.clone().requires_grad_()

        with host_sync_counter() as syncs:
            group_losses(loss, GROUPS, NUM_GROUPS).sum().backward()

        assert len(syncs) == 0


# --- Exact group DRO as existing ambiguity sets --------------------------------


@objective_parameters(WORST_GROUP)
def test_worst_group_objectives_equal_the_maximum_group_loss(
    spec: ObjectiveSpec,
) -> None:
    for seed in range(20):
        loss = _batch(seed, rows=1)[0]
        group_loss = group_losses(loss, GROUPS, NUM_GROUPS)

        value = _composed(spec)(loss)

        tolerance = spec.ulps * torch.finfo(DTYPE).eps * float(group_loss.abs().max())
        assert abs(float(value - group_loss.max())) <= tolerance


@objective_parameters(WORST_GROUP)
def test_worst_group_gradient_is_uniform_on_the_worst_group(
    spec: ObjectiveSpec,
) -> None:
    # Group DRO reweights samples, not groups: d/d loss_i is 1/n_g* on the
    # worst group g* and zero elsewhere.
    for seed in range(20):
        loss = _batch(seed, rows=1)[0]
        worst = int(group_losses(loss, GROUPS, NUM_GROUPS).argmax())

        result = _gradient(_composed(spec), loss)

        expected = torch.eq(GROUPS, worst).to(DTYPE) / COUNTS[worst]
        torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


def test_kl_and_chi_square_past_the_worst_group_saturation_are_bitwise_exact() -> None:
    # Past the saturation radius of the worst group, log(1/p*) for KL and
    # 1/p* - 1 for chi-square, the tilted sets return max(loss) itself, not a
    # rounded expectation, even when the other groups are not saturated.
    for seed in range(20):
        group_loss = group_losses(_batch(seed, rows=1)[0], GROUPS, NUM_GROUPS)
        worst = float(PROPORTIONS[group_loss.argmax()])
        kl = KLAmbiguitySet(PROPORTIONS, radius=-math.log(worst) * SATURATION_MARGIN)
        chi_square = ChiSquareAmbiguitySet(
            PROPORTIONS, radius=(1.0 / worst - 1.0) * SATURATION_MARGIN
        )

        assert torch.equal(kl.worst_case_expectation(group_loss), group_loss.max())
        assert torch.equal(
            chi_square.worst_case_expectation(group_loss), group_loss.max()
        )


def test_cvar_at_level_one_is_erm_and_interpolates_to_group_dro() -> None:
    group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)
    levels = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.0], dtype=DTYPE)

    values = CVaRAmbiguitySet(PROPORTIONS, alpha=levels).worst_case_expectation(
        group_loss
    )

    torch.testing.assert_close(values[0], LOSS.mean(), rtol=1e-15, atol=1e-16)
    assert bool(torch.all(values[1:] >= values[:-1]))
    torch.testing.assert_close(values[-1], group_loss.max(), rtol=1e-15, atol=0.0)


def test_cvar_ignores_groups_absent_from_the_proportions() -> None:
    # An empty group's placeholder loss of zero exceeds every negative group
    # loss; CVaR excludes it because its proportion is zero.
    loss = -1.0 - LOSS.abs()
    group_loss = group_losses(loss, GROUPS, NUM_GROUPS + 1)
    proportions = torch.cat([PROPORTIONS, torch.zeros(1, dtype=DTYPE)])

    value = CVaRAmbiguitySet(proportions, alpha=0.0).worst_case_expectation(group_loss)

    assert float(group_loss[-1]) == 0.0
    torch.testing.assert_close(
        value, group_loss[:NUM_GROUPS].max(), rtol=1e-15, atol=0.0
    )


# --- Registry-wide checks ------------------------------------------------------


@objective_parameters(OBJECTIVES)
def test_batched_loss_matches_rows(spec: ObjectiveSpec) -> None:
    loss = _batch(2)
    function = _composed(spec)

    result = function(loss)

    expected = torch.stack([function(row) for row in loss])
    assert result.shape == (5,)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


@objective_parameters(OBJECTIVES)
def test_per_row_group_assignments_match_rows(spec: ObjectiveSpec) -> None:
    loss = _batch(3)
    groups = torch.stack([torch.roll(GROUPS, shift) for shift in range(5)])
    objective = spec.build(PROPORTIONS)

    result = objective(group_losses(loss, groups, NUM_GROUPS))

    expected = torch.stack(
        [
            objective(group_losses(row, assignment, NUM_GROUPS))
            for row, assignment in zip(loss, groups, strict=True)
        ]
    )
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


@objective_parameters(OBJECTIVES)
def test_per_sample_gradient_is_the_group_weight_over_the_group_count(
    spec: ObjectiveSpec,
) -> None:
    objective = spec.build(PROPORTIONS)
    group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)
    weights = _gradient(objective, group_loss)

    result = _gradient(_composed(spec), LOSS)

    torch.testing.assert_close(
        result, weights[GROUPS] / COUNTS[GROUPS], rtol=0.0, atol=1e-15
    )
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=DTYPE))


@objective_parameters(OBJECTIVES)
def test_torch_func_grad_matches_autograd(spec: ObjectiveSpec) -> None:
    function = _composed(spec)

    torch.testing.assert_close(
        torch.func.grad(function)(LOSS), _gradient(function, LOSS), rtol=0.0, atol=0.0
    )


@objective_parameters(OBJECTIVES)
def test_torch_func_vmap_matches_a_batched_call(spec: ObjectiveSpec) -> None:
    function = _composed(spec)
    loss = _batch(4)

    torch.testing.assert_close(
        torch.func.vmap(function)(loss), function(loss), rtol=0.0, atol=1e-15
    )


@objective_parameters(OBJECTIVES)
def test_torch_func_vmap_over_group_assignments(spec: ObjectiveSpec) -> None:
    objective = spec.build(PROPORTIONS)
    groups = torch.stack([torch.roll(GROUPS, shift) for shift in range(3)])

    result = torch.func.vmap(
        lambda assignment: objective(group_losses(LOSS, assignment, NUM_GROUPS))
    )(groups)

    expected = objective(group_losses(LOSS, groups, NUM_GROUPS))
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


@objective_parameters(OBJECTIVES)
def test_forward_and_backward_never_synchronize(
    spec: ObjectiveSpec, host_sync_counter: Callable[[], Any]
) -> None:
    function = _composed(spec)
    loss = _batch(5).requires_grad_()

    with host_sync_counter() as syncs:
        function(loss).sum().backward()

    assert len(syncs) == 0


@objective_parameters(OBJECTIVES)
def test_float32_losses_return_float32_close_to_float64(spec: ObjectiveSpec) -> None:
    objective = spec.build(PROPORTIONS.float())
    loss = _batch(6)

    result = objective(group_losses(loss.float(), GROUPS, NUM_GROUPS))

    expected = spec.build(PROPORTIONS)(group_losses(loss, GROUPS, NUM_GROUPS))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.double(), expected, rtol=1e-6, atol=1e-6)


@objective_parameters(OBJECTIVES)
def test_half_precision_losses_are_solved_in_float32(
    spec: ObjectiveSpec, half_dtype: torch.dtype
) -> None:
    objective = spec.build(PROPORTIONS.float())
    loss = _batch(7).to(half_dtype)
    group_loss = group_losses(loss, GROUPS, NUM_GROUPS)

    result = objective(group_loss)

    assert result.dtype == torch.float32
    assert torch.equal(result, objective(group_loss.float()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@objective_parameters(OBJECTIVES)
def test_cuda_matches_cpu(spec: ObjectiveSpec) -> None:
    objective = spec.build(PROPORTIONS.cuda())
    loss = _batch(8)

    result = objective(group_losses(loss.cuda(), GROUPS.cuda(), NUM_GROUPS))

    assert result.device.type == "cuda"
    torch.testing.assert_close(result.cpu(), _composed(spec)(loss))


# --- OnlineGroupDRO -------------------------------------------------------------


def _uniform(*shape: int) -> torch.Tensor:
    return torch.full((*shape, NUM_GROUPS), 1.0 / NUM_GROUPS, dtype=DTYPE)


def _algorithm_one(
    weights: torch.Tensor, group_loss: torch.Tensor, step_size: float
) -> torch.Tensor:
    """One weight update of Sagawa et al.'s Algorithm 1, in probability space."""
    unnormalized = weights * torch.exp(step_size * group_loss)
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True)


class TestOnlineGroupDRO:
    """Construction, the ascent step, and the weighted loss."""

    @pytest.mark.parametrize(
        ("initial_weights", "step_size", "match"),
        [
            (torch.tensor(1.0, dtype=DTYPE), 0.1, "shape"),
            (torch.ones(3, dtype=torch.int64), 0.1, "floating-point"),
            (torch.ones(2, 0, dtype=DTYPE), 0.1, "at least one group"),
            (_uniform(), -0.1, "nonnegative"),
            (_uniform(), math.nan, "nonnegative"),
            (_uniform(), math.inf, "nonnegative"),
            (_uniform(), torch.ones(3, dtype=DTYPE), "broadcast"),
            (_uniform(2), torch.ones(3, dtype=DTYPE), "broadcast"),
        ],
        ids=[
            "scalar",
            "integer",
            "no_groups",
            "negative_step",
            "nan_step",
            "inf_step",
            "tensor_step_without_batch",
            "tensor_step_mismatch",
        ],
    )
    def test_rejects_invalid_arguments(
        self,
        initial_weights: torch.Tensor,
        step_size: float | torch.Tensor,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            OnlineGroupDRO(initial_weights, step_size)

    @pytest.mark.parametrize(
        ("initial_weights", "step_size", "match"),
        [
            (torch.tensor([0.5, 0.6], dtype=DTYPE), 0.1, "sum to one"),
            (torch.tensor([1.5, -0.5], dtype=DTYPE), 0.1, "nonnegative"),
            (torch.tensor([0.5, math.nan], dtype=DTYPE), 0.1, "nonnegative"),
            (
                torch.full((2, 2), 0.5, dtype=DTYPE),
                torch.tensor([0.1, -0.1], dtype=DTYPE),
                "step_size",
            ),
        ],
        ids=["mass", "negative", "nan", "negative_tensor_step"],
    )
    def test_validate_rejects_invalid_values(
        self,
        initial_weights: torch.Tensor,
        step_size: float | torch.Tensor,
        match: str,
    ) -> None:
        OnlineGroupDRO(initial_weights, step_size)
        with pytest.raises(ValueError, match=match):
            OnlineGroupDRO(initial_weights, step_size, validate=True)

    def test_validate_accepts_a_distribution(self) -> None:
        adversary = OnlineGroupDRO(
            PROPORTIONS.expand(2, -1), torch.tensor([0.1, 0.2]), validate=True
        )

        assert adversary.num_groups == NUM_GROUPS

    def test_weights_start_at_the_initial_weights_and_are_buffers(self) -> None:
        adversary = OnlineGroupDRO(PROPORTIONS, STEP_SIZE)

        torch.testing.assert_close(adversary.weights, PROPORTIONS, rtol=1e-15, atol=0.0)
        assert set(dict(adversary.named_buffers())) == {"log_weights"}
        assert not list(adversary.parameters())

    def test_tensor_step_size_is_a_buffer(self) -> None:
        adversary = OnlineGroupDRO(_uniform(3), torch.tensor([0.1, 0.2, 0.3]))

        assert set(dict(adversary.named_buffers())) == {"log_weights", "step_size"}

    def test_unnormalized_initial_weights_are_normalized(self) -> None:
        adversary = OnlineGroupDRO(COUNTS, STEP_SIZE)

        torch.testing.assert_close(adversary.weights, PROPORTIONS, rtol=1e-15, atol=0.0)

    def test_step_is_algorithm_one(self) -> None:
        adversary = OnlineGroupDRO(PROPORTIONS, STEP_SIZE)
        expected = PROPORTIONS
        for seed in range(10):
            group_loss = group_losses(_batch(seed, rows=1)[0], GROUPS, NUM_GROUPS)

            adversary.step(group_loss)

            expected = _algorithm_one(expected, group_loss, STEP_SIZE)
            torch.testing.assert_close(
                adversary.weights, expected, rtol=1e-14, atol=0.0
            )

    def test_a_group_absent_from_the_minibatch_keeps_its_unnormalized_weight(
        self,
    ) -> None:
        # A minibatch of one group is Algorithm 1's single-group update: only
        # the sampled group's weight is multiplied, then all are normalized.
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)
        loss = torch.tensor([1.0, 3.0], dtype=DTYPE)

        adversary.step(group_losses(loss, torch.tensor([2, 2]), NUM_GROUPS))

        ratios = adversary.weights / adversary.weights[0]
        torch.testing.assert_close(
            ratios,
            torch.tensor([1.0, 1.0, math.exp(STEP_SIZE * 2.0), 1.0], dtype=DTYPE),
            rtol=1e-14,
            atol=0.0,
        )

    def test_forward_is_the_weighted_group_loss_and_leaves_the_weights(self) -> None:
        adversary = OnlineGroupDRO(PROPORTIONS, STEP_SIZE)
        adversary.step(group_losses(LOSS, GROUPS, NUM_GROUPS))
        before = adversary.log_weights.clone()
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)

        value = adversary(group_loss)

        assert torch.equal(adversary.log_weights, before)
        torch.testing.assert_close(value, adversary.weights @ group_loss)
        torch.testing.assert_close(_gradient(adversary, group_loss), adversary.weights)

    def test_zero_step_size_is_erm_over_the_initial_weights(self) -> None:
        adversary = OnlineGroupDRO(PROPORTIONS, 0.0)
        for seed in range(5):
            adversary.step(group_losses(_batch(seed, rows=1)[0], GROUPS, NUM_GROUPS))

        value = adversary(group_losses(LOSS, GROUPS, NUM_GROUPS))

        torch.testing.assert_close(adversary.weights, PROPORTIONS, rtol=1e-15, atol=0.0)
        torch.testing.assert_close(value, LOSS.mean(), rtol=1e-15, atol=1e-16)

    @pytest.mark.parametrize("steps", [1, 3, 10, 40])
    def test_constant_losses_trace_the_kl_worst_case_path(self, steps: int) -> None:
        # T steps on fixed losses give the tilt q0 exp(T eta l) / Z, the
        # maximizer of the KL ball around q0 whose radius is its divergence.
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)
        adversary = OnlineGroupDRO(PROPORTIONS, STEP_SIZE)
        for _ in range(steps):
            adversary.step(group_loss)
        weights = adversary.weights
        radius = float(torch.sum(weights * torch.log(weights / PROPORTIONS)))
        kl = KLAmbiguitySet(PROPORTIONS, radius=radius)

        torch.testing.assert_close(
            _gradient(kl.worst_case_expectation, group_loss),
            weights,
            rtol=0.0,
            atol=1e-10,
        )
        torch.testing.assert_close(
            adversary(group_loss),
            kl.worst_case_expectation(group_loss),
            rtol=0.0,
            atol=1e-10,
        )

    def test_constant_losses_drive_the_weights_to_the_worst_group(self) -> None:
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)
        values = []
        for _ in range(200):
            adversary.step(group_loss)
            values.append(float(adversary(group_loss)))

        assert all(
            later >= earlier
            for earlier, later in zip(values[:-1], values[1:], strict=True)
        )
        worst = int(group_loss.argmax())
        assert float(adversary.weights[worst]) > 1.0 - 1e-6
        assert values[-1] == pytest.approx(float(group_loss.max()), abs=1e-6)

    def test_a_zero_initial_weight_stays_zero(self) -> None:
        initial = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=DTYPE)
        adversary = OnlineGroupDRO(initial, STEP_SIZE)
        group_loss = torch.tensor([0.0, 9.0, 1.0, 5.0], dtype=DTYPE)

        for _ in range(20):
            adversary.step(group_loss)

        assert torch.equal(adversary.weights[[1, 3]], torch.zeros(2, dtype=DTYPE))
        tilt = math.exp(20 * STEP_SIZE)
        assert float(adversary(group_loss)) == pytest.approx(tilt / (1.0 + tilt))

    def test_batched_adversaries_match_independent_ones(self) -> None:
        loss = _batch(9, rows=3)
        group_loss = group_losses(loss, GROUPS, NUM_GROUPS)
        batched = OnlineGroupDRO(_uniform(3), STEP_SIZE)
        singles = [OnlineGroupDRO(_uniform(), STEP_SIZE) for _ in range(3)]

        for _ in range(4):
            batched.step(group_loss)
            for single, row in zip(singles, group_loss, strict=True):
                single.step(row)

        expected = torch.stack([single.weights for single in singles])
        torch.testing.assert_close(batched.weights, expected, rtol=1e-15, atol=0.0)
        torch.testing.assert_close(
            batched(group_loss),
            torch.stack(
                [single(row) for single, row in zip(singles, group_loss, strict=True)]
            ),
        )

    def test_one_group_loss_steps_every_batched_adversary(self) -> None:
        batched = OnlineGroupDRO(_uniform(2, 3), STEP_SIZE)
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)

        batched.step(group_loss)

        expected = _algorithm_one(_uniform(), group_loss, STEP_SIZE)
        torch.testing.assert_close(
            batched.weights, expected.expand(2, 3, -1), rtol=1e-14, atol=0.0
        )

    def test_step_rejects_losses_with_more_batch_dimensions(self) -> None:
        adversary = OnlineGroupDRO(_uniform(3), STEP_SIZE)

        with pytest.raises(ValueError, match="broadcast to the weights"):
            adversary.step(torch.zeros(2, 3, NUM_GROUPS, dtype=DTYPE))
        with pytest.raises(ValueError, match="broadcast to the weights"):
            adversary.step(torch.zeros(2, NUM_GROUPS, dtype=DTYPE))

    @pytest.mark.parametrize("method", ["step", "forward"])
    def test_rejects_the_wrong_number_of_groups(self, method: str) -> None:
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)

        with pytest.raises(ValueError, match=rf"\(\.\.\., {NUM_GROUPS}\)"):
            getattr(adversary, method)(torch.zeros(NUM_GROUPS + 1, dtype=DTYPE))
        with pytest.raises(ValueError, match="shape"):
            getattr(adversary, method)(torch.tensor(0.0, dtype=DTYPE))

    def test_tensor_step_size_sweeps_in_one_module(self) -> None:
        step_sizes = [0.0, 0.1, 1.0]
        sweep = OnlineGroupDRO(_uniform(3), torch.tensor(step_sizes, dtype=DTYPE))
        singles = [OnlineGroupDRO(_uniform(), size) for size in step_sizes]
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)

        for _ in range(3):
            sweep.step(group_loss)
            for single in singles:
                single.step(group_loss)

        expected = torch.stack([single.weights for single in singles])
        torch.testing.assert_close(sweep.weights, expected, rtol=1e-15, atol=0.0)

    def test_step_runs_without_autograd(self) -> None:
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)
        loss = LOSS.clone().requires_grad_()

        adversary.step(group_losses(loss, GROUPS, NUM_GROUPS))

        assert not adversary.log_weights.requires_grad
        assert loss.grad is None

    def test_step_refuses_to_run_inside_a_torch_func_transform(self) -> None:
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)
        before = adversary.log_weights.clone()

        def function(group_loss: torch.Tensor) -> torch.Tensor:
            adversary.step(group_loss)
            return adversary(group_loss)

        with pytest.raises(RuntimeError, match="torch.func"):
            torch.func.grad(function)(group_losses(LOSS, GROUPS, NUM_GROUPS))
        assert torch.equal(adversary.log_weights, before)

    def test_forward_under_torch_func_hessian_is_zero(self) -> None:
        adversary = OnlineGroupDRO(PROPORTIONS, STEP_SIZE)
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)

        result = torch.func.hessian(adversary)(group_loss)

        assert torch.equal(result, torch.zeros(NUM_GROUPS, NUM_GROUPS, dtype=DTYPE))

    def test_state_dict_round_trips_the_weights(self) -> None:
        adversary = OnlineGroupDRO(_uniform(), STEP_SIZE)
        adversary.step(group_losses(LOSS, GROUPS, NUM_GROUPS))
        restored = OnlineGroupDRO(_uniform(), STEP_SIZE)

        restored.load_state_dict(adversary.state_dict())

        assert torch.equal(restored.weights, adversary.weights)

    def test_dtype_follows_the_module_and_the_loss(self, dtype: torch.dtype) -> None:
        adversary = OnlineGroupDRO(_uniform().to(dtype), STEP_SIZE)
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS).float()

        adversary.step(group_loss)

        assert adversary.log_weights.dtype == dtype
        assert adversary(group_loss).dtype == torch.promote_types(dtype, torch.float32)

    def test_step_and_forward_never_synchronize(
        self, host_sync_counter: Callable[[], Any]
    ) -> None:
        adversary = OnlineGroupDRO(_uniform(5), torch.full((5,), STEP_SIZE))
        loss = _batch(10).requires_grad_()

        with host_sync_counter() as syncs:
            group_loss = group_losses(loss, GROUPS, NUM_GROUPS)
            adversary.step(group_loss)
            adversary(group_loss).sum().backward()

        assert len(syncs) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_matches_cpu(self) -> None:
        group_loss = group_losses(LOSS, GROUPS, NUM_GROUPS)
        cpu = OnlineGroupDRO(_uniform(), STEP_SIZE)
        cuda = OnlineGroupDRO(_uniform(), STEP_SIZE).to("cuda")

        cpu.step(group_loss)
        cuda.step(group_loss.cuda())

        assert cuda.log_weights.device.type == "cuda"
        torch.testing.assert_close(cuda.weights.cpu(), cpu.weights)


def test_online_training_reaches_the_group_dro_minimax_solution() -> None:
    # min_theta max_g (theta - c_g)^2 has the closed-form solution theta* at
    # the midpoint of the extreme targets, value (spread / 2)^2, attained by
    # weights split evenly between the two extreme groups. Gradient descent
    # against exponentiated-gradient ascent (Algorithm 1 on exact group
    # losses) reaches it in its averaged iterates.
    targets = torch.tensor([-1.0, 0.2, 0.5, 2.0], dtype=DTYPE)
    theta = torch.zeros((), dtype=DTYPE, requires_grad=True)
    adversary = OnlineGroupDRO(_uniform(), step_size=0.5)
    optimizer = torch.optim.SGD([theta], lr=0.05)
    steps, averaged_theta, averaged_weights = 4000, 0.0, torch.zeros_like(targets)

    for step in range(steps):
        group_loss = (theta - targets) ** 2
        adversary.step(group_loss)
        optimizer.zero_grad()
        adversary(group_loss).backward()
        optimizer.step()
        if step >= steps // 2:
            averaged_theta += float(theta) / (steps - steps // 2)
            averaged_weights += adversary.weights / (steps - steps // 2)

    assert averaged_theta == pytest.approx(0.5, abs=1e-2)
    worst_group_risk = float(((averaged_theta - targets) ** 2).max())
    assert worst_group_risk == pytest.approx(1.5**2, rel=1e-3)
    torch.testing.assert_close(
        averaged_weights,
        torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=DTYPE),
        rtol=0.0,
        atol=2e-2,
    )
