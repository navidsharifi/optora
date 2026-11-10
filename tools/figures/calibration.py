r"""The calibrated chi-square interval, animated over the sample size.

The animation belongs to example 06. It grows a nested sample from a
standard exponential and, at every size $n$, builds the two-sided mean
interval $[-\sup_q \mathbb{E}_q[-Z],\, \sup_q \mathbb{E}_q[Z]]$ of the
chi-square ball at the calibrated radius $\chi^2_{1,1-\alpha}/n$ for each
of two hundred replications. The intervals shrink like $1/\sqrt{n}$, their
coverage climbs toward the nominal level, and the under-coverage the
skewness causes at small $n$ fades.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from optora.dro import ChiSquareAmbiguitySet, chi_square_radius
from tools.figures._base import Context, FigureSpec, set_color

HOLD_FRAMES = 10
SEED = 20260915
POPULATION_MEAN = 1.0
LEVELS = torch.cat(
    [
        torch.arange(0.50, 0.951, 0.025, dtype=torch.float64),
        torch.tensor([0.96, 0.97, 0.98, 0.99], dtype=torch.float64),
    ]
)
TRACKED = (0.50, 0.80, 0.90, 0.95, 0.99)
SHOWN_LEVEL = 0.90
SHOWN_INTERVALS = 50


def level_index(level: float) -> int:
    """Return the position of `level` in `LEVELS`."""
    return int(torch.isclose(LEVELS, torch.tensor(level, dtype=LEVELS.dtype)).nonzero())


def calibrated_intervals(
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the calibrated interval of every replication at every level.

    Args:
        samples: Replications of shape `(replications, n)`.

    Returns:
        Lower and upper ends, each of shape `(levels, replications)`.
    """
    size = samples.shape[-1]
    nominal = torch.full((size,), 1.0 / size, dtype=samples.dtype)
    radii = torch.tensor(
        [chi_square_radius(size, float(level)) for level in LEVELS],
        dtype=samples.dtype,
    )[:, None]
    ambiguity_set = ChiSquareAmbiguitySet(nominal, radii)
    lower = -ambiguity_set.worst_case_expectation(-samples)
    upper = ambiguity_set.worst_case_expectation(samples)
    return lower, upper


def render_calibration_coverage(context: Context) -> None:
    """Animate the intervals, their coverage and the coverage error as n grows."""
    theme = context.theme
    replications = context.size(200, 40)
    candidates = torch.logspace(np.log10(8), np.log10(320), context.frames(44))
    sizes = candidates.round().long().unique().tolist()
    frames = len(sizes)
    generator = torch.Generator().manual_seed(SEED)
    # Nested samples: every size reads the first n columns of one draw, so the
    # intervals shrink continuously instead of jumping between frames.
    samples = -torch.log(
        torch.rand(replications, sizes[-1], generator=generator, dtype=torch.float64)
    )
    shown = level_index(SHOWN_LEVEL)
    shown_intervals = min(SHOWN_INTERVALS, replications)
    coverage = torch.empty(frames, LEVELS.shape[0], dtype=torch.float64)
    intervals = torch.empty(frames, 2, shown_intervals, dtype=torch.float64)
    for frame, size in enumerate(sizes):
        lower, upper = calibrated_intervals(samples[:, :size])
        covered = (lower <= POPULATION_MEAN) & (upper >= POPULATION_MEAN)
        coverage[frame] = covered.to(torch.float64).mean(-1)
        intervals[frame, 0] = lower[shown, :shown_intervals]
        intervals[frame, 1] = upper[shown, :shown_intervals]
    errors = coverage - LEVELS
    color, miss_color = set_color(theme, "chi_square"), theme.series[6]
    level_colors = [theme.sequential[index] for index in (2, 3, 4, 5, 6)]

    fig, (strip, curve, error) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    rows = np.arange(shown_intervals)
    segments = LineCollection([], linewidths=1.8)
    strip.add_collection(segments)
    strip.axvline(POPULATION_MEAN, color=theme.ink, linewidth=1.2)
    strip.set_xlim(-0.3, 2.9)
    strip.set_ylim(-1, shown_intervals)
    strip.set_yticks([])
    strip.grid(False)
    strip.set_xlabel("mean of a standard exponential; the true mean is 1")
    strip.set_ylabel(f"the first {shown_intervals} replications")
    strip.set_title(f"{int(100 * SHOWN_LEVEL)}% intervals at the calibrated radius")
    strip.legend(
        handles=[
            plt.Line2D([], [], color=color, linewidth=1.8),
            plt.Line2D([], [], color=miss_color, linewidth=1.8),
        ],
        labels=["covers the true mean", "misses it"],
        loc="upper right",
        fontsize=7.5,
    )
    note = strip.text(
        0.02, 0.97, "", transform=strip.transAxes, ha="left", va="top", fontsize=9
    )

    levels_np = LEVELS.numpy()
    curve.plot(levels_np, levels_np, color=theme.ink, linestyle="--", linewidth=1.2)
    (coverage_line,) = curve.plot([], [], color=color, marker="o", markersize=4)
    curve.set_xlim(float(LEVELS[0]), 1.0)
    curve.set_ylim(0.4, 1.0)
    curve.set_xlabel("nominal confidence $1 - \\alpha$")
    curve.set_ylabel("empirical coverage")
    curve.set_title("coverage against the nominal level")
    curve.text(
        0.52, 0.53, "exact calibration", fontsize=8, color=theme.secondary, rotation=45
    )

    error_lines = {}
    sizes_np = np.asarray(sizes, dtype=float)
    for level, shade in zip(TRACKED, level_colors, strict=True):
        (error_lines[level],) = error.plot(
            [], [], color=shade, linewidth=1.4, label=f"$1-\\alpha$ = {level:.2f}"
        )
    error.axhline(0.0, color=theme.ink, linestyle="--", linewidth=1.2)
    error.set_xscale("log")
    error.set_xlim(sizes_np[0], sizes_np[-1])
    low = float(errors[:, [level_index(level) for level in TRACKED]].min())
    error.set_ylim(low * 1.15, 0.05)
    error.set_xlabel("sample size $n$  (log)")
    error.set_ylabel("coverage $-$ nominal")
    error.set_title("the under-coverage shrinks with $n$")
    error.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        lower, upper = intervals[frame].numpy()
        segments.set_segments(
            [
                ((lo, row), (hi, row))
                for lo, hi, row in zip(lower, upper, rows, strict=True)
            ]
        )
        covered = (lower <= POPULATION_MEAN) & (upper >= POPULATION_MEAN)
        segments.set_color([color if hit else miss_color for hit in covered])
        coverage_line.set_data(levels_np, coverage[frame].numpy())
        for level, line in error_lines.items():
            line.set_data(
                sizes_np[: frame + 1], errors[: frame + 1, level_index(level)].numpy()
            )
        note.set_text(
            f"$n$ = {sizes[frame]}\n"
            f"coverage at {SHOWN_LEVEL:.2f}: {float(coverage[frame, shown]):.3f}\n"
            f"({replications} replications)"
        )

    context.save_animation(
        fig, update, frames + HOLD_FRAMES, "calibration-coverage", fps=8
    )


FIGURES = (
    FigureSpec(
        name="calibration-coverage",
        page="examples/calibrated_ambiguity_radius.md",
        outputs=("calibration-coverage-{theme}.gif",),
        render=render_calibration_coverage,
    ),
)
