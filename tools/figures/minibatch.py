r"""The bias of the minibatch plug-in estimate, animated over the batch size.

The animation belongs to tutorial 08. On the tutorial's dataset and
least-squares fit it draws two thousand minibatches of every size $m$ from
$8$ to $2048$, runs the KL ball on each minibatch alone, and compares the
spread of those plug-in estimates with the full-batch worst case. The
plug-in is biased low, the bias shrinks with $m$, and it tracks how often a
minibatch contains one of the rare, high-loss points the adversary would
upweight. The multilevel estimator and the stochastic dual are marked at
their own batch sizes for comparison.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from optora.dro import (
    KLAmbiguitySet,
    MultilevelMonteCarloEstimator,
    StochasticPhiDual,
    kl_conjugate,
)
from tools.figures._base import Context, FigureSpec, set_bar_heights, set_color

HOLD_FRAMES = 10
SEED = 20261026
NUM_POINTS = 4096
MINORITY_SHARE = 0.1
NOISE = 0.3
MAJORITY_SLOPE = (1.0, 0.0)
MINORITY_SLOPE = (1.0, 1.5)
RADIUS = 0.2
BASE_BATCH, MAX_LEVEL = 16, 4
DUAL_BATCH = 64
RARE_QUANTILE = 0.99
BINS = 60


def fitted_losses(generator: torch.Generator) -> torch.Tensor:
    """Return the per-sample squared loss of the least-squares fit, shape `(N,)`."""
    minority = torch.rand(NUM_POINTS, generator=generator) < MINORITY_SHARE
    features = torch.randn(NUM_POINTS, 2, generator=generator, dtype=torch.float64)
    slope = torch.where(
        minority[:, None],
        torch.tensor(MINORITY_SLOPE, dtype=torch.float64),
        torch.tensor(MAJORITY_SLOPE, dtype=torch.float64),
    )
    noise = torch.randn(NUM_POINTS, generator=generator, dtype=torch.float64)
    targets = (slope * features).sum(-1) + NOISE * noise
    design = torch.cat([features, torch.ones(NUM_POINTS, 1, dtype=torch.float64)], -1)
    theta = torch.linalg.lstsq(design, targets[:, None]).solution[:, 0]
    return (design @ theta - targets) ** 2


def plug_in_excess(
    loss: torch.Tensor, size: int, draws: int, generator: torch.Generator
) -> torch.Tensor:
    """Return the plug-in robust excess of `draws` minibatches of `size`.

    Each estimate is the KL worst case of the minibatch minus the minibatch
    mean, the control variate tutorial 08 uses: it leaves the bias unchanged
    and removes most of the noise of the mean itself.
    """
    batch = loss[torch.randint(NUM_POINTS, (draws, size), generator=generator)]
    nominal = torch.full((size,), 1.0 / size, dtype=loss.dtype)
    return KLAmbiguitySet(nominal, RADIUS).worst_case_expectation(batch) - batch.mean(
        -1
    )


def multilevel_excess(
    loss: torch.Tensor, draws: int, generator: torch.Generator
) -> tuple[torch.Tensor, float]:
    """Return `draws` multilevel robust-excess estimates and their mean batch size."""
    estimator = MultilevelMonteCarloEstimator(
        functools.partial(KLAmbiguitySet, radius=RADIUS),
        BASE_BATCH,
        MAX_LEVEL,
        dtype=loss.dtype,
    )
    sizes = torch.tensor([estimator.sample_batch_size(generator) for _ in range(draws)])
    estimates = []
    for size in sizes.unique().tolist():
        count = int((sizes == size).sum())
        batch = loss[torch.randint(NUM_POINTS, (count, size), generator=generator)]
        estimates.append(estimator(batch) - batch[:, :BASE_BATCH].mean(-1))
    return torch.cat(estimates), float(sizes.to(torch.float64).mean())


def dual_excess(
    loss: torch.Tensor, draws: int, generator: torch.Generator
) -> torch.Tensor:
    """Return the stochastic dual's robust excess at the full-batch dual optimum."""
    dual = StochasticPhiDual(
        torch.tensor(RADIUS, dtype=loss.dtype),
        kl_conjugate,
        initial_lam=float(loss.mean()),
    )
    optimizer = torch.optim.LBFGS(
        list(dual.parameters()),
        max_iter=500,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = dual(loss)
        value.backward()
        return value

    optimizer.step(closure)
    batch = loss[torch.randint(NUM_POINTS, (draws, DUAL_BATCH), generator=generator)]
    with torch.no_grad():
        return dual(batch) - batch.mean(-1)


def render_plug_in_bias(context: Context) -> None:
    """Animate the plug-in spread, its relative bias and the rare-point share over m."""
    theme = context.theme
    draws = context.size(2000, 100)
    candidates = torch.logspace(np.log10(8), np.log10(2048), context.frames(44))
    sizes = candidates.round().long().unique().tolist()
    frames = len(sizes)
    generator = torch.Generator().manual_seed(SEED)
    loss = fitted_losses(generator)
    uniform = torch.full((NUM_POINTS,), 1.0 / NUM_POINTS, dtype=torch.float64)
    excess = float(
        KLAmbiguitySet(uniform, RADIUS).worst_case_expectation(loss) - loss.mean()
    )
    rare = torch.quantile(loss, RARE_QUANTILE)

    estimates = [plug_in_excess(loss, size, draws, generator) for size in sizes]
    means = torch.tensor([float(values.mean()) for values in estimates])
    errors = torch.tensor([float(values.std()) for values in estimates]) / math.sqrt(
        draws
    )
    bias, bias_error = (means - excess) / excess, 2.0 * errors / excess
    rare_share = torch.tensor(
        [
            float(
                (
                    loss[torch.randint(NUM_POINTS, (draws, size), generator=generator)]
                    > rare
                )
                .any(-1)
                .to(torch.float64)
                .mean()
            )
            for size in sizes
        ]
    )
    multilevel, multilevel_size = multilevel_excess(loss, draws, generator)
    dual = dual_excess(loss, draws, generator)
    multilevel_bias = (float(multilevel.mean()) - excess) / excess
    dual_bias = (float(dual.mean()) - excess) / excess

    plug_color = set_color(theme, "kl")
    multilevel_color, dual_color = theme.series[2], theme.series[1]
    edges = np.linspace(
        float(torch.quantile(estimates[0], 0.005)),
        max(float(torch.quantile(estimates[0], 0.99)), excess * 1.5),
        BINS + 1,
    )
    centers, widths = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    densities = [
        np.histogram(values.numpy(), bins=edges, density=True)[0]
        for values in estimates
    ]

    fig, (spread, trend, share) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=90, gridspec_kw={"width_ratios": [1.3, 1.1, 1]}
    )
    bars = spread.bar(centers, densities[0], width=widths, color=plug_color, alpha=0.8)
    spread.axvline(
        excess, color=theme.ink, linewidth=1.4, label="full-batch robust excess"
    )
    mean_line = spread.axvline(
        float(means[0]),
        color=plug_color,
        linewidth=1.4,
        linestyle="--",
        label="mean plug-in estimate",
    )
    spread.set_xlim(edges[0], edges[-1])
    spread.set_ylim(0, max(float(density.max()) for density in densities) * 1.08)
    spread.set_xlabel("plug-in estimate of the robust excess $F - \\bar\\ell$")
    spread.set_ylabel("density over minibatches")
    spread.set_title("the plug-in estimates of one minibatch")
    spread.legend(loc="upper right", fontsize=7.5)
    note = spread.text(
        0.98, 0.78, "", transform=spread.transAxes, ha="right", va="top", fontsize=9
    )

    sizes_np = np.asarray(sizes, dtype=float)
    trend.fill_between(
        sizes_np,
        (bias - bias_error).numpy(),
        (bias + bias_error).numpy(),
        color=plug_color,
        alpha=0.15,
        linewidth=0,
    )
    trend.plot(sizes_np, bias.numpy(), color=plug_color, label="plug-in on $m$ samples")
    (trend_dot,) = trend.plot([], [], "o", color=plug_color, markersize=8)
    trend.plot(
        [multilevel_size],
        [multilevel_bias],
        "D",
        color=multilevel_color,
        markersize=8,
        label=f"multilevel, mean batch {multilevel_size:.0f}",
    )
    trend.plot(
        [DUAL_BATCH],
        [dual_bias],
        "s",
        color=dual_color,
        markersize=8,
        label=f"stochastic dual on {DUAL_BATCH}",
    )
    trend.axhline(0.0, color=theme.ink, linewidth=1.2, linestyle="--")
    trend.set_xscale("log")
    trend.set_xlim(sizes_np[0], sizes_np[-1])
    trend.set_ylim(float(bias.min()) * 1.15, 0.12)
    trend.set_xlabel("minibatch size $m$  (log)")
    trend.set_ylabel("bias / robust excess")
    trend.set_title(f"relative bias at radius {RADIUS} (band: 2 s.e.)")
    trend.legend(loc="lower right", fontsize=7.5)

    share.plot(
        sizes_np,
        1.0 - (1.0 - (1.0 - RARE_QUANTILE)) ** sizes_np,
        color=theme.muted,
        linestyle="--",
        linewidth=1.2,
        label="$1 - 0.99^m$",
    )
    share.plot(sizes_np, rare_share.numpy(), color=plug_color, label="measured")
    (share_dot,) = share.plot([], [], "o", color=plug_color, markersize=8)
    share.set_xscale("log")
    share.set_xlim(sizes_np[0], sizes_np[-1])
    share.set_ylim(0, 1.05)
    share.set_xlabel("minibatch size $m$  (log)")
    share.set_ylabel("share of minibatches")
    share.set_title("minibatches holding a top-1% loss")
    share.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)

    def update(frame: int) -> None:
        frame = min(frame, frames - 1)
        size = sizes[frame]
        set_bar_heights(bars, torch.as_tensor(densities[frame]))
        mean_line.set_xdata([float(means[frame])] * 2)
        trend_dot.set_data([size], [float(bias[frame])])
        share_dot.set_data([size], [float(rare_share[frame])])
        note.set_text(
            f"$m$ = {size}\n"
            f"bias = {float(bias[frame]):+.1%} of the excess\n"
            f"{float(rare_share[frame]):.0%} of minibatches hold a top-1% loss"
        )

    context.save_animation(fig, update, frames + HOLD_FRAMES, "plug-in-bias", fps=8)


FIGURES = (
    FigureSpec(
        name="plug-in-bias",
        page="tutorials/minibatch_dro_training.md",
        outputs=("plug-in-bias-{theme}.gif",),
        render=render_plug_in_bias,
    ),
)
