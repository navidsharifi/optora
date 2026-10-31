"""Shared infrastructure for the generated documentation figures.

Every figure on the documentation site is rendered by `tools.figures` at
build time, once per colour scheme, into a git-ignored output directory.
This module holds what the figure modules share: the two colour schemes
and their validated categorical palette, the fixed colour each ambiguity
set keeps on every chart, the Matplotlib style that matches the site, and
the `Context` a figure renders through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
from cycler import cycler
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class Theme:
    """One colour scheme of the documentation site.

    Attributes:
        name: Suffix of every rendered file, `light` or `dark`; it matches the
            `#only-light` / `#only-dark` image fragments the pages use.
        surface: Page background the figure is drawn on. Static figures keep
            a transparent background; animations are rendered on it.
        ink: Primary text colour.
        secondary: Secondary text colour, used for axis labels and legends.
        muted: Muted colour for tick labels, the nominal distribution, and
            reference lines.
        grid: Hairline gridline colour, one step off the surface.
        axis: Axis spine colour.
        series: Eight categorical colours in a fixed, colour-blind-safe
            order (never cycled past eight).
        sequential: One-hue ramp from the surface outwards, for magnitudes.
        wash: Fill alpha for the area under a line.
    """

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]
    wash: float = 0.12


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#1b1b1b",
    secondary="#52514e",
    muted="#898781",
    grid="#e6e5df",
    axis="#c3c2b7",
    series=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
    sequential=(
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ),
)

DARK = Theme(
    name="dark",
    surface="#1e2129",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#8e8f97",
    grid="#30333d",
    axis="#454857",
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
    sequential=(
        "#0d366b",
        "#184f95",
        "#256abf",
        "#3987e5",
        "#6da7ec",
        "#9ec5f4",
        "#cde2fb",
    ),
)

THEMES: tuple[Theme, ...] = (LIGHT, DARK)

# Each ambiguity set owns one categorical slot on every chart, so a reader
# who learns "KL is blue" on one page can rely on it on the next. The order
# is the palette's colour-blind-safe order, not a ranking.
SET_SLOTS: dict[str, int] = {
    "kl": 0,
    "chi_square": 1,
    "total_variation": 2,
    "cvar": 3,
    "wasserstein": 4,
    "sinkhorn": 5,
    "mmd": 6,
    "phi": 7,
}

SET_LABELS: dict[str, str] = {
    "kl": "KL",
    "chi_square": "chi-square",
    "total_variation": "total variation",
    "cvar": "CVaR",
    "wasserstein": "Wasserstein",
    "sinkhorn": "Sinkhorn",
    "mmd": "MMD",
    "phi": "general phi",
}


def set_color(theme: Theme, key: str) -> str:
    """Return the colour the ambiguity set `key` keeps on every chart."""
    return theme.series[SET_SLOTS[key]]


def rc_params(theme: Theme) -> dict[str, Any]:
    """Return the Matplotlib rc parameters that match the site in `theme`."""
    return {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "svg.fonttype": "path",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "text.color": theme.ink,
        "axes.labelcolor": theme.secondary,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(theme.series)),
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": theme.surface,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": theme.secondary,
        "legend.handlelength": 1.6,
        "figure.titlesize": 12,
        "figure.titleweight": "semibold",
    }


@contextmanager
def themed(theme: Theme) -> Iterator[None]:
    """Apply the site-matching Matplotlib style of `theme` inside the block."""
    with matplotlib.rc_context(rc_params(theme)):
        yield


@dataclass
class Context:
    """What one figure rendering sees: the colour scheme and where to write.

    Attributes:
        theme: Colour scheme being rendered.
        output_dir: Directory every file is written to.
        quick: Render a small, fast variant (fewer frames, smaller problems)
            for tests and local iteration.
        device: Device the numerical work runs on.
    """

    theme: Theme
    output_dir: Path
    quick: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def frames(self, full: int) -> int:
        """Return the frame count of an animation, cut down in quick mode."""
        return min(full, 6) if self.quick else full

    def size(self, full: int, quick: int) -> int:
        """Return a problem size, replaced by `quick` in quick mode."""
        return quick if self.quick else full

    def path(self, filename: str) -> Path:
        """Return the output path of `filename`, creating the directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename

    def save_static(self, fig: Figure, stem: str, extension: str = "svg") -> Path:
        """Write `fig` as `<stem>-<theme>.<extension>` and close it."""
        path = self.path(f"{stem}-{self.theme.name}.{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.15, transparent=True)
        plt.close(fig)
        return path

    def save_animation(
        self,
        fig: Figure,
        update: Callable[[int], Any],
        frames: int,
        stem: str,
        fps: int = 12,
    ) -> Path:
        """Write a GIF of `fig` driven by `update(frame)` and close it."""
        path = self.path(f"{stem}-{self.theme.name}.gif")
        animation = FuncAnimation(fig, update, frames=frames, blit=False)
        animation.save(
            path,
            writer=PillowWriter(fps=fps),
            savefig_kwargs={"facecolor": self.theme.surface, "transparent": False},
        )
        plt.close(fig)
        return path


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """One registered figure: what it writes, where it is used, how to render it.

    Attributes:
        name: Registry name, also the `--only` identifier on the command line.
        page: Documentation page (relative to `docs/`) that embeds the output.
        outputs: File name patterns the renderer produces; `{theme}` expands
            to each colour scheme's name, a pattern without it is written
            once regardless of theme.
        render: Callable that draws and saves the figure through a `Context`.
    """

    name: str
    page: str
    outputs: tuple[str, ...]
    render: Callable[[Context], None]

    @property
    def themed(self) -> bool:
        """Whether the figure is rendered once per colour scheme."""
        return any("{theme}" in pattern for pattern in self.outputs)

    def files(self, themes: Iterable[Theme] = THEMES) -> list[str]:
        """Return the file names the figure writes for `themes`."""
        names: list[str] = []
        for pattern in self.outputs:
            if "{theme}" in pattern:
                names.extend(pattern.format(theme=theme.name) for theme in themes)
            else:
                names.append(pattern)
        return names


def worst_case_distributions(
    build: Callable[[torch.Tensor], Any],
    loss: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the worst-case value and distribution of `loss` at every radius.

    The worst-case distribution is the gradient of the worst-case expectation
    with respect to the loss (Danskin's theorem), so it is read off autograd
    rather than from any set's internals.

    Args:
        build: Constructor taking a tensor radius of shape `(R,)` and
            returning the ambiguity set.
        loss: Per-scenario loss of shape `(n,)`.
        radii: Radii of shape `(R,)`.

    Returns:
        The values of shape `(R,)` and the maximizers of shape `(R, n)`.
    """
    ambiguity_set = build(radii)
    batched = loss.detach().expand(radii.shape[0], -1).clone().requires_grad_()
    values = ambiguity_set.worst_case_expectation(batched)
    (maximizers,) = torch.autograd.grad(values.sum(), batched)
    return values.detach(), maximizers
