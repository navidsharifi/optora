"""Render the figures embedded in the documentation site.

Nothing binary is committed to the repository (see `examples/README.md`),
so every chart and animation the site shows is generated from the library
itself by this package::

    python -m tools.figures            # everything, light and dark
    python -m tools.figures --list     # what exists and which page uses it
    python -m tools.figures --only kl-tilt simplex-geometry --quick

Each registered `FigureSpec` writes its files into `docs/assets/figures/`
(git-ignored), once per colour scheme; pages embed the light and dark
variants with the `#only-light` / `#only-dark` image fragments. The
documentation workflow runs this before `tools.docs build`.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from tools.figures import (
    benchmarks,
    cvar,
    examples,
    generators,
    geometry,
    home,
    mmd,
    numerics,
    sinkhorn,
    tilt,
    training,
    wasserstein,
)
from tools.figures._base import THEMES, Context, FigureSpec, Theme, themed

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "assets" / "figures"

MODULES = (
    home,
    geometry,
    tilt,
    generators,
    cvar,
    wasserstein,
    sinkhorn,
    mmd,
    training,
    numerics,
    examples,
    benchmarks,
)

FIGURES: tuple[FigureSpec, ...] = tuple(
    spec for module in MODULES for spec in module.FIGURES
)


def expected_files(themes: Iterable[Theme] = THEMES) -> list[str]:
    """Return every file name the full registry writes."""
    return [name for spec in FIGURES for name in spec.files(themes)]


def select(names: Sequence[str] | None) -> list[FigureSpec]:
    """Return the registered figures named in `names`, or all of them.

    Raises:
        KeyError: If a name is not registered.
    """
    if not names:
        return list(FIGURES)
    by_name = {spec.name: spec for spec in FIGURES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        msg = f"unknown figures {unknown}; see --list"
        raise KeyError(msg)
    return [by_name[name] for name in names]


def render(
    specs: Iterable[FigureSpec],
    themes: Sequence[Theme],
    output_dir: Path,
    quick: bool = False,
    device: torch.device | None = None,
) -> list[Path]:
    """Render `specs` for `themes` into `output_dir` and return the files."""
    device = device or torch.device("cpu")
    written: list[Path] = []
    for spec in specs:
        for theme in themes if spec.themed else themes[:1]:
            context = Context(theme, output_dir, quick=quick, device=device)
            started = time.perf_counter()
            with themed(theme):
                spec.render(context)
            elapsed = time.perf_counter() - started
            label = f"{spec.name} ({theme.name})" if spec.themed else spec.name
            print(f"rendered {label:<40} {elapsed:6.1f}s", flush=True)
        written.extend(output_dir / name for name in spec.files(themes))
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="Render only these figures."
    )
    parser.add_argument(
        "--list", action="store_true", help="List the figures and exit."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR, help="Output directory."
    )
    parser.add_argument(
        "--theme",
        choices=[theme.name for theme in THEMES] + ["both"],
        default="both",
        help="Colour scheme(s) to render.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Render fast, low-fidelity variants for local iteration.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for the numerical work."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the requested figures."""
    args = build_parser().parse_args(argv)
    if args.list:
        for spec in FIGURES:
            print(f"{spec.name:<28} {spec.page:<40} {', '.join(spec.outputs)}")
        return 0
    themes = [theme for theme in THEMES if args.theme in (theme.name, "both")]
    render(
        select(args.only),
        themes,
        args.output.resolve(),
        quick=args.quick,
        device=torch.device(args.device),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
