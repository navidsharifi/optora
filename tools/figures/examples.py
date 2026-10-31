"""Run the example scripts under the site's style and collect their figures.

The scripts in `examples/` stay the single source of truth for what they
plot; this module only runs them with a Matplotlib configuration matching
each colour scheme and copies the figures they save.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cycler import Cycler

from tools.figures._base import Context, FigureSpec, Theme, rc_params

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
OUTPUTS = EXAMPLES / "outputs"

# Figure stems each script saves, in the order the scripts run.
SCRIPT_FIGURES: dict[str, tuple[str, ...]] = {
    "01_kl_dro_radius_sweep.py": ("01_kl_dro_radius_sweep",),
    "02_worst_case_distribution_shift.py": ("02_worst_case_distribution_shift",),
    "03_robust_decision_across_ambiguity_sets.py": (
        "03_robust_decision_across_ambiguity_sets",
    ),
    "04_convergence_diagnostics.py": (
        "04_outer_convergence",
        "04_inner_dual_landscape",
    ),
    "05_wasserstein_lipschitz_equivalence.py": (
        "05_wasserstein_lipschitz_equivalence",
    ),
    "06_calibrated_ambiguity_radius.py": ("06_calibrated_ambiguity_radius",),
    "07_subpopulation_shift_training.py": ("07_subpopulation_shift_training",),
    "08_minibatch_dro_training.py": ("08_minibatch_dro_training",),
}


def _rc_value(value: object) -> str:
    """Format one rc parameter for a `matplotlibrc` file."""
    if isinstance(value, Cycler):
        colors = ", ".join(f'"{color}"' for color in value.by_key()["color"])
        return f'cycler("color", [{colors}])'
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    # A bare `#` starts a comment in a matplotlibrc file, so hex colours
    # must be quoted.
    return f'"{value}"' if isinstance(value, str) and "#" in value else str(value)


def write_matplotlibrc(theme: Theme, directory: Path) -> Path:
    """Write the site style of `theme` as a `matplotlibrc` file."""
    lines = [f"{key}: {_rc_value(value)}" for key, value in rc_params(theme).items()]
    lines.append("savefig.transparent: True")
    path = directory / "matplotlibrc"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_examples(context: Context) -> None:
    """Run every example script under the theme's style and collect its figures."""
    scripts = list(SCRIPT_FIGURES) if not context.quick else list(SCRIPT_FIGURES)[:1]
    with tempfile.TemporaryDirectory(prefix="optora-figures-") as tmp:
        rc_path = write_matplotlibrc(context.theme, Path(tmp))
        env = {**os.environ, "MPLBACKEND": "Agg", "MATPLOTLIBRC": str(rc_path)}
        for script in scripts:
            print(f"  running examples/{script}", flush=True)
            subprocess.run(
                [sys.executable, str(EXAMPLES / script)],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            for stem in SCRIPT_FIGURES[script]:
                shutil.copyfile(
                    OUTPUTS / f"{stem}.png",
                    context.path(f"{stem}-{context.theme.name}.png"),
                )


FIGURES = (
    FigureSpec(
        name="examples",
        page="examples/index.md",
        outputs=tuple(
            f"{stem}-{{theme}}.png"
            for stems in SCRIPT_FIGURES.values()
            for stem in stems
        ),
        render=run_examples,
    ),
)
