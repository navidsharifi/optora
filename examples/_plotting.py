"""Shared plotting helper for the scripts in this folder.

Figures are written under ``examples/outputs/``, which is git-ignored (see
``.gitignore``) -- nothing produced here is meant to be committed. Any
figure worth publishing in documentation should be re-hosted externally
(CDN / image host) and linked from ``examples/README.md`` instead of
committed as a binary blob; see the "Media asset policy" section there.
"""

from pathlib import Path

from matplotlib.figure import Figure

OUTPUT_DIR = Path(__file__).parent / "outputs"


def save_figure(fig: Figure, name: str) -> Path:
    """Save `fig` as a PNG under `examples/outputs/` and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {path}")
    return path
