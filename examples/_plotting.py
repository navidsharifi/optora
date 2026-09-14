import os

from matplotlib.figure import Figure


def save_figure(fig: Figure, name: str) -> str:
    """Save `fig` as a PNG under `examples/outputs/` and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {path}")
    return path
