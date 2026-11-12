"""Clean-environment smoke test: install the built wheel and run every script.

Creates a fresh virtual environment outside the source tree, installs the
wheel for the current ``pyproject.toml`` version with its ``tutorials``
extra, checks that ``optora`` resolves to that installed copy, and runs every
numbered script under ``tutorials/`` and ``examples/`` headlessly. Build the
wheel first::

    python -m build
    python -m tools.smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.release_version import read_project_version

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATTERN = "[0-9][0-9]_*.py"
SCRIPT_DIRECTORIES = ("tutorials", "examples")
CHECK_INSTALLED = (
    "import pathlib, sys, optora; "
    "path = pathlib.Path(optora.__file__).resolve(); "
    "assert pathlib.Path(sys.prefix).resolve() in path.parents, path; "
    "print(f'optora {optora.__version__} imported from {path}')"
)


def find_wheel(root: Path, dist_dir: Path) -> Path:
    """Return the wheel in ``dist_dir`` matching the project version."""
    version = read_project_version(root)
    wheel = dist_dir / f"optora-{version}-py3-none-any.whl"
    if not wheel.is_file():
        msg = f"{wheel} not found; run `python -m build` first"
        raise FileNotFoundError(msg)
    return wheel


def find_scripts(directory: Path) -> list[Path]:
    """Return the numbered scripts under ``directory`` in execution order.

    Scripts may sit in numbered subdirectories, as the grouped examples do,
    so the search is recursive and the order is that of the relative paths.
    """
    scripts = sorted(directory.rglob(SCRIPT_PATTERN))
    if not scripts:
        msg = f"no scripts matching {SCRIPT_PATTERN!r} under {directory}"
        raise FileNotFoundError(msg)
    return scripts


def find_all_scripts(root: Path) -> list[Path]:
    """Return the tutorial scripts followed by the example scripts."""
    return [
        script
        for directory in SCRIPT_DIRECTORIES
        for script in find_scripts(root / directory)
    ]


def venv_python(env_dir: Path) -> Path:
    """Return the interpreter path inside a virtual environment."""
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def isolated_environ(base: Mapping[str, str]) -> dict[str, str]:
    """Return ``base`` without variables that could leak the source tree.

    ``MPLBACKEND`` is forced to ``Agg`` so the tutorials never open a window.
    """
    leaky = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    env = {key: value for key, value in base.items() if key not in leaky}
    env["MPLBACKEND"] = "Agg"
    return env


def run(args: Sequence[str | Path], cwd: Path, env: Mapping[str, str]) -> None:
    """Echo and run a command, raising on a nonzero exit status."""
    print("+", " ".join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def smoke(wheel: Path, scripts: Sequence[Path], torch_index_url: str | None) -> None:
    """Install ``wheel`` into a fresh environment and run ``scripts`` there."""
    env = isolated_environ(os.environ)
    with tempfile.TemporaryDirectory(prefix="optora-smoke-") as tmp:
        work = Path(tmp)
        venv.create(work / "venv", with_pip=True)
        python = venv_python(work / "venv")
        install = [python, "-m", "pip", "install", "--disable-pip-version-check"]
        if torch_index_url is not None:
            run([*install, "torch", "--index-url", torch_index_url], work, env)
        run([*install, f"{wheel.resolve()}[tutorials]"], work, env)
        run([python, "-c", CHECK_INSTALLED], work, env)
        for script in scripts:
            run([python, script.resolve()], work, env)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        help="Directory holding the built wheel. Defaults to <root>/dist.",
    )
    parser.add_argument(
        "--torch-index-url",
        help="Install PyTorch from this index first, e.g. the CPU-only index.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the clean-environment smoke test."""
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    wheel = find_wheel(root, args.dist or root / "dist")
    smoke(wheel, find_all_scripts(root), args.torch_index_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
