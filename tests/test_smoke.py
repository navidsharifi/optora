"""Tests for the clean-environment smoke test helpers."""

from pathlib import Path

import pytest

from tools.smoke import (
    ROOT,
    SCRIPT_DIRECTORIES,
    find_all_scripts,
    find_scripts,
    find_wheel,
    isolated_environ,
    venv_python,
)


def write_pyproject(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "optora"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def test_find_wheel_selects_the_project_version(tmp_path: Path) -> None:
    write_pyproject(tmp_path, "0.2.0")
    dist = tmp_path / "dist"
    dist.mkdir()
    for version in ("0.1.0", "0.2.0"):
        (dist / f"optora-{version}-py3-none-any.whl").touch()

    assert find_wheel(tmp_path, dist) == dist / "optora-0.2.0-py3-none-any.whl"


def test_find_wheel_rejects_a_stale_dist(tmp_path: Path) -> None:
    write_pyproject(tmp_path, "0.2.0")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "optora-0.1.0-py3-none-any.whl").touch()

    with pytest.raises(FileNotFoundError, match="python -m build"):
        find_wheel(tmp_path, dist)


def test_find_scripts_returns_numbered_scripts_in_order(tmp_path: Path) -> None:
    for name in ("02_b.py", "01_a.py", "_plotting.py", "README.md"):
        (tmp_path / name).touch()

    assert [path.name for path in find_scripts(tmp_path)] == ["01_a.py", "02_b.py"]


def test_find_scripts_orders_grouped_scripts_by_group_first(tmp_path: Path) -> None:
    for group, name in (("02_g", "01_c.py"), ("01_g", "02_b.py"), ("01_g", "01_a.py")):
        (tmp_path / group).mkdir(exist_ok=True)
        (tmp_path / group / name).touch()

    found = [path.relative_to(tmp_path).as_posix() for path in find_scripts(tmp_path)]

    assert found == ["01_g/01_a.py", "01_g/02_b.py", "02_g/01_c.py"]


def test_find_scripts_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no scripts"):
        find_scripts(tmp_path)


def test_find_all_scripts_runs_tutorials_before_examples() -> None:
    scripts = find_all_scripts(ROOT)
    owners = [path.relative_to(ROOT).parts[0] for path in scripts]

    assert owners == sorted(owners, key=SCRIPT_DIRECTORIES.index)
    assert set(owners) == set(SCRIPT_DIRECTORIES)
    assert all(path.stem[:2].isdigit() for path in scripts)
    assert ROOT / "tutorials" / "_plotting.py" not in scripts


def test_venv_python_matches_the_platform_layout(tmp_path: Path) -> None:
    python = venv_python(tmp_path)

    assert python.parent.parent == tmp_path
    assert python.parent.name in {"Scripts", "bin"}
    assert python.stem == "python"


def test_isolated_environ_drops_source_tree_leaks() -> None:
    env = isolated_environ(
        {
            "PATH": "/usr/bin",
            "PYTHONPATH": "/repo",
            "PYTHONHOME": "/python",
            "VIRTUAL_ENV": "/repo/.venv",
            "MPLBACKEND": "TkAgg",
        }
    )

    assert env == {"PATH": "/usr/bin", "MPLBACKEND": "Agg"}
