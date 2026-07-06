"""Tests for release version automation."""

from pathlib import Path

from tools.release_version import (
    Version,
    choose_bump,
    next_version,
    update_release_files,
    update_version_block,
)


def test_choose_bump_prefers_major() -> None:
    assert choose_bump("add solver #minor #major") == "major"


def test_choose_bump_defaults_to_patch() -> None:
    assert choose_bump("fix line search tolerance") == "patch"


def test_next_version_minor_resets_patch() -> None:
    version, bump = next_version(Version.parse("v0.1.4"), "restore docs #minor")

    assert version == Version(0, 2, 0)
    assert bump == "minor"


def test_next_version_major_resets_minor_and_patch() -> None:
    version, bump = next_version(Version.parse("v0.7.9"), "stabilize api #major")

    assert version == Version(1, 0, 0)
    assert bump == "major"


def test_update_version_block(tmp_path: Path) -> None:
    path = tmp_path / "index.md"
    path.write_text(
        "# Optora\n\n"
        "<!-- optora-version-start -->\n"
        "Latest release: `v0.1.0`\n"
        "<!-- optora-version-end -->\n",
        encoding="utf-8",
    )

    changed = update_version_block(path, Version(0, 1, 1))

    assert changed == path
    assert "Latest release: `v0.1.1`" in path.read_text(encoding="utf-8")


def test_release_files_include_documentation_release_page(tmp_path: Path) -> None:
    (tmp_path / "optora").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "optora"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "optora" / "_version.py").write_text(
        '"""Package version for Optora."""\n\n__version__ = "0.1.0"\n',
        encoding="utf-8",
    )
    for docs_file in ("index.md", "release.md"):
        (tmp_path / "docs" / docs_file).write_text(
            "# Optora\n\n"
            "<!-- optora-version-start -->\n"
            "Latest release: `v0.1.0`\n"
            "<!-- optora-version-end -->\n",
            encoding="utf-8",
        )

    changed = update_release_files(tmp_path, Version(0, 1, 1))

    assert sorted(path.relative_to(tmp_path).as_posix() for path in changed) == [
        "docs/index.md",
        "docs/release.md",
        "optora/_version.py",
        "pyproject.toml",
    ]
    assert "Latest release: `v0.1.1`" in (tmp_path / "docs" / "release.md").read_text(
        encoding="utf-8"
    )
