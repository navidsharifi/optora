"""Release version calculation and file updates for Optora."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Bump = Literal["major", "minor", "patch"]

SEMVER_RE = re.compile(
    r"^v?(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)$"
)
PYPROJECT_VERSION_RE = re.compile(r'(?m)^version = "(?P<version>[^"]+)"$')
VERSION_BLOCK_RE = re.compile(
    r"<!-- optora-version-start -->.*?<!-- optora-version-end -->",
    re.DOTALL,
)


@dataclass(frozen=True, order=True, slots=True)
class Version:
    """A strict three-part semantic version."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> Version:
        """Parse a version string with an optional leading ``v``."""
        match = SEMVER_RE.fullmatch(value.strip())
        if match is None:
            msg = f"expected a semantic version like 0.1.0 or v0.1.0, got {value!r}"
            raise ValueError(msg)
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def bumped(self, bump: Bump) -> Version:
        """Return the next version for the requested SemVer bump."""
        if bump == "major":
            return Version(self.major + 1, 0, 0)
        if bump == "minor":
            return Version(self.major, self.minor + 1, 0)
        return Version(self.major, self.minor, self.patch + 1)

    @property
    def tag(self) -> str:
        """Return the Git tag form of the version."""
        return f"v{self}"

    def __str__(self) -> str:
        """Return the PEP 440-compatible version string."""
        return f"{self.major}.{self.minor}.{self.patch}"


def choose_bump(commit_message: str) -> Bump:
    """Choose the version bump from a commit message."""
    normalized = commit_message.lower()
    if "#major" in normalized:
        return "major"
    if "#minor" in normalized:
        return "minor"
    return "patch"


def next_version(base_version: Version, commit_message: str) -> tuple[Version, Bump]:
    """Calculate the next version and bump kind from the base version."""
    bump = choose_bump(commit_message)
    return base_version.bumped(bump), bump


def read_project_version(root: Path) -> Version:
    """Read the static project version from ``pyproject.toml``."""
    pyproject = root / "pyproject.toml"
    match = PYPROJECT_VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if match is None:
        msg = "could not find [project] version in pyproject.toml"
        raise RuntimeError(msg)
    return Version.parse(match.group("version"))


def latest_tag_version(root: Path) -> Version | None:
    """Return the highest local ``vX.Y.Z`` tag, if Git metadata is available."""
    result = run_git(root, ["tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"], check=False)
    if result.returncode != 0:
        return None

    versions: list[Version] = []
    for tag in result.stdout.splitlines():
        try:
            versions.append(Version.parse(tag))
        except ValueError:
            continue
    return max(versions) if versions else None


def resolve_base_version(root: Path, explicit: str | None = None) -> Version:
    """Resolve the release base version from an argument, Git tag, or metadata."""
    if explicit is not None:
        return Version.parse(explicit)
    return latest_tag_version(root) or read_project_version(root)


def read_head_commit_message(root: Path) -> str:
    """Read the full message for ``HEAD``."""
    result = run_git(root, ["log", "-1", "--pretty=%B"], check=False)
    if result.returncode != 0:
        msg = "could not read the latest Git commit message"
        raise RuntimeError(msg)
    return result.stdout.strip()


def tag_exists(root: Path, tag: str) -> bool:
    """Return whether a Git tag already exists."""
    result = run_git(root, ["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"])
    return result.returncode == 0


def update_release_files(root: Path, version: Version) -> list[Path]:
    """Update version-bearing files before a release commit."""
    changed = [
        update_pyproject(root / "pyproject.toml", version),
        update_version_module(root / "optora" / "_version.py", version),
        update_version_block(root / "docs" / "index.md", version),
        update_version_block(root / "docs" / "release.md", version),
    ]
    return [path for path in changed if path is not None]


def update_pyproject(path: Path, version: Version) -> Path | None:
    """Update the package version in ``pyproject.toml``."""
    text = path.read_text(encoding="utf-8")
    updated, count = PYPROJECT_VERSION_RE.subn(
        f'version = "{version}"',
        text,
        count=1,
    )
    if count != 1:
        msg = f"could not update version in {path}"
        raise RuntimeError(msg)
    return write_if_changed(path, updated)


def update_version_module(path: Path, version: Version) -> Path | None:
    """Update the importable package version module."""
    text = f'"""Package version for Optora."""\n\n__version__ = "{version}"\n'
    return write_if_changed(path, text)


def update_version_block(path: Path, version: Version) -> Path | None:
    """Update a marked latest-version block in a Markdown file."""
    text = path.read_text(encoding="utf-8")
    block = (
        "<!-- optora-version-start -->\n"
        f"Latest release: `{version.tag}`\n"
        "<!-- optora-version-end -->"
    )
    updated, count = VERSION_BLOCK_RE.subn(block, text, count=1)
    if count != 1:
        msg = f"could not find optora version marker block in {path}"
        raise RuntimeError(msg)
    return write_if_changed(path, updated)


def write_if_changed(path: Path, content: str) -> Path | None:
    """Write a file only when the content has changed."""
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == content:
        return None
    path.write_text(content, encoding="utf-8")
    return path


def run_git(
    root: Path,
    args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command in the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def emit_outputs(values: dict[str, str]) -> None:
    """Write GitHub Actions outputs when running inside Actions."""
    for key, value in values.items():
        print(f"{key}={value}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path is None:
        return

    with Path(output_path).open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--base-version",
        help="Explicit base version. Defaults to the highest vX.Y.Z tag.",
    )
    parser.add_argument(
        "--message",
        help="Explicit commit message. Defaults to the latest Git commit message.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preview", help="Print the next release version.")
    subparsers.add_parser("bump", help="Update release files for the next version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release versioning command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = resolve_base_version(root, args.base_version)
    message = (
        args.message if args.message is not None else read_head_commit_message(root)
    )
    version, bump = next_version(base, message)

    if args.command == "bump":
        if tag_exists(root, version.tag):
            parser.error(f"tag {version.tag} already exists")
        changed = update_release_files(root, version)
        changed_files = ",".join(path.relative_to(root).as_posix() for path in changed)
    else:
        changed_files = ""

    emit_outputs(
        {
            "base-version": str(base),
            "bump": bump,
            "version": str(version),
            "tag": version.tag,
            "changed-files": changed_files,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
