"""Tests that `CITATION.cff` agrees with the package metadata it repeats.

The installed distribution metadata is what PyPI shows, so it is the
reference; the citation file must never contradict it.
"""

from importlib.metadata import PackageMetadata, metadata
from pathlib import Path
from typing import Any

import pytest
import yaml

CITATION = Path(__file__).resolve().parents[1] / "CITATION.cff"


@pytest.fixture(scope="module")
def citation() -> dict[str, Any]:
    """Return the parsed `CITATION.cff`."""
    parsed: dict[str, Any] = yaml.safe_load(CITATION.read_text(encoding="utf-8"))
    return parsed


@pytest.fixture(scope="module")
def package() -> PackageMetadata:
    """Return the installed `optora` distribution metadata."""
    return metadata("optora")


def test_declares_the_required_keys(citation: dict[str, Any]) -> None:
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["message"]
    assert citation["authors"]


def test_every_author_is_a_person_or_an_entity(citation: dict[str, Any]) -> None:
    for author in citation["authors"]:
        assert {"family-names", "given-names"} <= author.keys() or "name" in author


def test_title_and_abstract_match_the_package(
    citation: dict[str, Any], package: PackageMetadata
) -> None:
    assert citation["title"].casefold() == package["Name"].casefold()
    assert citation["abstract"] == package["Summary"]


def test_license_matches_the_package(
    citation: dict[str, Any], package: PackageMetadata
) -> None:
    assert citation["license"] == package["License-Expression"]


def test_urls_match_the_package(
    citation: dict[str, Any], package: PackageMetadata
) -> None:
    urls = dict(entry.split(", ", 1) for entry in package.get_all("Project-URL", []))

    assert citation["repository-code"] == urls["Repository"]
    assert citation["url"] == urls["Documentation"]


def test_version_is_left_to_the_release_tag(citation: dict[str, Any]) -> None:
    # Zenodo reads the version and date from the release tag; a copy in the
    # citation file would go stale with every release.
    assert "version" not in citation
    assert "date-released" not in citation
