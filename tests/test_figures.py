"""Tests for the documentation figure renderer in `tools.figures`.

The site's figures are generated at build time, so what the tests pin is
the contract between the renderer and the pages: every registered output is
embedded by the page that owns it, every embedded figure is registered, and
a quick render really writes the files it declares.
"""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from tools import figures  # noqa: E402
from tools.figures import examples  # noqa: E402
from tools.figures._base import DARK, LIGHT, THEMES, Context, FigureSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EMBED = re.compile(r"assets/figures/([A-Za-z0-9_.-]+?)(?:#only-(?:light|dark))?[)\"' ]")
SNIPPET = re.compile(r'--8<-- "docs/assets/figures/([^"]+)"')


def referenced_files() -> dict[str, set[str]]:
    """Return, per documentation page, the generated files it references."""
    references: dict[str, set[str]] = {}
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        found = set(EMBED.findall(text)) | set(SNIPPET.findall(text))
        if found:
            references[path.relative_to(DOCS).as_posix()] = found
    return references


def test_figure_names_are_unique() -> None:
    names = [spec.name for spec in figures.FIGURES]

    assert len(names) == len(set(names))


def test_every_registered_output_is_embedded_by_some_page() -> None:
    embedded = set().union(*referenced_files().values())

    for spec in figures.FIGURES:
        missing = set(spec.files(THEMES)) - embedded
        assert not missing, f"{spec.name}: no page embeds {sorted(missing)}"


def test_every_registered_page_exists() -> None:
    for spec in figures.FIGURES:
        assert (DOCS / spec.page).is_file(), f"{spec.name}: {spec.page} is missing"


def test_every_embedded_figure_is_registered() -> None:
    registered = set(figures.expected_files(THEMES))

    for page, files in referenced_files().items():
        unknown = files - registered
        assert not unknown, f"{page} embeds unregistered files {sorted(unknown)}"


def test_every_themed_figure_is_embedded_in_both_schemes() -> None:
    for page, files in referenced_files().items():
        for name in files:
            for theme, other in ((LIGHT, DARK), (DARK, LIGHT)):
                if f"-{theme.name}." in name:
                    twin = name.replace(f"-{theme.name}.", f"-{other.name}.")
                    assert twin in files, f"{page} embeds {name} without {twin}"


def test_files_expand_the_theme_pattern() -> None:
    spec = FigureSpec("x", "index.md", ("x-{theme}.svg", "x.json"), lambda _: None)

    assert spec.themed
    assert spec.files(THEMES) == ["x-light.svg", "x-dark.svg", "x.json"]
    assert not FigureSpec("y", "index.md", ("y.json",), lambda _: None).themed


def test_select_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown figures"):
        figures.select(["no-such-figure"])

    assert [spec.name for spec in figures.select(["kl-tilt"])] == ["kl-tilt"]
    assert figures.select(None) == list(figures.FIGURES)


def test_quick_render_writes_every_declared_file(tmp_path: Path) -> None:
    specs = figures.select(
        ["phi-generators", "kl-tilt", "benchmarks-data", "benchmarks-charts"]
    )

    written = figures.render(specs, [LIGHT], tmp_path, quick=True)

    assert written == [
        tmp_path / name for spec in specs for name in spec.files([LIGHT])
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_examples_matplotlibrc_loads_with_the_theme_colours(tmp_path: Path) -> None:
    path = examples.write_matplotlibrc(DARK, tmp_path)

    loaded = matplotlib.rc_params_from_file(path, use_default_template=False)

    assert loaded["text.color"] == DARK.ink
    assert loaded["axes.prop_cycle"].by_key()["color"] == list(DARK.series)
    assert loaded["savefig.transparent"] is True


def test_context_quick_mode_shrinks_frames_and_sizes(tmp_path: Path) -> None:
    quick = Context(LIGHT, tmp_path, quick=True)
    full = Context(LIGHT, tmp_path)

    assert quick.frames(60) == 6
    assert full.frames(60) == 60
    assert quick.size(256, 16) == 16
    assert full.size(256, 16) == 256
    assert quick.path("a.svg") == tmp_path / "a.svg"
