"""Tests for documentation automation."""

from pathlib import Path

import pytest

from tools.docs import (
    GENERATED_NOTICE,
    ExampleDoc,
    ModuleDoc,
    discover_examples,
    discover_modules,
    display_name,
    generate_documentation,
    module_symbols,
    outdated_documentation,
    render_examples_navigation,
    strip_order_prefix,
)

ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = (
    "nav = [\n"
    "  # BEGIN GENERATED EXAMPLES NAVIGATION\n"
    "  # END GENERATED EXAMPLES NAVIGATION\n"
    "  # BEGIN GENERATED API NAVIGATION\n"
    "  # END GENERATED API NAVIGATION\n"
    "]\n"
)


def write_example(root: Path, group: str, name: str, docstring: str) -> Path:
    """Write a one-line example script under `root/examples/group/name`."""
    path = root / "examples" / group / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'"""{docstring}"""\n\nprint(1)\n', encoding="utf-8")
    return path


def make_site(root: Path) -> None:
    """Create the minimal tree the generator needs: a package and a config."""
    (root / "optora").mkdir()
    (root / "zensical.toml").write_text(NAVIGATION, encoding="utf-8")


def test_module_symbols_returns_public_classes_and_functions(tmp_path: Path) -> None:
    module = tmp_path / "solver.py"
    module.write_text(
        "class Solver:\n    pass\n\n\n"
        "def solve() -> None:\n    pass\n\n\n"
        "def _helper() -> None:\n    pass\n",
        encoding="utf-8",
    )

    assert module_symbols(module) == ("Solver", "solve")


def test_discover_modules_excludes_private_and_package_modules(tmp_path: Path) -> None:
    package = tmp_path / "optora"
    (package / "core").mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "_version.py").touch()
    (package / "core" / "__init__.py").touch()
    (package / "core" / "solver.py").write_text(
        "class Solver:\n    pass\n", encoding="utf-8"
    )

    assert discover_modules(package) == [
        ModuleDoc(
            import_path="optora.core.solver",
            package_parts=("core",),
            name="solver",
            symbols=("Solver",),
        )
    ]


def test_module_doc_places_root_modules_under_optora() -> None:
    module = ModuleDoc(
        import_path="optora.minimize",
        package_parts=(),
        name="minimize",
        symbols=(),
    )

    assert module.relative_path == Path("optora/minimize.md")


def test_display_name_preserves_domain_acronyms() -> None:
    assert display_name("dro") == "DRO"
    assert display_name("kl_dro") == "KL-DRO"
    assert display_name("f_divergence") == "f-Divergence"


def test_generate_documentation_documents_discovered_modules(tmp_path: Path) -> None:
    package = tmp_path / "optora"
    (package / "solvers").mkdir(parents=True)
    (package / "solvers" / "gradient_descent.py").write_text(
        "class GradientDescent:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "zensical.toml").write_text(
        "nav = [\n"
        "  # BEGIN GENERATED API NAVIGATION\n"
        "  # END GENERATED API NAVIGATION\n"
        "]\n",
        encoding="utf-8",
    )

    outputs = generate_documentation(tmp_path)

    module_page = tmp_path / "docs" / "api" / "solvers" / "gradient_descent.md"
    assert module_page in outputs
    assert "::: optora.solvers.gradient_descent" in module_page.read_text(
        encoding="utf-8"
    )
    package_index = (tmp_path / "docs" / "api" / "solvers" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "[`optora.solvers.gradient_descent`](gradient_descent.md)" in package_index
    assert (
        "[`GradientDescent`](gradient_descent.md"
        "#optora.solvers.gradient_descent.GradientDescent)"
    ) in package_index
    navigation = (tmp_path / "zensical.toml").read_text(encoding="utf-8")
    assert '"Solvers" = [' in navigation
    assert '"Gradient Descent" = "api/solvers/gradient_descent.md"' in navigation


def test_generate_documentation_renders_overview_cards(tmp_path: Path) -> None:
    package = tmp_path / "optora"
    (package / "solvers").mkdir(parents=True)
    (package / "solvers" / "gradient_descent.py").touch()
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "zensical.toml").write_text(
        "nav = [\n"
        "  # BEGIN GENERATED API NAVIGATION\n"
        "  # END GENERATED API NAVIGATION\n"
        "]\n",
        encoding="utf-8",
    )

    generate_documentation(tmp_path)

    overview = (tmp_path / "docs" / "api" / "index.md").read_text(encoding="utf-8")
    assert overview.startswith("---\n")
    assert GENERATED_NOTICE in overview
    assert '<div class="grid cards" markdown>' in overview
    assert "__`optora.solvers`__" in overview
    assert "[:octicons-arrow-right-24: Browse](solvers/index.md)" in overview


def test_outdated_documentation_detects_and_clears_drift(tmp_path: Path) -> None:
    package = tmp_path / "optora"
    (package / "solvers").mkdir(parents=True)
    (package / "solvers" / "gradient_descent.py").touch()
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "zensical.toml").write_text(
        "nav = [\n"
        "  # BEGIN GENERATED API NAVIGATION\n"
        "  # END GENERATED API NAVIGATION\n"
        "]\n",
        encoding="utf-8",
    )

    assert outdated_documentation(tmp_path)

    generate_documentation(tmp_path)

    assert outdated_documentation(tmp_path) == []


def test_generate_documentation_removes_only_stale_generated_pages(
    tmp_path: Path,
) -> None:
    package = tmp_path / "optora"
    package.mkdir()
    api_root = tmp_path / "docs" / "api"
    api_root.mkdir(parents=True)
    stale = api_root / "stale.md"
    stale.write_text(f"{GENERATED_NOTICE}\n", encoding="utf-8")
    handwritten = api_root / "handwritten.md"
    handwritten.write_text("# Keep me\n", encoding="utf-8")
    (tmp_path / "zensical.toml").write_text(
        "nav = [\n"
        "  # BEGIN GENERATED API NAVIGATION\n"
        "  # END GENERATED API NAVIGATION\n"
        "]\n",
        encoding="utf-8",
    )

    generate_documentation(tmp_path)

    assert not stale.exists()
    assert handwritten.exists()


def test_strip_order_prefix_drops_the_two_digit_prefix() -> None:
    assert strip_order_prefix("02_ambiguity_sets") == "ambiguity_sets"
    assert strip_order_prefix("10_kl_ball") == "kl_ball"


@pytest.mark.parametrize("name", ["kl_ball", "1_kl_ball", "01-kl-ball", "01_"])
def test_strip_order_prefix_rejects_unnumbered_names(name: str) -> None:
    with pytest.raises(ValueError, match="two-digit prefix"):
        strip_order_prefix(name)


def test_discover_examples_reads_title_and_summary_in_order(tmp_path: Path) -> None:
    write_example(tmp_path, "02_solvers", "01_descent.py", "Descent.\n\nRuns it.\n")
    write_example(
        tmp_path, "01_sets", "02_mmd.py", "MMD ball.\n\nFirst line.\nSecond.\n"
    )
    write_example(tmp_path, "01_sets", "01_kl.py", "KL ball.\n\nSolves it.\n")
    (tmp_path / "examples" / "01_sets" / "_helper.py").touch()
    (tmp_path / "examples" / "README.md").touch()

    examples = discover_examples(tmp_path / "examples")

    assert examples == [
        ExampleDoc(
            script="examples/01_sets/01_kl.py",
            group="sets",
            name="kl",
            title="KL ball",
            summary="Solves it.",
        ),
        ExampleDoc(
            script="examples/01_sets/02_mmd.py",
            group="sets",
            name="mmd",
            title="MMD ball",
            summary="First line.\nSecond.",
        ),
        ExampleDoc(
            script="examples/02_solvers/01_descent.py",
            group="solvers",
            name="descent",
            title="Descent",
            summary="Runs it.",
        ),
    ]


def test_discover_examples_requires_a_docstring(tmp_path: Path) -> None:
    path = tmp_path / "examples" / "01_sets" / "01_kl.py"
    path.parent.mkdir(parents=True)
    path.write_text("print(1)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="module docstring"):
        discover_examples(tmp_path / "examples")


def test_example_doc_labels_groups_in_sentence_case() -> None:
    example = ExampleDoc(
        script="examples/02_ambiguity_sets/01_kl.py",
        group="ambiguity_sets",
        name="kl",
        title="KL ball",
        summary="",
    )

    assert example.group_label == "Ambiguity sets"
    assert example.relative_path == Path("ambiguity_sets/kl.md")


def test_render_examples_navigation_escapes_titles() -> None:
    example = ExampleDoc(
        script="examples/01_a/01_b.py",
        group="a",
        name="b",
        title='Say "hi"',
        summary="",
    )

    navigation = render_examples_navigation([example])

    assert '{ "Say \\"hi\\"" = "examples/a/b.md" },' in navigation
    assert navigation.startswith("  # BEGIN GENERATED EXAMPLES NAVIGATION")
    assert navigation.endswith("  # END GENERATED EXAMPLES NAVIGATION")


def test_generate_documentation_writes_example_pages_and_navigation(
    tmp_path: Path,
) -> None:
    make_site(tmp_path)
    write_example(
        tmp_path, "01_sets", "01_kl.py", "KL ball.\n\nSolves the `KL` ball.\n"
    )

    outputs = generate_documentation(tmp_path)

    page = tmp_path / "docs" / "examples" / "sets" / "kl.md"
    assert page in outputs
    text = page.read_text(encoding="utf-8")
    assert text.startswith(GENERATED_NOTICE)
    assert "# KL ball\n\nSolves the `KL` ball.\n" in text
    assert "python examples/01_sets/01_kl.py" in text
    assert '--8<-- "examples/01_sets/01_kl.py"' in text
    overview = (tmp_path / "docs" / "examples" / "index.md").read_text(encoding="utf-8")
    assert "## Sets" in overview
    assert "| [KL ball](sets/kl.md) | Solves the `KL` ball. |" in overview
    navigation = (tmp_path / "zensical.toml").read_text(encoding="utf-8")
    assert '{ "Sets" = [' in navigation
    assert '{ "KL ball" = "examples/sets/kl.md" },' in navigation
    assert outdated_documentation(tmp_path) == []


def test_generate_documentation_removes_stale_example_pages(tmp_path: Path) -> None:
    make_site(tmp_path)
    script = write_example(tmp_path, "01_sets", "01_kl.py", "KL ball.\n\nSolves it.\n")
    generate_documentation(tmp_path)
    page = tmp_path / "docs" / "examples" / "sets" / "kl.md"
    script.rename(script.with_name("01_kl_divergence_ball.py"))

    assert page in outdated_documentation(tmp_path)

    generate_documentation(tmp_path)

    assert not page.exists()
    assert (tmp_path / "docs" / "examples" / "sets" / "kl_divergence_ball.md").exists()


def test_generate_documentation_requires_the_examples_navigation_block(
    tmp_path: Path,
) -> None:
    make_site(tmp_path)
    (tmp_path / "zensical.toml").write_text(
        "nav = [\n"
        "  # BEGIN GENERATED API NAVIGATION\n"
        "  # END GENERATED API NAVIGATION\n"
        "]\n",
        encoding="utf-8",
    )
    write_example(tmp_path, "01_sets", "01_kl.py", "KL ball.\n\nSolves it.\n")

    with pytest.raises(RuntimeError, match="EXAMPLES nav block"):
        generate_documentation(tmp_path)


def test_repository_documentation_is_up_to_date() -> None:
    assert outdated_documentation(ROOT) == []
