"""Tests for documentation automation."""

from pathlib import Path

from tools.docs import (
    GENERATED_NOTICE,
    ModuleDoc,
    discover_modules,
    display_name,
    generate_api_reference,
    module_symbols,
    outdated_documentation,
)


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


def test_generate_api_reference_documents_discovered_modules(tmp_path: Path) -> None:
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

    outputs = generate_api_reference(tmp_path)

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


def test_generate_api_reference_renders_overview_cards(tmp_path: Path) -> None:
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

    generate_api_reference(tmp_path)

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

    generate_api_reference(tmp_path)

    assert outdated_documentation(tmp_path) == []


def test_generate_api_reference_removes_only_stale_generated_pages(
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

    generate_api_reference(tmp_path)

    assert not stale.exists()
    assert handwritten.exists()
