"""Tests for the top-level ``optora`` export policy."""

import importlib
import inspect
import subprocess
import sys

import pytest

import optora
from optora import _version

SUBPACKAGES = ("core", "divergences", "solvers", "dro")

# Subpackage exports deliberately kept off the root namespace: convergence
# diagnostics and helper functions are reached through ``optora.core``.
NOT_REEXPORTED = frozenset(
    {
        "ConvergenceDiagnostics",
        "ConvergenceStatus",
        "ConvergenceTracker",
        "require_gradient",
        "validate_check_interval",
    }
)

# Root-level workflow dispatchers are forbidden by the naming vocabulary.
FORBIDDEN_ROOT_FUNCTIONS = ("minimize", "minimize_risk", "decide", "train", "run")


def _subpackage_exports() -> dict[str, str]:
    """Map every subpackage export name to the subpackage that owns it."""
    owners: dict[str, str] = {}
    for name in SUBPACKAGES:
        module = importlib.import_module(f"optora.{name}")
        for symbol in module.__all__:
            assert symbol not in owners, f"{symbol} exported by two subpackages"
            owners[symbol] = name
    return owners


def test_version_matches_version_module() -> None:
    assert optora.__version__ == _version.__version__


def test_all_is_sorted_and_unique() -> None:
    assert len(set(optora.__all__)) == len(optora.__all__)
    assert optora.__all__ == sorted(optora.__all__)


@pytest.mark.parametrize("name", optora.__all__)
def test_every_listed_name_resolves(name: str) -> None:
    assert hasattr(optora, name)


def test_star_import_exposes_exactly_all() -> None:
    namespace: dict[str, object] = {}
    exec("from optora import *", namespace)  # noqa: S102
    exported = set(namespace) - {"__builtins__"}

    assert exported == set(optora.__all__)


@pytest.mark.parametrize("symbol", sorted(_subpackage_exports()))
def test_subpackage_nouns_are_the_same_objects(symbol: str) -> None:
    owner = _subpackage_exports()[symbol]
    if symbol in NOT_REEXPORTED:
        assert symbol not in optora.__all__
        return

    assert getattr(optora, symbol) is getattr(
        importlib.import_module(f"optora.{owner}"), symbol
    )


def test_root_exports_nothing_beyond_subpackage_nouns() -> None:
    expected = (set(_subpackage_exports()) - NOT_REEXPORTED) | {"__version__"}

    assert set(optora.__all__) == expected


def test_root_exports_only_classes_and_version() -> None:
    for name in optora.__all__:
        if name == "__version__":
            continue
        assert inspect.isclass(getattr(optora, name)), name


@pytest.mark.parametrize("name", FORBIDDEN_ROOT_FUNCTIONS)
def test_no_root_workflow_dispatchers(name: str) -> None:
    assert not hasattr(optora, name)


@pytest.mark.parametrize("subpackage", ["optora.core", "optora.solvers", "optora.dro"])
def test_subpackages_import_standalone_in_a_fresh_interpreter(
    subpackage: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {subpackage}"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_root_import_solves_end_to_end() -> None:
    import torch

    nominal = torch.full((4,), 0.25, dtype=torch.float64)
    loss = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float64)
    ambiguity_set = optora.KLAmbiguitySet(
        nominal=nominal,
        radius=0.1,
        dual_solver=optora.GradientDescent(step_size=0.1, max_iter=300, tol=1e-7),
    )

    worst_case = ambiguity_set.worst_case_expectation(loss)

    assert isinstance(ambiguity_set, optora.AmbiguitySet)
    assert worst_case >= (nominal * loss).sum()
    assert worst_case <= loss.max()
