"""Run every script under `examples/` and require it to pass its own checks.

Each example asserts the numbers it prints against a closed form or an
independent computation, so a clean exit verifies the example. The scripts
run in a subprocess because they set the default dtype and seed the global
random number generator.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.docs import discover_examples

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = discover_examples(ROOT / "examples")


def test_every_group_has_examples() -> None:
    assert EXAMPLES
    assert len({example.group for example in EXAMPLES}) >= 2


@pytest.mark.parametrize("script", [example.script for example in EXAMPLES])
def test_example_runs_and_passes_its_checks(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-W", "error::SyntaxWarning", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
