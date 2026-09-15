# Contributing to Optora

Thank you for your interest in contributing to Optora! We welcome bug fixes,
documentation improvements, new examples, performance work, and new solvers,
divergences, or ambiguity sets.

> If Optora is useful for your work, please consider starring the repository.
> It is the main way people discover the project.

## Quick Start

Clone your fork and install the development dependencies:

```powershell
git clone https://github.com/<your-github-username>/optora.git
cd optora
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,docs]"
```

## Development Workflow

1. Claim the issue. Check that nobody is assigned and no open Pull Request
   already links it, then comment on the issue to say you are taking it.
   Pull Requests that duplicate an earlier one for the same issue are closed.
2. Create a new branch from `main`.
3. Make your changes.
4. Format your code (`ruff format .`) and lint it (`ruff check .`).
5. Type-check with `mypy` (strict mode).
6. Run the test suite (`pytest`).
7. Commit using Conventional Commits.
8. Open a Pull Request linked to the relevant issue.

## Dependencies

`torch` is the only runtime dependency, and `optora/` imports nothing else.
Data loaders, plotting, and example scripts belong under `examples/`, not the
library.

The optional extras (`dev`, `docs`, `examples`) serve the repository, never
the library at runtime. A new runtime extra is justified only when a module
under `optora/` needs it, and then it must:

- import the package lazily at the call site, never at module import time;
- raise an `ImportError` naming the extra to install;
- be named after the capability it unlocks, not after a paper or algorithm.

## Coding Style and Rules

- Surgical diffs only. Do not reformat, reorder imports, or touch lines
  unrelated to the issue you are closing.
- Use type hints consistently; type annotations go in the signature, not the
  docstring.
- Write Google-style docstrings (`r"""..."""` for any docstring containing a
  LaTeX backslash). Math is standard LaTeX: `$...$` inline, `$$...$$` block.
  Never ASCII pseudo-math.
- Avoid inline comments unless they clarify non-obvious numerical logic.
- Never modify a shared ABC (`Solver`, `Divergence`, `AmbiguitySet`) unless
  the issue names it.
- Never add an external dependency.
- This is a GPU-forward, performance-sensitive library: avoid host syncs
  (`.item()`, `.cpu()`, Python `and`/`or` on tensors), unnecessary copies,
  and Python loops over tensor batches. Prefer vectorized, shape-aware
  operations.
- Reuse existing abstractions instead of adding a parallel implementation.
  Avoid redundant or unnecessarily verbose code and docstrings.
- Never touch GitHub workflows, CI/CD, or `.yml` files.
- Write pytest tests for your change in the same style and convention as the
  rest of the `tests/` tree, covering edge cases across scale.
- Ensure all tests, `ruff check`, `ruff format --check`, and `mypy` pass
  before opening a Pull Request.

## Pull Request Checklist

Before opening a Pull Request, ensure that:

- No earlier open Pull Request targets the same issue.
- Tests pass (`pytest`).
- Code is formatted and linted (`ruff format --check .`, `ruff check .`).
- Type checks pass (`mypy`).
- Commit messages follow the Conventional Commits format.
- **No AI attribution in commits, PR title, or PR description**: no
  `Co-authored-by` trailers for AI tools, no "generated with" footers, no
  mentions of assistants. Trailers land in the contributors graph and PR
  text lands in the changelog; both must name people only.
- Related issues are linked when applicable.

## Note

If you notice something adjacent while working (a bug, a missing test, an
unclear docstring), do not expand your PR to cover it. Open a separate issue
using a terse scope and a `Done when:` line, and link it from your PR.

## Support the Project

Beyond code, the most useful things you can do are to star the repository,
open an issue when something is unclear or broken, and cite Optora if it
supports published work.
