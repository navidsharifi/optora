# AGENTS.md

## Required Workflow

- Before searching or reading anything else in the codebase, read `router.md` if it exists at the repository root. If it does not exist, read `progress/router.md`.
- Read the relevant files in `progress/` before changing code or docs.
- Keep `progress/working_memory.md` updated aggressively while work is in progress.
- Keep `progress/working_memory.md` scoped to the current job only. Clear it after a standalone task is complete unless the work is part of a larger active scope.
- Keep the rest of `progress/` accurate as implementation details, status, or decisions change.
- Do not commit files. Only suggest git commands.
- Suggested commit messages must be one-line, lowercase, and follow Angular convention, for example:
  `git add path/to/file && git commit -m "docs(progress): update agent routing notes"`
- Never include `coauthored by` in suggested commit messages.

- never commit messages!
- only suggest files to be committed (full path) and one line tldr commit message in angluar convention (single line optimally.) eg, git add path/to/file , git commit -m "feat(message): one line tldr all lowercase in angular convention"

- never mention things like coauthored by etc.! never inclue commit ids in commit messages! or pr messages! pr other unreadable or irrelevant info.

## Project Priorities (Extremely Important)

Optora is a machine learning optimization library. The current public direction is a small gpu-first deterministic optimization core that can later grow into stochastic methods, differentiable backends, optimal transport, and reinforcement learning.

Write clean, maintainable, standard Python. Prefer composable modules, small public APIs, explicit interfaces, and reusable abstractions already present in the project. Do not duplicate parallel implementations when an existing component can be extended cleanly.

This is a performance-sensitive, GPU-forward library. Avoid patterns that would block future accelerator backends:

- Avoid unnecessary CPU synchronizations and scalar round-trips.
- Prefer vectorized array operations over Python loops.
- Keep solver interfaces backend-friendly.
- Avoid hidden copies unless they protect correctness.
- Keep numerical work high throughput and shape-aware.

Tests must mirror optora architecture. Use pytest and write standard tests aggressively and cover all edge cases to avoid regressions across different scale.

Prefer standard abstraction architectures only when it is necessary and is a standard practise. Pay attention and avoid unnecessary and excessive nesting.

Always istall new libraries and dependencies inside the project virtual environment .venv

Always activate .venv before running anything.

Avoid distructive commands eg, deletions or non reversible commands.

Always use academic standards for terminology and taxonomy.

avoid over complicating things and ensure everything is super scalable and maintainable. always prefer clean, simple (but keep all functionalities), and standard code.

never commit progress (tracker) folder

always run all tests before pr etc. including pytest, ruff check, ruff format, mypy, etc

## Python Style

- Use type hints consistently.
- Write Google-style docstrings for public modules, classes, functions, and nontrivial private helpers.
- Keep docstrings precise and useful.
- Avoid inline comments unless they clarify non-obvious numerical logic.
- Follow the configured tooling in `pyproject.toml`: Ruff, mypy strict mode, and pytest.
- Prefer ASCII unless a file already requires another character set.


## Repository Map

- `README.md`: project overview, first usage example, development commands, and roadmap.
- `pyproject.toml`: package metadata, dependencies, and tool configuration.
- `optora/minimize.py`: public `minimize(...)` interface and solver selection.
- `optora/core/objective.py`: objective wrapper and finite-difference gradients.
- `optora/core/result.py`: optimization result container.
- `optora/solvers/base.py`: solver interface.
- `optora/solvers/line_search.py`: shared line-search routines.
- `optora/solvers/deterministic/gradient_descent.py`: gradient descent solver.
- `optora/solvers/deterministic/bfgs.py`: BFGS solver.
- `tests/`: pytest coverage mirroring the package architecture.
- `progress/router.md`: first-read routing guide for agents.
- `progress/architecture.md`: current architecture notes.
- `progress/decisions.md`: durable design decisions.
- `progress/status.md`: current project status and known gaps.
- `progress/changelog.md`: dated progress log.
- `progress/notes.md`: short-lived observations that may become decisions or tasks.
- `progress/working_memory.md`: current active job only.

## Current Caveats

The package now uses a root-level `optora/` layout rather than `src/optora/`.
Always run tools through `.venv` and keep package discovery, docs paths, and developer commands aligned with that layout.
