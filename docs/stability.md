---
icon: lucide/shield-check
---

# API stability

Optora follows [Semantic Versioning 2.0.0](https://semver.org/). Before
`1.0.0`, SemVer itself promises nothing (its rule 4: "anything MAY change at
any time"), so this page states what Optora commits to during `0.x`.

## Public API

The public API is:

- every name in `optora.__all__`, and in the `__all__` of `optora.core`,
  `optora.divergences`, `optora.dro`, and `optora.solvers`;
- for those names, the constructor arguments, methods, attributes, and
  behaviour documented in the [API reference](api/index.md) and on
  [Numerical behaviour](numerics.md);
- `optora.__version__`.

Everything else is internal and may change in any release: names and modules
with a leading underscore (including underscore methods that subclasses of
`AmbiguitySet`, `Divergence`, or `Solver` inherit), names missing from every
`__all__`, and the repository's `tests/`, `tools/`, and `examples/`.

!!! note "Extending the contracts"

    The abstract methods of `Solver`, `Divergence`, and `AmbiguitySet` are
    public: a subclass that implements them keeps working within a minor
    series. A subclass that calls an inherited underscore helper relies on
    an internal detail.

## Versioning during 0.x

| Release | Example | May contain |
| --- | --- | --- |
| patch `0.y.Z` | `0.1.0` to `0.1.1` | bug fixes, performance work, documentation; no incompatible change to the public API |
| minor `0.Y.0` | `0.1.4` to `0.2.0` | new features and incompatible changes to the public API |
| major `1.0.0` | `0.9.2` to `1.0.0` | declares the public API stable; from then on an incompatible change requires a major release |

An incompatible change is marked in its pull request title with a
Conventional Commits `!` (for example `feat(dro)!: rename radius to budget`),
so it appears in the release notes GitHub generates from merged pull
requests.

Where feasible, a public name or argument is deprecated for at least one
minor release before it is removed: it keeps working and emits a
`DeprecationWarning` that names its replacement. A change that fixes an
incorrect result is not deprecated first.

!!! tip "Pinning a dependency"

    To accept fixes but no incompatible change, pin the minor series with a
    compatible-release specifier, for example `optora~=0.1.0`
    (equivalent to `>=0.1.0, <0.2`).

## Numerical results

Values are reproducible to the tolerances stated on
[Numerical behaviour](numerics.md), not bit for bit across releases,
platforms, or PyTorch versions. A patch release may change a value when it
fixes a defect, and its release notes say so.

## Supported environments

The minimum supported Python and PyTorch versions (`requires-python` and the
`torch` requirement in `pyproject.toml`) are raised only in a minor release.
CI tests both the minimum and the latest PyTorch on Linux, macOS, and
Windows.
