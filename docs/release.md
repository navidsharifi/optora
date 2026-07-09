---
icon: lucide/package-check
---

# Release workflow

<!-- optora-version-start -->
Latest release: `v0.0.1`
<!-- optora-version-end -->

Optora uses automated SemVer releases. Pull requests preview the next version,
and merges to `main` perform the release only when the final commit message
contains `#release`.

## Version markers

The release workflow reads the latest commit message on the branch being
released.

| Commit marker | Version change | Example from `v0.1.4` |
| --- | --- | --- |
| `#major` | major | `v1.0.0` |
| `#minor` | minor | `v0.2.0` |
| no marker | patch | `v0.1.5` |

If both `#major` and `#minor` appear, `#major` wins.

## Release gate

Publishing requires `#release` in the latest commit message. Combine it with
the version marker when needed:

| Commit marker | Release behavior |
| --- | --- |
| `#release` | publish a patch release |
| `#release #minor` | publish a minor release |
| `#release #major` | publish a major release |
| no `#release` | do not publish |

Release preparation can only be triggered by the configured release actor. Set
the GitHub Actions repository variable `PYPI_RELEASE_ACTOR` to the allowed
GitHub username. If the variable is not set, the workflow falls back to the
repository owner.

## Maintainer setup

1. Enable GitHub Pages for the repository and set the Pages build source to
   GitHub Actions.
2. Until Pages is enabled, the documentation workflow still builds the site and
   skips deployment successfully.
3. Create or confirm the GitHub environment named `pypi`.
4. Set `PYPI_RELEASE_ACTOR` in GitHub Actions repository variables when the
   release actor is not the repository owner.
5. In PyPI, add a trusted publisher for the GitHub repository, workflow
   `.github/workflows/release.yml`, and environment `pypi`.
6. Merge changes through pull requests. Use squash merges when you want the
   release markers to be unambiguous in the final commit message.
7. Include `#release` in the final commit message only when the change should
   publish to PyPI.
8. Let GitHub Actions build, test, create the release commit and tag, publish
   the tagged package to PyPI, publish a GitHub release, and redeploy the
   documentation site.

Publishing runs from the pushed `vX.Y.Z` tag. If PyPI configuration is wrong
on the first attempt, fix the trusted-publisher settings and rerun the failed
tag workflow instead of creating a new version.
