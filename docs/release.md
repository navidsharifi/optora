---
icon: lucide/package-check
---

# Release workflow

<!-- optora-version-start -->
Latest release: `v0.2.0`
<!-- optora-version-end -->

Optora releases are automated and follow SemVer. A pull request previews the
next version, and a merge to `main` publishes a release only when the final
commit message contains `#release`. [API stability](stability.md) defines
which marker a change needs during `0.x`.

## Version markers

The release workflow reads the latest commit message on the branch being
released.

| Commit marker | Version change | Example from `v0.1.4` |
| --- | --- | --- |
| `#major` | major | `v1.0.0` |
| `#minor` | minor | `v0.2.0` |
| no marker | patch | `v0.1.5` |

If both `#major` and `#minor` appear, `#major` takes precedence.

## Release gate

Publishing requires `#release` in the latest commit message. Combine it with
a version marker when the release should not be a patch.

| Commit marker | Release behavior |
| --- | --- |
| `#release` | publish a patch release |
| `#release #minor` | publish a minor release |
| `#release #major` | publish a major release |
| no `#release` | do not publish |

Only the configured release actor can trigger release preparation. Set the
GitHub Actions repository variable `PYPI_RELEASE_ACTOR` to the allowed
GitHub username. If the variable is not set, the workflow falls back to the
repository owner.

## Maintainer setup

1. Enable GitHub Pages for the repository and set the Pages build source to
   GitHub Actions.
2. The documentation workflow builds the site on every push to `main` and
   deploys it. Its `actions/configure-pages` step is set to enable Pages
   when it is not enabled yet.
3. Create or confirm the GitHub environment named `pypi`.
4. Set `PYPI_RELEASE_ACTOR` in GitHub Actions repository variables when the
   release actor is not the repository owner.
5. In PyPI, add a trusted publisher for the GitHub repository, workflow
   `.github/workflows/release.yml`, and environment `pypi`. Restrict the
   `pypi` environment's deployment branches and tags to the `v*` tag pattern.
6. Protect `main` by requiring the single CI check `ci gate`, which passes
   only when every other CI job does, and by disallowing force pushes and
   branch deletion. Leave administrators unenforced, because release
   preparation pushes its `chore(release)` commit with the `RELEASE_PAT`
   secret, a personal access token of a repository administrator.
7. Merge changes through pull requests. Use squash merges when you want the
   release markers to be unambiguous in the final commit message.
8. Include `#release` in the final commit message only when the change should
   publish to PyPI.
9. Let GitHub Actions build, test, create the release commit and tag, check out
   that tag for publishing, publish the package to PyPI, publish a GitHub
   release with notes generated from the merged pull requests, and redeploy
   the documentation site.
10. Set up Zenodo archiving once. Sign in at
    [zenodo.org](https://zenodo.org/) with GitHub and switch the repository
    on under *Account > GitHub*. Every GitHub release published afterwards
    is archived with a version DOI, using the metadata in `CITATION.cff`.
    The concept DOI, which always resolves to the latest version, is the
    `doi` field of `CITATION.cff`. The badges and citation text in
    `README.md` and `docs/index.md` repeat it, and `tests/test_citation.py`
    fails if they differ from it.

Publishing runs from the `vX.Y.Z` tag created by release preparation. If the
PyPI configuration is wrong on the first attempt, fix the trusted-publisher
settings and rerun the failed publish job instead of creating a new version.

The publish job is defined in the same workflow file as release preparation,
but it runs in a separate workflow run that the tag push starts. Release
preparation therefore pushes the tag with the `RELEASE_PAT` secret, because
GitHub does not start most workflow runs from events triggered with the
repository `GITHUB_TOKEN`.
