# Release Process

This document describes how to release a new version of pycortex.

## Prerequisites

- Push access to the main repository
- Maintainer access to create GitHub releases
- PyPI/TestPyPI environments must be configured with trusted publishing (done by repository admins)

## Simplified Release Process

Version numbers are now automatically derived from git tags using `setuptools-scm`. You no longer need to manually edit version files.

### Steps

#### 1. Create a GitHub Release

1. Go to https://github.com/gallantlab/pycortex/releases
2. Click "Draft a new release"
3. Click "Choose a tag" and type a new tag name following semantic versioning (e.g., `v1.3.0`)
4. Click "Create new tag: vX.Y.Z on publish"
5. Set the release title (e.g., "Version 1.3.0")
6. Use "Generate release notes" to auto-populate from merged PRs, or write custom notes
7. Click "Publish release"

That's it! The GitHub Actions workflow will automatically:
- Detect the new release event
- Checkout the code with full git history
- Use `setuptools-scm` to derive the version from the git tag
- Build the source distribution and wheel
- Publish to PyPI (from the `pypi` environment)
- Publish to TestPyPI (from the `testpypi` environment)

#### 2. Verify the Release

- Check that the GitHub Actions workflow completed successfully
- Verify the package appears on PyPI: https://pypi.org/project/pycortex/
- Verify the package appears on TestPyPI: https://test.pypi.org/project/pycortex/
- Test installing the new version: `pip install --upgrade pycortex`

## Versioning

- Version numbers are automatically derived from git tags by `setuptools-scm`
- Release versions: `vX.Y.Z` (e.g., `v1.3.0`) → published as `X.Y.Z` on PyPI
- Development versions between releases: `X.Y.Z.devN` (automatically generated)
- Use semantic versioning: MAJOR.MINOR.PATCH

## Manual Testing (TestPyPI only)

To test the release process without publishing to PyPI:

1. Go to https://github.com/gallantlab/pycortex/actions/workflows/publish.yml
2. Click "Run workflow"
3. Select the branch to test
4. Click "Run workflow"

This will build the package and publish only to TestPyPI, not to PyPI.

## Troubleshooting

### Build fails due to shallow clone

The workflow uses `fetch-depth: 0` to ensure full git history is available for `setuptools-scm`. If builds fail with version detection errors, check that this setting is present in the workflow.

### Version number is incorrect

`setuptools-scm` derives versions from git tags. Ensure:
- Tags follow the format `vX.Y.Z` or `X.Y.Z`
- Tags are annotated tags (created with `git tag -a`)
- The repository has at least one tag

### Publishing fails

Check that:
- The PyPI/TestPyPI environments are configured in the repository settings
- Trusted publishing is configured for this repository on PyPI/TestPyPI
- The workflow has the necessary permissions (`id-token: write`)

