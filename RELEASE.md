# Release Process

This document describes how to release a new version of pycortex.

## Prerequisites

- Push access to the main repository
- PyPI publishing is handled automatically via GitHub Actions

## Steps

### 1. Update the version number

Edit `cortex/version.py` and change the version from development to release:

```python
# Change from:
__version__ = '1.3.0.dev0'

# To:
__version__ = '1.2.12'
```

### 2. Commit the version change

```bash
git add cortex/version.py
git commit -m "MNT version 1.2.12"
```

### 3. Create and push the tag

Use an annotated tag:

```bash
git tag -a 1.2.12 -m "Version 1.2.12"
git push origin main
git push origin 1.2.12
```

This triggers GitHub Actions to:
- Build and publish the source distribution to PyPI
- Build and deploy documentation to GitHub Pages

### 4. Create GitHub Release (optional)

Go to https://github.com/gallantlab/pycortex/releases and create a new release from the tag. Use "Generate release notes" to auto-populate from merged PRs.

### 5. Bump back to development version

```bash
# Edit cortex/version.py back to dev version
# e.g., __version__ = '1.3.0.dev0'

git add cortex/version.py
git commit -m "MNT back to dev [skip ci]"
git push origin main
```

## Versioning

- Release versions: `X.Y.Z` (e.g., `1.2.12`)
- Development versions: `X.Y.Z.dev0` (e.g., `1.3.0.dev0`)

The version in `cortex/version.py` is the single source of truth, used by `setup.py` and documentation.
