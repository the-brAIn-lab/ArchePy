# Releasing ArchePy

This document is for maintainers cutting a release. Contributors don't need
to read it.

## Versioning

ArchePy follows [Semantic Versioning](https://semver.org). For a `0.x.y`
package the rules are:

- **PATCH** (`0.1.0` → `0.1.1`) — bug fixes only, no API changes.
- **MINOR** (`0.1.0` → `0.2.0`) — new features and/or any breaking changes.
- **MAJOR** (`0.x.y` → `1.0.0`) — first stable release. Once we hit `1.0.0`,
  breaking changes will require a major bump.

## Pre-flight checklist

Before you start the release process, confirm:

- [ ] All planned changes are merged into `main`.
- [ ] CI is green on `main`.
- [ ] You can build and install the package locally:
      `python -m build && pip install dist/archepy-*.whl`
- [ ] All tests pass on Python 3.9–3.12.
- [ ] `CHANGELOG.md` has an `## [Unreleased]` section with all the changes.

## Release procedure

### 1. Bump the version

Edit two files in lockstep:

- `src/archepy/_version.py` — change `__version__`.
- `pyproject.toml` — change the `version` field in `[project]`.

Open a PR titled "Release vX.Y.Z" with just the version bump and the
changelog finalization (next step). Merge it once CI passes.

### 2. Finalize the changelog

In `CHANGELOG.md`:

1. Move the contents of `## [Unreleased]` to a new `## [X.Y.Z] — YYYY-MM-DD`
   section.
2. Reset `## [Unreleased]` to empty subsections (Added/Changed/etc).
3. Update the link references at the bottom of the file: add a new
   `[X.Y.Z]:` link and update `[Unreleased]:` to compare from the new tag.

The release notes should be **exhaustive** — list every API change, new
feature, and bug fix. Future you and your users will appreciate it.

### 3. Tag and push

```bash
git checkout main
git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Tag names start with `v` (e.g., `v0.2.0`).

### 4. Build distribution artifacts

In a clean checkout:

```bash
# Make sure the build environment is current.
pip install --upgrade build twine

# Wipe any stale build artifacts.
rm -rf dist/ build/ src/*.egg-info

# Build sdist and wheel.
python -m build

# Sanity check: the README will render correctly on PyPI.
twine check dist/*
```

You should see two files in `dist/`: `archepy-X.Y.Z.tar.gz` and
`archepy-X.Y.Z-py3-none-any.whl`.

### 5. Test on TestPyPI first

[TestPyPI](https://test.pypi.org/) is a separate sandbox that lets you verify
the release artifacts before pushing to the real index. **Once a version is on
PyPI, it cannot be replaced** — you can yank it, but the version number is
permanently burned.

If you don't already have a TestPyPI account, register at
<https://test.pypi.org/account/register/> and create an API token with scope
"Entire account" or specific to ArchePy.

```bash
twine upload --repository testpypi dist/*
```

When prompted, use `__token__` as the username and your TestPyPI API token
(starting with `pypi-`) as the password. To avoid retyping, you can add this
to `~/.pypirc`:

```ini
[testpypi]
username = __token__
password = pypi-AgENdGV...your-token-here
```

Verify the upload by installing from TestPyPI in a fresh virtual environment:

```bash
python -m venv /tmp/archepy-test
source /tmp/archepy-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            archepy
python -c "import archepy; print(archepy.__version__)"
```

The `--extra-index-url` flag is important: TestPyPI doesn't mirror NumPy,
so without it, `pip install archepy` will fail to resolve dependencies.

### 6. Upload to real PyPI

Once TestPyPI looks good:

```bash
twine upload dist/*
```

Same credentials story: `__token__` + a real PyPI API token.

Verify:

```bash
python -m venv /tmp/archepy-real
source /tmp/archepy-real/bin/activate
pip install archepy
python -c "import archepy; print(archepy.__version__)"
```

### 7. Draft a GitHub Release

1. Go to the [Releases page](https://github.com/the-brAIn-lab/archepy/releases)
   and click **Draft a new release**.
2. Choose the tag you pushed in step 3 (`vX.Y.Z`).
3. Title: `vX.Y.Z (Month YYYY)` (e.g., `v0.2.0 (May 2026)`).
4. Description: paste the relevant section from `CHANGELOG.md`. GitHub will
   auto-link issue/PR numbers.
5. Attach `dist/archepy-X.Y.Z.tar.gz` and the `.whl` file (drag and drop).
6. Click **Publish release**.

### 8. Announce (optional)

For minor and major releases, consider:

- Posting on the relevant mailing lists or social media.
- Updating the BibTeX entry in the README to reference the new DOI (if you
  mint one via Zenodo).
- Letting the upstream MATLAB toolbox authors know.

## If something goes wrong

### A bad release made it to PyPI

You **cannot** delete and re-upload the same version number. Options:

1. **Yank** the bad release (PyPI web UI → Manage release → Yank). Yanking
   keeps the version visible but tells `pip` not to install it for new users
   unless they pin it explicitly.
2. **Release a patch** (e.g., `0.2.0` → `0.2.1`) with the fix.

For minor mistakes, just release a patch. Yanking is for releases that are
seriously broken or have a security issue.

### TestPyPI doesn't reflect your upload

TestPyPI prunes packages periodically. If a previous test upload disappeared,
just re-upload — that's expected.

### The `twine upload` step hangs or fails

Check `~/.pypirc` permissions (`chmod 600 ~/.pypirc` on Unix). Confirm you're
using `__token__` as the literal username, not your account username.

## Future automation

Steps 4–7 can be automated with a GitHub Actions workflow that triggers on
tag push (`on: push: tags: 'v*'`) and uses a [trusted publisher](https://docs.pypi.org/trusted-publishers/)
relationship with PyPI to avoid storing tokens. This is on the roadmap; for
now, manual is fine while the package is small.
