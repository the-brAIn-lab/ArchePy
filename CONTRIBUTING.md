# Contributing to ArchePy

Thank you for considering contributing to ArchePy! This document describes how
to set up a development environment, run the test suite, and submit changes.

> [!IMPORTANT]
> ArchePy is distributed under the **Multisubject Archetypal Analysis Toolbox
> license** (academic / non-profit use only — see [LICENSE](LICENSE)). By
> contributing, you agree that your contributions will be distributed under
> the same terms. If your contribution would substantially extend the package
> beyond its current scope (e.g., a new analysis variant), please open an
> issue first to discuss licensing implications.

## Code of Conduct

Everyone interacting in the ArchePy project's codebase, issue tracker, and pull
requests is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Quick links

- [Setting up a development environment](#setting-up-a-development-environment)
- [Running the test suite](#running-the-test-suite)
- [Code style](#code-style)
- [Writing docstrings](#writing-docstrings)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs](#reporting-bugs)
- [Proposing features](#proposing-features)

## Setting up a development environment

ArchePy uses the modern PEP 621 `pyproject.toml` layout with `hatchling` as the
build backend. To get a working development environment:

```bash
# Clone your fork
git clone https://github.com/the-brAIn-lab/archepy
cd archepy

# Create a virtual environment (highly recommended)
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install in editable mode with all development dependencies
pip install -e ".[dev]"

# Optional: install fMRI helpers and/or GPU support
pip install -e ".[dev,fmri]"       # adds nibabel
pip install -e ".[dev,fmri,gpu]"   # adds CuPy (requires CUDA 12.x)
```

Editable installs (`-e`) mean your changes to the source are picked up without
reinstalling. You only need to re-run `pip install -e .` if you change
`pyproject.toml` itself.

## Running the test suite

```bash
# Run all tests
pytest

# Run a specific test file or test
pytest tests/test_smoke.py
pytest tests/test_smoke.py::test_furthest_sum_kernel_mode

# Run with coverage report
pytest --cov=archepy --cov-report=term-missing
```

Tests should pass before you open a pull request. CI will also run on Linux,
macOS, and Windows across Python 3.9–3.12.

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
Before committing:

```bash
ruff format src tests           # auto-format
ruff check --fix src tests      # lint and auto-fix what's safe
```

CI runs `ruff check` and `ruff format --check`, so any style issues will be
caught at PR time.

A few project-specific conventions:

- **Module names** are `snake_case` (e.g., `furthest_sum.py`, not `FurthestSum.py`).
- **Class names** are `CamelCase` (e.g., `Subject`).
- **Internal modules** start with an underscore (e.g., `_s_update.py`,
  `_utils.py`). They are not part of the public API and may change without
  deprecation notice.
- **Public API** is exposed from `src/archepy/__init__.py`. If you add a new
  user-facing function or class, export it there.

## Writing docstrings

We use [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html)
docstrings throughout. This format is parsed by Sphinx for the documentation
website. A minimal example:

```python
def add(a, b):
    """
    Sum two numbers.

    Parameters
    ----------
    a : int or float
        First addend.
    b : int or float
        Second addend.

    Returns
    -------
    int or float
        The sum ``a + b``.

    Examples
    --------
    >>> add(2, 3)
    5
    """
    return a + b
```

Every public function should have a docstring. For algorithm code that is a
direct port of MATLAB originals, it is appropriate to credit the original
authors at the top of the docstring or module.

## Submitting a pull request

1. **Open an issue first** for any non-trivial change. This avoids you investing
   time in something we can't accept (or that someone else is already working on).
2. **Fork the repository** and create a branch off `main`. Use a descriptive
   branch name like `fix/sigma-init-bug` or `feat/sparse-init`.
3. **Make your changes**, including tests for any new functionality.
4. **Run the test suite** locally (`pytest`) and confirm it passes.
5. **Format and lint** with `ruff format` + `ruff check --fix`.
6. **Update the changelog** by adding an entry to `CHANGELOG.md` under
   `## [Unreleased]`.
7. **Open a PR** with a clear description of what changed and why. Reference
   the issue it addresses.

Small, focused PRs get reviewed faster than large ones. If you have a bigger
change in mind, consider breaking it into multiple PRs.

## Reporting bugs

Open an issue using the **Bug report** template. Include:

- A short, top-level summary of the bug (1–2 sentences).
- A minimal, self-contained code snippet that reproduces the issue.
- The expected vs. actual behavior.
- Your platform: OS, Python version, NumPy version, and (if relevant) CuPy /
  CUDA version.

## Proposing features

Open an issue using the **Feature request** template. Describe:

- The problem you're trying to solve (not just the proposed solution).
- Why the current API is insufficient.
- A rough sketch of how the feature would work, if you have one.

ArchePy is a focused package — we aim to be a faithful, well-tested Python port
of the MS-AA toolbox plus minimal fMRI conveniences. Features that go beyond
this scope (e.g., generic AA, single-subject methods) are likely better placed
in a separate package; we may suggest you contribute to
[`archetypes`](https://github.com/aleixalcacer/archetypes) or
[`py_pcha`](https://github.com/ulfaslak/py_pcha) instead.

## Releasing

If you're a maintainer cutting a release, see [RELEASING.md](RELEASING.md).

## Acknowledging contributions

All contributors will be acknowledged in the release notes. For substantial
algorithmic contributions, we may also list you as a co-author on the package
metadata. Please let us know your preferred name and (optional) ORCID.

## Questions?

Open a [Discussion](https://github.com/paxe/archepy/discussions) or
contact the maintainer directly.
