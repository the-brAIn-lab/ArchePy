"""
Sphinx configuration for ArchePy documentation.

Build locally with:
    cd docs && make html
Output goes to docs/_build/html/index.html.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable for autodoc.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

import archepy  # noqa: E402

project = "ArchePy"
author = "ArchePy contributors"
copyright = f"2026, {author}"
release = archepy.__version__
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",        # pull docstrings into docs
    "sphinx.ext.autosummary",    # auto-generate API summary tables
    "sphinx.ext.napoleon",       # parse NumPy-style docstrings
    "sphinx.ext.intersphinx",    # link to numpy/scipy/etc. docs
    "sphinx.ext.viewcode",       # "[source]" links next to API entries
    "sphinx.ext.mathjax",        # render math
    "myst_parser",               # Markdown support (CHANGELOG, README inclusion)
    "nbsphinx",                  # render Jupyter notebooks as docs pages
]

# Suppress noisy warnings:
# - myst.xref_missing fires on relative links like [LICENSE](LICENSE) inside
#   included .md files; those resolve fine on GitHub but Sphinx can't follow
#   them. They're not real broken links.
suppress_warnings = ["myst.xref_missing"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

# ---------------------------------------------------------------------------
# autodoc / autosummary
# ---------------------------------------------------------------------------

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # render type hints in the description, not signature
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# Don't fail the build if optional deps aren't installed (CuPy, nibabel, matplotlib).
autodoc_mock_imports = ["cupy", "nibabel", "matplotlib"]

# ---------------------------------------------------------------------------
# intersphinx — link to other projects' docs
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------------------------------
# nbsphinx
# ---------------------------------------------------------------------------

# Don't try to execute notebooks at build time (too slow, may need GPU/data).
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"  # clean, modern, dark-mode-friendly
html_static_path = ["_static"]
html_title = f"ArchePy {version}"

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/the-brAIn-lab/archepy",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/the-brAIn-lab/archepy",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}
