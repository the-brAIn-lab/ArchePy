"""
Internal utilities used across ArchePy modules.

This module is private — the leading underscore signals it is not part
of the public API. Anything imported from here may be renamed or removed
without a deprecation cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mgetopt(opts: Any, varname: str, default: Any, *args: Any) -> Any:
    """
    Get an option value from a dict-or-attribute container, with a default.

    This is a thin replacement for the MATLAB `mgetopt` shim used by the
    upstream toolbox. It accepts either a plain ``dict`` (looked up by key)
    or any object exposing the option as an attribute (e.g. a ``dataclass``
    or ``SimpleNamespace``).

    Parameters
    ----------
    opts : dict | object | None
        Option container. If ``None``, ``default`` is returned.
    varname : str
        Option name.
    default : Any
        Value to return if the option is absent.
    *args :
        Ignored. Kept for signature compatibility with the MATLAB original.

    Returns
    -------
    Any
        The option value, or ``default`` if not present.
    """
    if opts is None:
        return default

    if isinstance(opts, dict):
        return opts.get(varname, default)

    if hasattr(opts, varname):
        return getattr(opts, varname)

    try:
        return opts[varname]
    except Exception:
        return default


def to_numpy(a, run_gpu: bool) -> np.ndarray:
    """
    Bring an array (possibly on the GPU) back to NumPy.

    Used at module boundaries where downstream code expects host arrays
    (e.g., the inner S-update step, or returning results to the user).
    """
    if run_gpu and hasattr(a, "get"):
        try:
            return a.get()
        except Exception:
            return np.array(a)
    return np.asarray(a)


def to_float(a, run_gpu: bool) -> float:
    """Materialize a scalar (possibly on the GPU) as a Python ``float``."""
    if run_gpu and hasattr(a, "get"):
        return float(a.get())
    return float(a)
