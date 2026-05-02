"""
FurthestSum initialization (CPU).

Picks an initial set of archetype indices by greedily selecting the column
that is farthest (in summed Euclidean distance) from the points already
chosen. Faithful port of ``FurthestSum.m`` from the upstream MATLAB toolbox.

For GPU acceleration on large problems, see :func:`archepy.init.furthest_sum_gpu`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def furthest_sum(
    K: np.ndarray,
    noc: int,
    i: int | Iterable[int],
    exclude: Iterable[int] | None = None,
    *,
    treat_as_kernel: bool | None = None,
    one_based: bool = False,
) -> list[int]:
    """
    Greedy FurthestSum selection of ``noc`` candidate archetype indices.

    Parameters
    ----------
    K : ndarray
        Either a data matrix of shape ``(D, N)`` (observations as columns)
        or a symmetric kernel matrix of shape ``(N, N)``.
    noc : int
        Number of indices to select.
    i : int or iterable of int
        Initial seed index/indices.
    exclude : iterable of int, optional
        Indices that must not be selected.
    treat_as_kernel : bool, optional
        If ``None`` (default), auto-detect: a square symmetric ``K`` is
        treated as a kernel; otherwise as a data matrix. Pass ``False``
        explicitly for large square data matrices to skip the symmetry check.
    one_based : bool, default False
        If ``True``, treat input/output indices as 1-based (MATLAB style).

    Returns
    -------
    list[int]
        Selected indices.
    """
    A = np.asarray(K)
    if A.ndim != 2:
        raise ValueError("K must be 2D.")

    if treat_as_kernel is None:
        treat_as_kernel = (A.shape[0] == A.shape[1]) and np.allclose(A, A.T, atol=1e-10)

    if treat_as_kernel:
        N = A.shape[0]
    else:
        # Data matrix: D x N — observations are columns.
        N = A.shape[1]

    def to0(idx: Iterable[int] | int) -> list[int]:
        if isinstance(idx, (int, np.integer)):
            return [int(idx) - 1 if one_based else int(idx)]
        return [int(x) - 1 if one_based else int(x) for x in idx]

    selected: list[int] = to0(i)
    if len(selected) == 0:
        raise ValueError("Initial seed `i` must contain at least one index.")
    rolling_idx = selected[0]

    banned = np.zeros(N, dtype=bool)
    if exclude is not None:
        banned[to0(exclude)] = True
    banned[np.clip(selected, 0, N - 1)] = True

    sum_dist = np.zeros(N, dtype=float)

    if treat_as_kernel:
        Kdiag = np.diag(A).astype(float)

        def add_from(seed: int):
            d2 = Kdiag - 2.0 * A[seed, :] + Kdiag[seed]
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            return d2

        def remove_from(seed: int):
            return add_from(seed)
    else:
        X = A
        norms2 = np.sum(X * X, axis=0)

        def add_from(seed: int):
            Kq = np.dot(X[:, seed].T, X)
            d2 = norms2 - 2.0 * Kq + norms2[seed]
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            return d2

        def remove_from(seed: int):
            return add_from(seed)

    for k in range(1, noc + 10 + 1):
        if k > (noc - 1) and len(selected) > 0:
            to_remove = selected[0]
            sum_dist -= remove_from(to_remove)
            banned[to_remove] = False
            selected.pop(0)

        sum_dist += add_from(rolling_idx)

        candidates = np.where(~banned)[0]
        if candidates.size == 0:
            break
        farthest_idx = candidates[np.argmax(sum_dist[candidates])]

        rolling_idx = farthest_idx
        selected.append(farthest_idx)
        banned[farthest_idx] = True

    if len(selected) > noc:
        selected = selected[-noc:]
    elif len(selected) < noc:
        fill_needed = noc - len(selected)
        more = np.setdiff1d(np.arange(N), np.array(selected), assume_unique=False)
        selected.extend(list(more[:fill_needed]))

    if one_based:
        return [s + 1 for s in selected]
    return selected
