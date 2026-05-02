"""
GPU FurthestSum implementation (CuPy).

Imported lazily via :func:`archepy.init.furthest_sum_gpu` so that users
without CuPy installed can still ``import archepy``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def furthest_sum_gpu(
    K,
    noc: int,
    i: int | Iterable[int],
    exclude: Optional[Iterable[int]] = None,
    *,
    treat_as_kernel: Optional[bool] = None,
    one_based: bool = False,
    device: int | None = None,
) -> List[int]:
    """
    GPU FurthestSum. See :func:`archepy.init.furthest_sum` for parameters.

    Requires CuPy (``pip install archepy[gpu]``).
    """
    if cp is None:
        raise ImportError(
            "CuPy is required for furthest_sum_gpu. "
            "Install with: pip install archepy[gpu]"
        )

    if device is not None:
        cp.cuda.Device(device).use()

    A = cp.asarray(K)

    if A.ndim != 2:
        raise ValueError("K must be a 2-D array.")

    if treat_as_kernel is None:
        if A.shape[0] == A.shape[1]:
            treat_as_kernel = bool(cp.allclose(A, A.T, atol=1e-10))
        else:
            treat_as_kernel = False

    if treat_as_kernel:
        N = A.shape[0]
        Kdiag = cp.diag(A).astype(A.dtype)

        def dist_from(seed: int) -> "cp.ndarray":
            d2 = Kdiag - 2.0 * A[seed, :] + Kdiag[seed]
            cp.maximum(d2, 0.0, out=d2)
            return cp.sqrt(d2)
    else:
        # Data mode: observations are columns (D, N), matching the CPU version.
        X = A
        N = int(X.shape[1])
        norms2 = cp.sum(X * X, axis=0)

        def dist_from(seed: int) -> "cp.ndarray":
            Kq = X[:, seed] @ X
            d2 = norms2 - 2.0 * Kq + norms2[seed]
            cp.maximum(d2, 0.0, out=d2)
            return cp.sqrt(d2)

    def to0(idx) -> List[int]:
        if isinstance(idx, (int, np.integer)):
            return [int(idx) - 1 if one_based else int(idx)]
        return [int(x) - 1 if one_based else int(x) for x in idx]

    selected: List[int] = to0(i)
    if not selected:
        raise ValueError("Initial seed `i` must contain at least one index.")
    rolling_idx = selected[0]

    available = cp.ones(N, dtype=cp.bool_)
    if exclude is not None:
        available[cp.asarray(to0(exclude))] = False
    if selected:
        available[cp.asarray(selected)] = False

    sum_dist = cp.zeros(N, dtype=cp.float64)

    total_iters = noc + 10
    report_every = max(1, noc // 10)

    for k in range(1, total_iters + 1):
        if k <= noc and k % report_every == 0:
            print(f"  [furthest_sum_gpu] {k}/{noc} archetypes initialised...")

        if k > (noc - 1) and selected:
            to_remove = selected[0]
            sum_dist -= dist_from(to_remove)
            available[to_remove] = True
            selected.pop(0)

        sum_dist += dist_from(rolling_idx)

        cand_idx = cp.nonzero(available)[0]
        if cand_idx.size == 0:
            break

        local_arg = int(cp.argmax(sum_dist[cand_idx]).get())
        farthest = int(cand_idx[local_arg].get())

        rolling_idx = farthest
        selected.append(farthest)
        available[farthest] = False

    print(f"  [furthest_sum_gpu] done — {min(len(selected), noc)} archetypes selected.")

    if len(selected) > noc:
        selected = selected[-noc:]
    elif len(selected) < noc:
        mask = cp.ones(N, dtype=cp.bool_)
        mask[cp.asarray(selected)] = False
        rest = cp.nonzero(mask)[0].get().tolist()
        selected.extend(rest[: noc - len(selected)])

    if one_based:
        return [s + 1 for s in selected]
    return selected
