"""
Temporal Multi-Subject Archetypal Analysis (MS-AA).

Direct Python port of ``MultiSubjectAA_T.m`` from the upstream Multisubject
Archetypal Analysis Toolbox.

Original MATLAB authors:
    Jesper L. Hinrich, Sophia E. Bardenfleth, Morten Mørup
    Copyright (C) 2016 Technical University of Denmark.
    Distributed under the terms of the Multisubject Archetypal Analysis
    Toolbox license (see LICENSE).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from archepy._utils import mgetopt, to_numpy
from archepy.core._s_update import supdate_indi_step
from archepy.init.furthest_sum import furthest_sum


@dataclass
class SubjectT:
    """
    Temporal MS-AA subject container.

    Parameters
    ----------
    X : ndarray, shape (V, T)
        Voxels × time points.
    sX : ndarray, shape (V, sT)
        Voxels × "tilde" time points.
    """

    X: np.ndarray
    sX: np.ndarray


def multi_subject_aa_T(
    subj: List[SubjectT],
    noc: int,
    opts: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, float, float]:
    """
    Fit Multi-Subject Archetypal Analysis (temporal variant).

    See :func:`archepy.multi_subject_aa` for the spatial variant and a
    description of the ``opts`` dictionary; the same options apply here.
    """
    if opts is None:
        opts = {}

    conv_crit = mgetopt(opts, "conv_crit", 1e-6)
    maxiter = int(mgetopt(opts, "maxiter", 100))
    fix_var_iter = int(mgetopt(opts, "fix_var_iter", 5))
    runGPU = bool(mgetopt(opts, "use_gpu", False))
    voxelVariance = bool(mgetopt(opts, "heteroscedastic", True))
    numCstep = int(mgetopt(opts, "numCstep", 10))
    numSstep = int(mgetopt(opts, "numSstep", 20))
    sort_crit = mgetopt(opts, "sort_crit", "corr")
    init_type = mgetopt(opts, "init", "FurthestSum")
    initial_S_steps = int(mgetopt(opts, "initSstep", 250))
    rngSEED = mgetopt(opts, "rngSEED", None)

    if runGPU and cp is None:
        raise ImportError(
            "opts['use_gpu']=True requested but CuPy is not installed. "
            "Install with: pip install archepy[gpu]"
        )

    rng = np.random.default_rng(rngSEED)

    V = subj[0].X.shape[0]
    T = subj[0].X.shape[1]
    sT = subj[0].sX.shape[1]
    B = len(subj)
    V_list = [s.sX.shape[0] for s in subj]

    xp = cp if runGPU else np

    # ---- Initialize C (sT x noc) ----
    if init_type.lower() == "furthestsum":
        total_V = sum(V_list)
        Xcombined = np.zeros((total_V, sT), dtype=float)
        off = 0
        for s in subj:
            vv = s.sX.shape[0]
            Xcombined[off : off + vv, :] = np.asarray(s.sX, dtype=float)
            off += vv

        seed = rng.integers(low=0, high=sT)
        idx = furthest_sum(Xcombined, noc=noc, i=int(seed), exclude=None, treat_as_kernel=False)
        C = xp.zeros((sT, noc), dtype=float)
        C[idx, xp.arange(noc)] = 1.0
        muC = xp.array(1.0)
    else:
        C = xp.asarray(rng.random((sT, noc)), dtype=float)
        C /= C.sum(axis=0, keepdims=True) + xp.finfo(float).eps
        muC = xp.array(1.0)

    # ---- Move to GPU & SST ----
    for s in subj:
        if runGPU:
            s.X = cp.asarray(s.X, dtype=float)
            s.sX = cp.asarray(s.sX, dtype=float)
        s.V = s.sX.shape[0]
        s.SST = (s.X * s.X).sum()

    SST = float(sum([float(s.SST.get()) if runGPU else float(s.SST) for s in subj]))

    # ---- Initialize per-subject ----
    for s in subj:
        if voxelVariance:
            s.sigmaSq = xp.ones((s.V, 1), dtype=float) * (SST / (sum(V_list) * T))
        else:
            s.sigmaSq = xp.ones((s.V, 1), dtype=float)

        U = xp.asarray(rng.random((noc, T)), dtype=float)
        s.S = -xp.log(U + xp.finfo(float).tiny)
        s.S /= s.S.sum(axis=0, keepdims=True) + xp.finfo(float).eps
        s.muS = xp.ones((1,), dtype=float)

        s.sXC = s.sX @ C
        s.XCtX = s.sXC.T @ s.X
        s.CtXtXC = s.sXC.T @ s.sXC
        s.SSt = s.S @ s.S.T
        s.XSt = s.X @ s.S.T

        S_np, muS_np, SSt_np = supdate_indi_step(
            to_numpy(s.S, runGPU),
            to_numpy(s.XCtX, runGPU),
            to_numpy(s.CtXtXC, runGPU),
            np.ones(s.S.shape[1], dtype=float),
            int(s.V),
            int(initial_S_steps),
        )
        s.S = xp.asarray(S_np)
        s.muS = xp.ones((s.S.shape[1],), dtype=float)
        s.SSt = xp.asarray(SSt_np)

    # ---- Initial NLL ----
    NLL = 0.0
    SST_sigmaSq = 0.0
    for s in subj:
        inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
        s.XSt = (s.X @ s.S.T) * inv_sqrt_sig
        s.sXC = (s.sX @ C) * inv_sqrt_sig
        s.CtXtXC = s.sXC.T @ s.sXC
        s.XCtX = ((s.sXC * inv_sqrt_sig).T) @ s.X

        s.SST_sigmaSq = (s.X * (s.X / s.sigmaSq)).sum()
        s.NLL = (
            0.5 * s.SST_sigmaSq
            - (s.sXC * s.XSt).sum()
            + 0.5 * (s.CtXtXC * s.SSt).sum()
            + (T / 2.0) * (s.V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())
        )
        NLL += float(s.NLL.get()) if runGPU else float(s.NLL)
        SST_sigmaSq += float(s.SST_sigmaSq.get()) if runGPU else float(s.SST_sigmaSq)

    t_start = time.perf_counter()
    cost_fun = np.zeros((maxiter,), dtype=float)

    noise_threshold_opt = mgetopt(opts, "noise_threshold", None)
    if noise_threshold_opt is None:
        var_threshold = (SST / (sum(V_list) * T)) * 1e-3
    else:
        var_threshold = float(noise_threshold_opt)

    # ---- Main loop ----
    iter_ = 0
    dNLL = np.inf
    while ((abs(dNLL) >= conv_crit * abs(NLL)) or (fix_var_iter >= iter_)) and (iter_ < maxiter):
        iter_ += 1
        NLL_old = NLL
        cost_fun[iter_ - 1] = NLL

        C, muC, NLL = _Cupdate_multi_subjects_T(subj, C, muC, NLL, numCstep, runGPU)

        for s in subj:
            inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
            s.sXC = (s.sX @ C) * inv_sqrt_sig
            s.XCtX = ((s.sXC * inv_sqrt_sig).T) @ s.X
            s.CtXtXC = s.sXC.T @ s.sXC
            s.NLL = (
                0.5 * s.SST_sigmaSq
                - (s.XCtX * s.S).sum()
                + 0.5 * (s.CtXtXC * s.SSt).sum()
                + (T / 2.0) * (s.V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())
            )

        NLL = 0.0
        SST_sigmaSq = 0.0
        for s in subj:
            S_np, muS_np, SSt_np = supdate_indi_step(
                to_numpy(s.S, runGPU),
                to_numpy(s.XCtX, runGPU),
                to_numpy(s.CtXtXC, runGPU),
                to_numpy(s.muS, runGPU),
                int(s.V),
                int(numSstep),
            )
            s.S = xp.asarray(S_np)
            s.muS = xp.asarray(muS_np)
            s.SSt = xp.asarray(SSt_np)

            if voxelVariance and (iter_ > fix_var_iter):
                resid = s.X - (s.sX @ C) @ s.S
                s.sigmaSq = (resid * resid).sum(axis=1, keepdims=True) / float(T)
                s.sigmaSq = xp.maximum(s.sigmaSq, var_threshold)

                inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
                s.XSt = (s.X @ s.S.T) * inv_sqrt_sig
                s.sXC = (s.sX @ C) * inv_sqrt_sig
                s.XCtX = ((s.sXC * inv_sqrt_sig).T) @ s.X
                s.CtXtXC = s.sXC.T @ s.sXC

                s.SST_sigmaSq = (s.X * (s.X / s.sigmaSq)).sum()
            else:
                inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
                s.XSt = (s.X @ s.S.T) * inv_sqrt_sig

            s.NLL = (
                0.5 * s.SST_sigmaSq
                - (s.sXC * s.XSt).sum()
                + 0.5 * (s.CtXtXC * s.SSt).sum()
                + (T / 2.0) * (s.V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())
            )

            NLL += float(s.NLL.get()) if runGPU else float(s.NLL)
            SST_sigmaSq += float(s.SST_sigmaSq.get()) if runGPU else float(s.SST_sigmaSq)

        dNLL = NLL_old - NLL

    elapsed = time.perf_counter() - t_start

    SSE = []
    for s in subj:
        recon = (s.sX @ C) @ s.S
        if runGPU:
            sse = float(cp.linalg.norm(s.X - recon, ord="fro") ** 2)
        else:
            sse = float(np.linalg.norm(s.X - recon, ord="fro") ** 2)
        SSE.append(sse)
    varexpl = (SST - sum(SSE)) / SST

    ind = np.arange(noc)
    if sort_crit.lower() == "corr" and (sum(V_list) == max(V_list) * B):
        arch = np.zeros((V_list[0], B))
        mean_corr = np.zeros((noc,))
        for j in range(noc):
            for bi, s in enumerate(subj):
                arch[:, bi] = to_numpy(s.sXC[:, j], runGPU)
            Ccorr = np.corrcoef(arch, rowvar=False)
            iu = np.triu_indices(B, k=1)
            mean_corr[j] = float(Ccorr[iu].mean())
        ind = np.argsort(mean_corr)[::-1]

    if not np.array_equal(ind, np.arange(noc)):
        C = C[:, ind]
        for s in subj:
            s.S = s.S[ind, :]
            s.sXC = s.sXC[:, ind]

    C_np = to_numpy(C, runGPU)
    results_subj: List[Dict[str, Any]] = []
    for bi, s in enumerate(subj):
        out = {
            "S": to_numpy(s.S, runGPU),
            "sXC": to_numpy(s.sXC, runGPU),
            "sigmaSq": to_numpy(s.sigmaSq, runGPU),
            "NLL": float(to_numpy(s.NLL, runGPU)),
            "SSE": float(SSE[bi]),
            "SST": float(to_numpy(s.SST, runGPU)),
            "SST_sigmaSq": float(to_numpy(s.SST_sigmaSq, runGPU)),
        }
        results_subj.append(out)

    return results_subj, C_np, cost_fun[:iter_], float(varexpl), float(elapsed)


# ------------------------- helpers -------------------------


def _Cupdate_multi_subjects_T(
    subj: List[Any],
    C,
    muC,
    NLL: float,
    niter: int,
    runGPU: bool,
):
    xp = cp if runGPU else np
    sT, noc = C.shape
    T = subj[0].X.shape[1]

    XtXSt_list = []
    for s in subj:
        inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
        XtXSt_list.append(s.sX.T @ (s.XSt * inv_sqrt_sig))

    total_V = sum([int(s.V) for s in subj])

    for _ in range(niter):
        NLL_old = NLL

        g = xp.zeros((sT, noc), dtype=float)
        for s, XtXSt in zip(subj, XtXSt_list):
            inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
            g += s.sX.T @ ((s.sXC * inv_sqrt_sig) @ s.SSt) - XtXSt
        g /= total_V * sT

        col_dots = (g * C).sum(axis=0, keepdims=True)
        g = g - col_dots

        stop = False
        Cold = C.copy()
        while not stop:
            C = Cold - muC * g
            C = xp.maximum(C, 0.0)
            C /= C.sum(axis=0, keepdims=True) + xp.finfo(float).eps

            NLL_tmp = 0.0
            for s in subj:
                inv_sqrt_sig = 1.0 / xp.sqrt(s.sigmaSq)
                s.sXC = (s.sX @ C) * inv_sqrt_sig
                s.CtXtXC = s.sXC.T @ s.sXC
                term = (
                    0.5 * s.SST_sigmaSq
                    - (s.sXC * s.XSt).sum()
                    + 0.5 * (s.CtXtXC * s.SSt).sum()
                    + (T / 2.0)
                    * (s.V * np.log(2.0 * np.pi) + (cp if runGPU else np).log(s.sigmaSq).sum())
                )
                NLL_tmp += float(term.get()) if runGPU else float(term)

            if NLL_tmp <= NLL_old * (1.0 + 1e-9):
                muC = muC * 1.2
                NLL = NLL_tmp
                stop = True
            else:
                muC = muC / 2.0

    return C, muC, NLL
