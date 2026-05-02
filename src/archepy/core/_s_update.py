"""
Inner S-update step (private).

Direct port of ``SupdateIndiStep.m`` from the upstream Multisubject Archetypal
Analysis Toolbox. Not part of the public API.

Original MATLAB authors:
    Jesper L. Hinrich, Sophia E. Bardenfleth, Morten Mørup
    Copyright (C) 2016 Technical University of Denmark.
    Distributed under the terms of the Multisubject Archetypal Analysis
    Toolbox license (see LICENSE).
"""

from __future__ import annotations

import numpy as np


def supdate_indi_step(
    S: np.ndarray,
    XCtX: np.ndarray,
    CtXtXC: np.ndarray,
    muS: np.ndarray | float,
    numObs: int,
    niter: int,
    sigmaSq: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = np.array(S, dtype=np.float64, order="C")
    XCtX = np.array(XCtX, dtype=np.float64, order="C")
    CtXtXC = np.array(CtXtXC, dtype=np.float64, order="C")

    K, F = S.shape

    if np.isscalar(muS):
        muS = np.ones(F, dtype=np.float64)
    else:
        muS = np.asarray(muS, dtype=np.float64).reshape(-1)
        if muS.size != F:
            raise ValueError(f"muS must have length F={F} (got {muS.size}).")

    scale = 1.0 / (numObs * F)

    CtS = CtXtXC @ S
    cost = np.einsum("kf,kf->f", S * (-2.0), XCtX) + np.einsum("kf,kf->f", S, CtS)

    k = 1
    rel_delta_cost = np.inf
    denom_eps = 1e-30

    while k <= niter and rel_delta_cost > 1e-12:
        g = (CtS - XCtX) * scale
        col_dots = np.einsum("kf,kf->f", g, S)
        g -= col_dots

        Sold = S.copy()
        S = Sold - g * muS

        np.maximum(S, 0.0, out=S)
        col_sums = S.sum(axis=0)
        col_sums[col_sums == 0.0] = 1.0
        S /= col_sums

        CtS = CtXtXC @ S
        cost_new = np.einsum("kf,kf->f", S * (-2.0), XCtX) + np.einsum("kf,kf->f", S, CtS)

        accept = cost_new <= cost

        diff = cost_new - cost
        rel_delta_cost = float(np.dot(diff, diff) / max(np.dot(cost, cost), denom_eps))

        if not accept.all():
            reject = ~accept
            S[:, reject] = Sold[:, reject]
            CtS[:, reject] = CtXtXC @ Sold[:, reject]

        muS[accept] *= 1.2
        muS[~accept] *= 0.5
        cost[accept] = cost_new[accept]

        k += 1

    if sigmaSq is not None:
        sigmaSq = np.asarray(sigmaSq, dtype=np.float64).reshape(-1)
        if sigmaSq.size != F:
            raise ValueError(f"sigmaSq must have length F={F} (got {sigmaSq.size}).")
        SSt = S @ (S / sigmaSq).T
    else:
        SSt = S @ S.T

    return S, muS, SSt
