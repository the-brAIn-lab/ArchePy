"""
Smoke tests for ArchePy.

These are fast checks designed to catch regressions, not exhaustive tests
of correctness. Run with ``pytest`` from the repository root.

Shape conventions (spatial MS-AA):
    X     : (T, V)        — T time points × V voxels
    sX    : (T, sV)       — usually equal to X (so sV == V)
    C     : (sV, K)       — sparse selector: each column picks ~one voxel
    sXC   : (T, K)        — the K archetype time-courses
    S     : (K, V)        — each column mixes the archetypes for one voxel
    Recon : sXC @ S       — shape (T, V)
"""

from __future__ import annotations

import numpy as np
import pytest

import archepy
from archepy import Subject, furthest_sum, multi_subject_aa


def test_version_is_string():
    """The package should expose a version string."""
    assert isinstance(archepy.__version__, str)
    assert len(archepy.__version__) > 0


def test_furthest_sum_returns_correct_count():
    """FurthestSum should return exactly `noc` indices."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 200))  # D x N data matrix
    idx = furthest_sum(X, noc=15, i=0, treat_as_kernel=False)
    assert len(idx) == 15
    assert len(set(idx)) == 15
    assert all(0 <= i < 200 for i in idx)


def test_furthest_sum_kernel_mode():
    """Kernel mode: identity kernel should produce a valid selection."""
    K = np.eye(100)
    idx = furthest_sum(K, noc=10, i=0, treat_as_kernel=True)
    assert len(idx) == 10
    assert len(set(idx)) == 10


def test_furthest_sum_deterministic_with_seed():
    """Re-running with the same seed should produce the same selection."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 500))
    idx_a = furthest_sum(X, noc=10, i=3, treat_as_kernel=False)
    idx_b = furthest_sum(X, noc=10, i=3, treat_as_kernel=False)
    assert idx_a == idx_b


def test_msaa_runs_on_planted_structure():
    """End-to-end: MS-AA should recover most variance from low-rank data."""
    rng = np.random.default_rng(0)
    n_subjects, T, V = 2, 30, 80
    K = 4

    # Plant structure consistent with the AA model: each voxel is a sparse
    # mixture of K underlying archetype time-courses.
    archetype_courses = rng.standard_normal((T, K))  # (T, K)

    subjects = []
    for _ in range(n_subjects):
        S_mix = rng.dirichlet(np.ones(K), size=V).T  # (K, V)
        X = archetype_courses @ S_mix + 0.05 * rng.standard_normal((T, V))
        subjects.append(Subject(X=X.astype(float), sX=X.astype(float)))

    results, C, cost, varexpl, elapsed = multi_subject_aa(
        subjects,
        noc=K,
        opts={
            "maxiter": 50,
            "conv_crit": 1e-6,
            "fix_var_iter": 2,
            "use_gpu": False,
            "heteroscedastic": False,
            "rngSEED": 0,
        },
    )

    # Shape sanity (see module docstring for conventions)
    assert C.shape == (V, K)
    assert len(results) == n_subjects
    assert results[0]["S"].shape == (K, V)
    assert results[0]["sXC"].shape == (T, K)

    # Cost should be finite and not increasing overall
    assert np.all(np.isfinite(cost))
    assert cost[-1] <= cost[0]

    # Variance explained should be high for planted structure
    assert 0.0 <= varexpl <= 1.0
    assert varexpl > 0.7, f"Expected high varexpl on planted data, got {varexpl:.3f}"

    assert elapsed > 0


def test_msaa_random_init():
    """Random init path should also produce a valid fit."""
    rng = np.random.default_rng(1)
    T, V = 20, 50
    subjects = [
        Subject(
            X=rng.standard_normal((T, V)),
            sX=rng.standard_normal((T, V)),
        )
        for _ in range(2)
    ]
    results, C, cost, varexpl, elapsed = multi_subject_aa(
        subjects,
        noc=3,
        opts={
            "maxiter": 10,
            "fix_var_iter": 1,
            "init": "random",
            "use_gpu": False,
            "heteroscedastic": False,
            "rngSEED": 1,
        },
    )
    assert C.shape == (V, 3)
    assert all(np.isfinite(cost))


@pytest.mark.parametrize("init", ["FurthestSum", "random"])
def test_msaa_both_init_strategies(init):
    """Both init strategies should run without error."""
    rng = np.random.default_rng(2)
    T, V = 15, 40
    subjects = [
        Subject(
            X=rng.standard_normal((T, V)),
            sX=rng.standard_normal((T, V)),
        )
        for _ in range(2)
    ]
    results, C, cost, varexpl, elapsed = multi_subject_aa(
        subjects,
        noc=3,
        opts={"maxiter": 5, "fix_var_iter": 1, "init": init, "rngSEED": 0},
    )
    assert C.shape == (V, 3)
