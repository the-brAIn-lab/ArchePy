"""
fMRI noise helpers.

Two utilities are provided:

- :func:`estimate_background_noise` — estimate a noise variance threshold
  from a NIfTI file by computing per-voxel temporal variance over background
  voxels.
- :func:`generate_synthetic_noise` — generate radial variance maps for
  simulating spatially-varying scanner noise.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def estimate_background_noise(file: str, file_filt: Optional[str]) -> float:
    """
    Estimate a noise variance threshold from background voxels of a NIfTI.

    Parameters
    ----------
    file : str
        Path to raw fMRI NIfTI (used to compute the background mask).
    file_filt : str or None
        Path to filtered/preprocessed NIfTI (used to compute noise variance).
        If ``None``, the raw file is used for both steps.

    Returns
    -------
    float
        Mean of per-voxel temporal variances over background voxels.

    Notes
    -----
    Requires ``nibabel``. Install with ``pip install archepy[fmri]``.
    """
    if nib is None:
        raise ImportError(
            "nibabel is required for estimate_background_noise. "
            "Install with: pip install archepy[fmri]"
        )

    img_raw = nib.load(file)
    data_raw = img_raw.get_fdata(dtype=np.float32)
    if data_raw.ndim != 4:
        raise ValueError(f"`file` must be 4D (got shape {data_raw.shape}).")

    x, y, z, T = data_raw.shape
    V = x * y * z
    X = data_raw.reshape(V, T)

    voxel_means = X.mean(axis=1)
    global_mean = X.mean()
    not_brain = voxel_means < (0.8 * global_mean)

    if file_filt is None:
        data_filt = data_raw
    else:
        img_filt = nib.load(file_filt)
        data_filt = img_filt.get_fdata(dtype=np.float32)
        if data_filt.shape != (x, y, z, T):
            raise ValueError(
                f"`file` and `file_filt` must have same shape. "
                f"Got raw {data_raw.shape} vs filt {data_filt.shape}."
            )

    F = data_filt.reshape(V, T)

    if not np.any(not_brain):
        raise ValueError("Background mask is empty; check your data or threshold.")

    var_X = np.var(F[not_brain, :], axis=1, ddof=1)
    positive = var_X > 0
    if not np.any(positive):
        raise ValueError("All background variances are zero or NaN; check inputs.")

    return float(var_X[positive].mean())


def generate_synthetic_noise(
    sx: int,
    sy: int,
    noise_var: Optional[Sequence[float]] = None,
    stepsize: int = 8,
    show_plot: bool = False,
) -> np.ndarray:
    """
    Generate radial variance maps for simulating spatially-varying noise.

    Parameters
    ----------
    sx, sy : int
        Map size in x (rows) and y (cols).
    noise_var : sequence of float, optional
        Maximum variances for each noise map. Defaults to ``[1, 4, 16]``.
    stepsize : int, default 8
        Radial step size for the nested circular contours (in pixels).
    show_plot : bool, default False
        If ``True``, render diagnostic plots (requires ``matplotlib``).

    Returns
    -------
    ndarray, shape (sx, sy, K)
        Stack of variance maps, one per entry in ``noise_var``.
    """
    if noise_var is None:
        noise_var = np.array([1, 2, 4], dtype=float) ** 2
    else:
        noise_var = np.asarray(noise_var, dtype=float)

    cx = sx / 2.0 + 0.5
    cy = sy / 2.0 + 0.5
    rx = sx / 2.0

    num_noise = noise_var.size
    noise = np.zeros((sx, sy, num_noise), dtype=float)

    rr, cc = np.meshgrid(
        np.arange(1, sx + 1, dtype=float),
        np.arange(1, sy + 1, dtype=float),
        indexing="ij",
    )
    dist = np.sqrt((rr - cx) ** 2 + (cc - cy) ** 2)

    noise_contours = np.arange(rx, 0, -stepsize)

    for j in range(num_noise):
        X = np.zeros((sx, sy), dtype=float)
        var_levels = np.linspace(
            0.01 * noise_var[j], noise_var[j], num=noise_contours.size
        )
        for r, v in zip(noise_contours, var_levels):
            mask = dist <= r
            X[mask] = v
        noise[:, :, j] = X

    if show_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for show_plot=True. "
                "Install with: pip install archepy[viz]"
            ) from exc

        import numpy.ma as ma

        ncols = num_noise
        fig, axes = plt.subplots(3, ncols, figsize=(3.8 * ncols, 10), constrained_layout=True)

        if ncols == 1:
            axes = np.asarray(axes).reshape(3, 1)

        vmax = float(np.max(noise_var))
        for j in range(num_noise):
            X = noise[:, :, j]
            Xstd = np.sqrt(X)
            sample = np.random.randn(sx, sy) * Xstd

            mX = ma.masked_where(X == 0, X)
            mXstd = ma.masked_where(Xstd == 0, Xstd)
            mSample = ma.masked_where(X == 0, sample)

            im0 = axes[0, j].imshow(mX, origin="lower", aspect="equal")
            axes[0, j].set_title(f"Max variance {noise_var[j]:.2f}")
            fig.colorbar(im0, ax=axes[0, j])

            im1 = axes[1, j].imshow(mXstd, origin="lower", aspect="equal")
            axes[1, j].set_title(f"Max std dev {np.sqrt(noise_var[j]):.2f}")
            fig.colorbar(im1, ax=axes[1, j])

            im2 = axes[2, j].imshow(mSample, origin="lower", aspect="equal", vmin=0, vmax=vmax)
            axes[2, j].set_title(f"Normal noise N(0,{int(noise_var[j])})")
            fig.colorbar(im2, ax=axes[2, j])

            for row in range(3):
                axes[row, j].set_xlim(0, sy - 1)
                axes[row, j].set_ylim(0, sx - 1)
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])

        plt.show()

    return noise
