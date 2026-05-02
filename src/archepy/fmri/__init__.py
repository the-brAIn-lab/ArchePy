"""
fMRI-specific utilities.

These helpers depend on optional packages (``nibabel``, ``matplotlib``).
Install with ``pip install archepy[fmri]`` (NIfTI loading) and/or
``pip install archepy[viz]`` (plotting).
"""

from archepy.fmri.noise import estimate_background_noise, generate_synthetic_noise

__all__ = ["estimate_background_noise", "generate_synthetic_noise"]
