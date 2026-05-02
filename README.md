# ArchePy

**Multi-Subject Archetypal Analysis (MS-AA) for fMRI data, in Python.**

A Python port of the [Multisubject Archetypal Analysis Toolbox](https://github.com/JesperLH/Multisubject-Archetypal-Analysis)
originally developed in MATLAB at the Technical University of Denmark by Hinrich, Bardenfleth, Røge,
Churchill, Madsen, and Mørup. Implements both spatial and temporal MS-AA with optional GPU
acceleration via [CuPy](https://cupy.dev/).

> [!IMPORTANT]
> **Academic / non-profit use only.** This package is distributed under the same license as the
> upstream MATLAB toolbox, which restricts use to academic and non-profit institutions.
> Commercial users must obtain a separate license from DTU. See [LICENSE](LICENSE) for full terms.

> [!NOTE]
> **Related packages.** ArchePy is one of several Python tools for archetypal analysis.
> If MS-AA isn't what you need, you may want:
> - [`archetypes`](https://github.com/aleixalcacer/archetypes) — general scikit-learn-compatible AA (BSD-3).
> - [`py_pcha`](https://github.com/ulfaslak/py_pcha) — Python implementation of PCHA, by one of MS-AA's
>   upstream authors (Mørup).
>
> ArchePy is specifically the multi-subject, heteroscedastic, GPU-accelerated variant for
> neuroimaging, ported from the [DTU MATLAB toolbox](https://github.com/JesperLH/Multisubject-Archetypal-Analysis).

---

## Installation

```bash
# CPU only
pip install archepy

# With GPU acceleration (CUDA 12.x — requires a CUDA-capable NVIDIA GPU)
pip install archepy[gpu]

# With fMRI helpers (NIfTI loading)
pip install archepy[fmri]

# Everything
pip install archepy[all]
```

For development:

```bash
git clone https://github.com/the-brAIn-lab/archepy
cd archepy
pip install -e .[dev]
pytest
```

## Requirements

- **Python** 3.9 or newer.
- **Required:** [NumPy](https://numpy.org/) >= 1.23.
- **Optional, for GPU acceleration:** [CuPy](https://cupy.dev/) (CUDA 12.x via `archepy[gpu]`).
  CUDA 11.x users should install `cupy-cuda11x` manually.
- **Optional, for fMRI helpers:** [nibabel](https://nipy.org/nibabel/) >= 4.0 (via `archepy[fmri]`).
- **Optional, for plotting:** [matplotlib](https://matplotlib.org/) >= 3.5 (via `archepy[viz]`).

The full dependency list is declared in [`pyproject.toml`](pyproject.toml).

## Quick start

```python
import numpy as np
from archepy import Subject, multi_subject_aa

# Each subject's data: a (T, V) array — time points × voxels.
# (For temporal MS-AA, see archepy.multi_subject_aa_T which uses (V, T).)
rng = np.random.default_rng(0)
n_subjects, T, V = 3, 200, 1000

subjects = [
    Subject(
        X=rng.standard_normal((T, V)).astype(float),
        sX=rng.standard_normal((T, V)).astype(float),
    )
    for _ in range(n_subjects)
]

# Fit MS-AA with K=10 archetypes.
results, C, cost, varexpl, elapsed = multi_subject_aa(
    subjects,
    noc=10,
    opts={
        "maxiter": 100,
        "conv_crit": 1e-6,
        "use_gpu": False,           # set True if you installed archepy[gpu]
        "heteroscedastic": True,
        "rngSEED": 42,
    },
)

print(f"Variance explained: {varexpl * 100:.1f}%   ({elapsed:.1f}s, {len(cost)} iters)")
print(f"Shared generator C shape:        {C.shape}")
print(f"Subject 0 archetypal mixture S:  {results[0]['S'].shape}")
print(f"Subject 0 reconstructed XC:      {results[0]['sXC'].shape}")
```

For a fuller end-to-end example using real fMRI data, see
[`examples/01_fit_msaa_flexible.ipynb`](examples/01_fit_msaa_flexible.ipynb).

## What this package contains

| Module | Purpose |
|---|---|
| `archepy.multi_subject_aa` | Spatial MS-AA: archetypes are linear combinations of time points, mixed across voxels. |
| `archepy.multi_subject_aa_T` | Temporal MS-AA: archetypes are linear combinations of voxels, mixed across time. |
| `archepy.Subject` | Per-subject data container (`X`, `sX`). |
| `archepy.furthest_sum` | FurthestSum initialization (CPU). |
| `archepy.init.furthest_sum_gpu` | FurthestSum initialization (GPU, CuPy). |
| `archepy.fmri.estimate_background_noise` | Estimate noise threshold from a NIfTI file. |
| `archepy.fmri.generate_synthetic_noise` | Generate synthetic radial noise maps. |

## Testing

ArchePy includes a smoke-test suite that exercises the FurthestSum
initialization and runs end-to-end MS-AA fits on planted low-rank data.
After installing the dev extras (`pip install -e .[dev]`), run:

```bash
pytest                                          # all tests
pytest --cov=archepy --cov-report=term-missing  # with coverage
```

The full suite finishes in seconds. CI runs the same suite on Linux, macOS,
and Windows across Python 3.9-3.12 — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Contributing

Contributions are welcome — bug reports, feature requests, documentation
improvements, and pull requests. Please:

- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev-environment setup,
  code style, and PR workflow.
- Review the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before participating.
- For non-trivial changes, open an issue first to discuss.

By contributing, you agree that your contributions will be distributed under
the same academic / non-profit terms as the rest of the package.

## Citing

If you use ArchePy in a publication, please cite **both** the original paper and this port:

```bibtex
@article{hinrich2016archetypal,
  title   = {Archetypal Analysis for Modeling Multisubject {fMRI} Data},
  author  = {Hinrich, Jesper L\o{}ve and Bardenfleth, Sophia Elizabeth and
             R\o{}ge, Rasmus and Churchill, Nathan W. and Madsen, Kristoffer H.
             and M\o{}rup, Morten},
  journal = {IEEE Journal on Selected Topics in Signal Processing},
  volume  = {10},
  number  = {7},
  pages   = {1160--1171},
  year    = {2016},
  doi     = {10.1109/JSTSP.2016.2595103}
}

@software{archepy,
  author  = {Alex Shepherd},
  title   = {ArchePy: Multi-Subject Archetypal Analysis for fMRI in Python},
  url     = {https://github.com/the-brAIn-lab/archepy},
  version = {0.1.0},
  year    = {2026}
}
```

## Acknowledgments

This package would not exist without the original MATLAB toolbox by
**Jesper L. Hinrich, Sophia E. Bardenfleth, and Morten Mørup** at DTU's
Section for Cognitive Systems. The Python port preserves their algorithms
faithfully; any bugs in this port are mine alone.

## Status

Alpha. The algorithm code is a direct port of the MATLAB reference and reproduces
its outputs, but the Python API may evolve before 1.0. Pin to an exact version if you
need stability.
