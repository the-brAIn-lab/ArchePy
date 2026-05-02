# Changelog

All notable changes to ArchePy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version numbers follow `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible API changes.
- **MINOR** — backwards-compatible additions of functionality.
- **PATCH** — backwards-compatible bug fixes.

While ArchePy is at `0.x.y`, the API is not yet considered stable; minor
version bumps may include breaking changes. We will document any such
break clearly in the release notes for that version.

## [Unreleased]

### Added
- _Items added since the last release go here._

### Changed
- _Items modified since the last release go here._

### Deprecated
- _Items scheduled for removal go here._

### Removed
- _Items removed in this release go here._

### Fixed
- _Bug fixes go here._

### Security
- _Security-related changes go here._

## [0.1.0] — Initial release

### Added
- Spatial Multi-Subject Archetypal Analysis (`multi_subject_aa`), faithful
  Python port of the MATLAB ``MultiSubjectAA.m`` from Hinrich et al. 2016.
- Temporal Multi-Subject Archetypal Analysis (`multi_subject_aa_T`), Python
  port of ``MultiSubjectAA_T.m``.
- `Subject` and `SubjectT` data containers for spatial and temporal variants.
- FurthestSum initialization on CPU (`furthest_sum`) and GPU
  (`archepy.init.furthest_sum_gpu`, lazily imported via CuPy).
- Optional GPU acceleration via CuPy (`pip install archepy[gpu]`).
- fMRI helpers under `archepy.fmri`:
  - `estimate_background_noise` — estimate a noise variance threshold from a
    NIfTI file.
  - `generate_synthetic_noise` — generate radial variance maps for
    simulating spatially-varying scanner noise.
- Smoke tests covering `furthest_sum` (CPU + kernel modes) and end-to-end
  MS-AA on planted low-rank data.
- GitHub Actions CI matrix: Linux/macOS/Windows × Python 3.9–3.12.
- Documentation skeleton in ``docs/``.
- Example notebook in ``examples/``.

### Notes
- ArchePy is distributed under the same license as the upstream MATLAB
  toolbox: **academic / non-profit use only**. Commercial use requires a
  separate license from DTU. See [LICENSE](LICENSE) for full terms.

[Unreleased]: https://github.com/the-brAIn-lab/archepy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/the-brAIn-lab/archepy/releases/tag/v0.1.0
