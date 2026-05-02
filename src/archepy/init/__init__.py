"""Archetype initialization strategies (FurthestSum, etc.)."""

from archepy.init.furthest_sum import furthest_sum

__all__ = ["furthest_sum"]


def furthest_sum_gpu(*args, **kwargs):
    """
    Lazy wrapper around the GPU FurthestSum.

    Imported lazily so that ``import archepy`` does not fail when CuPy
    is not installed. Install GPU support with ``pip install archepy[gpu]``.
    """
    from archepy.init._gpu import furthest_sum_gpu as _impl

    return _impl(*args, **kwargs)
