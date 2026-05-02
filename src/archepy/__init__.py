"""
ArchePy — Multi-Subject Archetypal Analysis for fMRI data.

Python port of the MATLAB Multisubject Archetypal Analysis Toolbox by
Hinrich, Bardenfleth, Røge, Churchill, Madsen, and Mørup (DTU, 2016).

Academic / non-profit use only. See LICENSE.
"""

from archepy._version import __version__
from archepy.core.spatial import Subject, multi_subject_aa
from archepy.core.temporal import SubjectT, multi_subject_aa_T
from archepy.init.furthest_sum import furthest_sum

__all__ = [
    "__version__",
    "Subject",
    "SubjectT",
    "multi_subject_aa",
    "multi_subject_aa_T",
    "furthest_sum",
]
