API reference
=============

This page documents the public API of ArchePy. Internal modules
(those starting with an underscore, e.g. ``archepy._utils``) are not part of
the public API and may change without deprecation notice.

Spatial MS-AA
-------------

.. autoclass:: archepy.Subject
   :members:
   :show-inheritance:

.. autofunction:: archepy.multi_subject_aa

Temporal MS-AA
--------------

.. autoclass:: archepy.SubjectT
   :members:
   :show-inheritance:

.. autofunction:: archepy.multi_subject_aa_T

Initialization
--------------

.. autofunction:: archepy.furthest_sum

.. autofunction:: archepy.init.furthest_sum_gpu

fMRI helpers
------------

.. autofunction:: archepy.fmri.estimate_background_noise

.. autofunction:: archepy.fmri.generate_synthetic_noise
