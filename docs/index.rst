ArchePy: Multi-Subject Archetypal Analysis for fMRI
====================================================

**ArchePy** is a Python port of the Multisubject Archetypal Analysis (MS-AA)
Toolbox originally developed in MATLAB at the Technical University of Denmark
by Hinrich, Bardenfleth, Røge, Churchill, Madsen, and Mørup.

It implements both spatial and temporal MS-AA with optional GPU acceleration
via `CuPy <https://cupy.dev/>`_ and includes minimal helpers for working with
fMRI NIfTI data.

.. important::

   **Academic / non-profit use only.** ArchePy is distributed under the same
   license as the upstream MATLAB toolbox, which restricts use to academic and
   non-profit institutions. Commercial users must obtain a separate license
   from DTU. See the :doc:`license <license>` page for full terms.

Quick example
-------------

.. code-block:: python

   import numpy as np
   from archepy import Subject, multi_subject_aa

   rng = np.random.default_rng(0)
   subjects = [
       Subject(
           X=rng.standard_normal((200, 1000)),
           sX=rng.standard_normal((200, 1000)),
       )
       for _ in range(3)
   ]

   results, C, cost, varexpl, elapsed = multi_subject_aa(
       subjects, noc=10, opts={"maxiter": 100, "rngSEED": 42}
   )
   print(f"Variance explained: {varexpl * 100:.1f}%")

For an end-to-end fit on real fMRI data, see the
:doc:`tutorials <tutorials>` section.

Installation
------------

.. code-block:: bash

   pip install archepy           # CPU only
   pip install archepy[gpu]      # with CuPy (CUDA 12.x)
   pip install archepy[fmri]     # with nibabel
   pip install archepy[all]      # everything

See the `README <https://github.com/the-brAIn-lab/archepy#installation>`_
for full installation details and dev setup.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog
   license

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   GitHub repository <https://github.com/the-brAIn-lab/archepy>

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
