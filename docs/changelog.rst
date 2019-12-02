.. _changelog:

Changelog
==========

Unreleased
----------


1.1.0
----

**Added**

- Added function ``cortex.utils.download_subject`` to download new subjects for pycortex from web URLs. This function allows users to download FreeSurfer's ``fsaverage`` surface, with a flatmap and ROI labels made by Mark Lescroart (`#344 <https://github.com/gallantlab/pycortex/pull/344>`_)
- Vertex objects have a new method ``.map()`` that allows mapping from one surface to another (`#334 <https://github.com/gallantlab/pycortex/pull/334>`_)
- Add ``cortex.freesurfer.get_mri_surf2surf_matrix`` to create a sparse matrix implementing the ``mri_surf2surf`` command (`#334 <https://github.com/gallantlab/pycortex/pull/334>`_)
- Add function to plot and save 3D views and plot panels (`#337 <https://github.com/gallantlab/pycortex/pull/337>`_)
- Axis object can be passed to ``quickshow`` (`#325 <https://github.com/gallantlab/pycortex/pull/325>`_)
- Help menu for the 3D WebGL viewer can be accessed with a shortcut (`#319 <https://github.com/gallantlab/pycortex/pull/319>`_, `#321 <https://github.com/gallantlab/pycortex/pull/321>`_)
- New keyboard shortcuts (`list <https://gallantlab.github.io/userguide/webgl.html#keyboard-shortcuts>`_)
- Convenience function to import data preprocessed with fmriprep (`#301 <https://github.com/gallantlab/pycortex/pull/301>`_)
- Added option to use FreeSurfer's BBR for automatic alignment and function to use Freeview for manual alignment

**Fixed**

- Fix ``DataView2D`` to allow plotting of 2D datasets with quickflat
- Fix ``VertexRGB`` and ``VolumeRGB`` when alpha is not set
- Allow arbitrary positioning of the colorbar with quickflat
- Make ``quickflat`` more robust to extraneous polygons (`#333 <https://github.com/gallantlab/pycortex/pull/333>`_)
- Fix mouse behavior when unfold > 0.5 in WebGL viewer (`#330 <https://github.com/gallantlab/pycortex/pull/330>`_)
- Sub surfaces fixes (`#307 <https://github.com/gallantlab/pycortex/pull/306>`_)
- Firefox compatibility fixes  (`#306 <https://github.com/gallantlab/pycortex/pull/306>`_)
- Miscellaneous python 3 fixes

