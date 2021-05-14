.. _changelog:

Changelog
==========


1.2.1
-----

**Added**

- Allow option to perform automatic alignment with Freesurfer's BBR for volumes without distortion correction (`5d7d6ca <https://github.com/gallantlab/pycortex/commit/5d7d6ca73845986fcc182899289218132be99604>`_)
- Add ``line_lanczos`` to available mappers (`7721253 <https://github.com/gallantlab/pycortex/commit/77212535da1ad17930697b8cf2f497ae09d2898b>`_)
- Use Freesurfer's BBR by default for automatic alignment (`#384 <https://github.com/gallantlab/pycortex/pull/384>`_)


**Fixed**

- Parameter names for ``freesurfer.import_flat`` (`4371b26 <https://github.com/gallantlab/pycortex/commit/4371b2633a0b7180e3893484af61a941ba5029b9>`_)
- Fix left/right location of hemispheres for medial-pivot view (`c3b029e <https://github.com/gallantlab/pycortex/commit/c3b029e96c7ffa67c8c35c7af47c045e0161abc3>`_)
- Animation functions in ``webgl.view`` (`90c1431 <https://github.com/gallantlab/pycortex/commit/90c1431e20c8cc76bb593f7fcd5416ca476508a7>`_)
- Various fixes to 2D volumes
- String encoding in ``openctm.pyx`` (`6dc80db <https://github.com/gallantlab/pycortex/commit/6dc80db6305f0ad97e7b857e083958636fab2233>`_)


1.2.0
-----

**Added**

- More functionality to import fMRIprep-processed data (`#315 <https://github.com/gallantlab/pycortex/pull/315>`_)
- Function to rotate flatmaps (`f88a <https://github.com/gallantlab/pycortex/commit/f88a195382c9611c492eda2c525e9ab5595bcc37>`_)
- Option to ignore NaNs when averaging across the thickness of cortex (`#355 <https://github.com/gallantlab/pycortex/pull/355>`_)
- New views for ``cortex.export.plot_panels`` and passing of additional kwargs (`#361 <https://github.com/gallantlab/pycortex/pull/361>`_, `#362 <https://github.com/gallantlab/pycortex/pull/362>`_)
- Specify colorbar location with default positionings for quickflat (`#364 <https://github.com/gallantlab/pycortex/pull/364>`_)
- Pass custom colors for ``VolumeRGB`` (`#366 <https://github.com/gallantlab/pycortex/pull/366>`_)


**Fixed**

- Bug with automatic creation of SVGs with ROIs (`#340 <https://github.com/gallantlab/pycortex/pull/340>`_)
- Python 2/3 compatibilities for svgoverlay (`#352 <https://github.com/gallantlab/pycortex/pull/352>`_)
- ``from_freesurfer``/``to_freesurfer`` handling of NIFTI images (`#354 <https://github.com/gallantlab/pycortex/pull/354>`_)
- ROI label display parameters (`#356 <https://github.com/gallantlab/pycortex/pull/356>`_)
- Improve documentation for install and demo (`#359 <https://github.com/gallantlab/pycortex/pull/359>`_)
- Inkscape compatibility with newer versions (`#365 <https://github.com/gallantlab/pycortex/pull/365>`_)
- Use correct contrast for automatic alignment with FreeSurfer's ``bbregister`` (`39d8 <https://github.com/gallantlab/pycortex/commit/39d8fe6766f4ecacd0251a5798ea354528ec8eae>`_)


1.1.1
-----

**Fixed**

- Installation fix for PyPi

1.1.0
-----

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

