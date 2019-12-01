:orphan:

.. _api_reference:

====================
Python API Reference
====================

These are the classes and functions in pycortex.

.. contents::
   :local:
   :depth: 2


.. currentmodule:: cortex

Most commonly used modules
==========================

quickflat
---------

.. automodule:: cortex.quickflat

.. autosummary::
    :toctree:generated/

    add_curvature
    add_data
    add_rois
    add_sulci
    add_hatch
    add_colorbar
    add_custom
    add_cutout
    make_figure
    make_png
    make_svg
    get_flatmask
    get_flatcache


webgl
-----

.. automodule:: cortex.webgl

.. autosummary::
    :toctree:generated/

    show
    make_static


dataset
-------

.. automodule:: cortex.dataset

.. autosummary::
    :toctree:generated/
    :template:class.rst

    Volume
    Volume2D
    VolumeRGB
    Vertex
    Vertex2D
    VertexRGB
    Dataset

All the other modules
=====================

align
------

.. automodule:: cortex.align

.. autosummary::
    :toctree:generated/

    manual
    automatic
    autotweak
    

anat
------

.. automodule:: cortex.anat

.. autosummary::
    :toctree:generated/

	brainmask
	whitematter
    voxelize


database
--------

.. automodule:: cortex.database

.. autosummary::
    :toctree:generated/
    :template:class.rst

    Database


freesurfer
----------

.. automodule:: cortex.freesurfer

.. autosummary::
    :toctree:generated/

    get_paths
    autorecon
    flatten
    import_subj
    import_flat
    show_surf
	make_fiducial
	parse_surf
	parse_curv
	parse_patch
	get_surf
	get_curv
	write_dot
	read_dot
	write_decimated
	SpringLayout
	stretch_mwall


mapper
------

.. automodule:: cortex.mapper

.. autosummary::
    :toctree:generated/

    Mapper
    get_mapper


mni
---

.. automodule:: cortex.mni

.. autosummary::
    :toctree:generated/

    compute_mni_transform
    transform_to_mni
    transform_surface_to_mni
    transform_mni_to_subject


polyutils
---------

.. automodule:: cortex.polyutils

.. autosummary::
    :toctree:generated/
    :template:class.rst

    Surface
    Distortion


segment
-------

.. automodule:: cortex.segment

.. autosummary::
    :toctree:generated/

    init_subject
    fix_wm
    fix_pia
    cut_surface


surfinfo
--------

.. automodule:: cortex.surfinfo

.. autosummary::
    :toctree:generated/

    curvature
    distortion
    thickness
    tissots_indicatrix
    flat_border


utils
------

.. automodule:: cortex.utils

.. autosummary::
    :toctree:generated/

    add_roi
    anat2epispace
    get_aseg_mask
    get_cmap
    get_cortical_mask
    get_ctmmap
    get_ctmpack
    get_dropout
    get_hemi_masks
    get_roi_mask
    get_roi_masks
    get_roi_verts
    get_vox_dist
    make_movie
    vertex_to_voxel
 

volume
------

.. automodule:: cortex.volume

.. autosummary::
    :toctree:generated/

    unmask
    mosaic
    epi2anatspace
    anat2epispace
    epi2anatspace_fsl
    anat2epispace_fsl
    show_slice
    show_mip
    show_glass
    

xfm
------

.. automodule:: cortex.xfm

.. autosummary::
    :toctree:generated/

    Transform