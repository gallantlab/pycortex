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



quickflat
---------

.. automodule:: cortex.quickflat

.. autosummary::
    :toctree:generated/

    make_figure
    make
    make_png


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


database
--------

.. automodule:: cortex.database

.. autosummary::
	:toctree:generated/

	Database


mapper
------

.. automodule:: cortex.mapper

.. autosummary::
	:toctree:generated/

	Mapper
	get_mapper


svgroi
------

.. automodule:: cortex.svgroi

.. autosummary::
	:toctree:generated/

	ROIPack
	ROI
	get_roipack


utils
------

.. automodule:: cortex.utils

.. autosummary::
	:toctree:generated/

	anat2epispace
	get_aseg_mask
	get_cortical_mask
	get_ctmmap
	get_ctmpack
	get_dropout
	get_hemi_masks
	get_roi_masks
	get_roi_mask
	get_roi_verts
	get_vox_dist


segment
-------

.. automodule:: cortex.segment

.. autosummary::
	:toctree:generated/

	init_subject
	fix_wm
	fix_pia
	cut_surface


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


anat
------

.. automodule:: cortex.anat

.. autosummary::
	:toctree:generated/

	voxelize
	

xfm
------

.. automodule:: cortex.xfm

.. autosummary::
	:toctree:generated/

	Transform


align
------

.. automodule:: cortex.align

.. autosummary::
	:toctree:generated/

	manual
	automatic
	autotweak


polyutils
---------

.. automodule:: cortex.polyutils

.. autosummary::
	:toctree:generated/
	:template:class.rst

	Surface
	Distortion
