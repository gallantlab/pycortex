Segmentation Tutorial
=====================
In order to plot data, you will need to create surfaces for your particular subject. General surfaces such as fsaverage is NOT recommended, since individual subject anatomay can be highly variable. Averaging across subjects will destroy your signal specificity!  The recommended path for generating surfaces is with Freesurfer_.

This document will walk you through a general guide for how to create a Freesurfer surface usable with pycortex.

Installation
------------
Unfortunately, there is no simple unified method for installing Freesurfer on your computer. You will need to download a package, then acquire a (free) registration. For additional instructions, go to http://surfer.nmr.mgh.harvard.edu/fswiki/Download.

Segmentation
------------
Segmentation is the process of identifying the boundary between white matter and gray matter, and between gray matter and dura. With Caret_, only one surface is estimated: the midway point between white matter and pia, also known as the "fiducial" surface. This boundary is converted into a triangular mesh representation using a `marching cubes <http://en.wikipedia.org/wiki/Marching_Cubes>`_ algorithm. However, segmenting this boundary is nontrivial.

Pycortex wraps many of the segmentation steps from Freesurfer_ for a simpler, more integrated process. Generally, the four functions in cortex.segment is all that's required to go from anatomical image to segmented and flattened surface.

.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu/
.. _Caret: http://brainvis.wustl.edu/wiki/index.php/Caret:Download

Quickstart
----------

	#. Collect a T1 MPRAGE image of the subject. `Preferred protocols <http://freesurfer.net/fswiki/FreeSurferWiki?action=AttachFile&do=get&target=FreeSurfer_Suggested_Morphometry_Protocols.pdf>`_: some variety of multiecho T1.
	#. cortex.segment.init_subject