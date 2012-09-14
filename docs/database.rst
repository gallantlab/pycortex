Surface Database
================
The surface database for pycortex holds all the VTK files and transforms required to plot data on a cortical sheet.

"Database" is technically a misnomer, since all the files are simply stored in the filestore by a coded filename. To access surface reconstructions::

    from cortex import surfs
    pts, polys, norms = surfs.getVTK("AH", "fiducial", merge=True)
    #pts is a (p, 3) array, p = number of vertices
    #polys is a (f, 3) array, f = number of faces
    #norms is a (p, 3) array, or None if no normals are defined in the vtk

To retrieve a transform::

    xfm, epifile = surfs.getXfm("AH", "AH_huth", xfmtype='coord')

For a slightly flashier way to view the database immediate::

    surfs.AH.surfaces.fiducial.show()

Surfaces
--------
pycortex fundamentally operates on triangular mesh geometry computed from a subject's anatomy. Surface geometries are usually created from a marching cubes reconstruction of the segmented cortical sheet. This first reconstruction is known as the fiducial surface. The fiducial surface is inflated and cut along anatomical and functional boundaries, and is morphed by an energy metric to be on a flattened 2D surface.

Unfortunately, pycortex currently has no way of generating or editing these geometries. The recommended software for doing segmentation and flattening is Caret_. Another package which is generally more automated, but tends to fail for some subjects is Freesurfer_. Feel free to explore both options for generating cortical reconstructions.

A surface in pycortex is any VTK file specifying the triangular mesh geometry of a subject. Surfaces generally have three variables associated:

    * **Subject** : a unique identifier for the subject whom this surface belongs,
    * **Type** : the identifier for the type of geometry. These generally fall in three categories: Fiducial, inflated, and flat.
    * **Hemisphere** : which hemisphere the surface belongs to.

The VTK files for a specific subject and hemisphere must have the same number of vertices across all the different types. Without this information, the mapping from fiducial to flatmap is not preserved, and there is no way to display data on the flatmap. Caret_ automatically returns VTK files of this format. pycortex does not check the validity of surfaces, and will break in unexpected ways if the number of vertices do not match! It is your job to make sure that all surfaces are valid.

In order to plot cortical data for a subject, at least the fiducial and flat geometries must be available for that subject. Surfaces must be stored in VTK v. 1 format (also known as the ASCII format).

.. _Caret: http://brainvis.wustl.edu/wiki/index.php/Main_Page
.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu/

Accessing surfaces
^^^^^^^^^^^^^^^^^^
Two methods exist for accessing the surface data once they are committed to the database: direct command access, or via a convienient tab function.

For the direct command access, there are two call signatures::

    from cortex import surfs
    pts, polys, norms = surfs.getVTK('AH', 'fiducial', merge=True)

This returns the positions, polygons, and normals 

Adding new surfaces
^^^^^^^^^^^^^^^^^^^

Transforms
----------
Functional data, usually collected by an epi sequence, typically does not have the same scan parameters as the anatomical MPRAGE scan used to generate the surfaces. Additionally, fMRI sequences which are usually optimized for T2*, have drastically different and larger distortions than a typical T1 anatomical sequence. While automatic algorithms exist to align these two scan types, they will sometimes fail spectacularly, especially if a partial volume slice prescription is necessary.

pycortex includes a tool based on mayavi_ to do manual alignments. Please see the :module:`align` module for more information. Alternatively, if an automatic algorithm works well enough, you can also commit your own transform to the database. Transforms in pycortex always go from **fiducial to functional** space. They have four variables associated:

    * **Subject** : name of the subject, must match the surfaces used to create the transform
    * **Name** : A unique identifier for this transform
    * **type** : The type of transform -- from fiducial to functional **magnet** space, or fiducial to **coord** inate space
    * **epifile** : the filename of the functional data that the fiducial is aligned to

Transforms always store the epifile in order to allow visual validation of alignment using the :module:`align` module.

Database details
----------------
The "database" is stored in the filestore defined in defaults.json. Within the filestore, there are a sequence of directories:
