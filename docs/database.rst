Surface Database
================
The surface database for pycortex holds all the VTK files and transforms required to plot data on a cortical sheet.

"Database" is technically a misnomer, since all the files are simply stored in the filestore by a coded filename. To access surface reconstructions::

    from cortex import surfs
    pts, poly = surfs.getSurf("AH", "fiducial", merge=True)
    #pts is a (p, 3) array, p = number of vertices
    #polys is a (f, 3) array, f = number of faces

To retrieve a transform::

    xfm = surfs.getXfm("AH", "AH_huth", xfmtype='coord')

For a slightly flashier way to view the database immediately::

    surfs.AH.surfaces.fiducial.show()

Surfaces
--------
pycortex fundamentally operates on triangular mesh geometry computed from a subject's anatomy. Surface geometries are usually created from a `marching cubes`_ reconstruction of the segmented cortical sheet. This undistorted reconstruction in the original anatomical space is known as the fiducial surface. The fiducial surface is inflated and cut along anatomical and functional boundaries, and is morphed by an energy metric to be on a flattened 2D surface.

Unfortunately, pycortex currently has no way of generating or editing these geometries directly. The recommended software for doing segmentation and flattening is Freesurfer_. Another package which is generally more user friendly is Caret_. pycortex includes some utility functions to interact with Freesurfer_, documented '''HERE'''.

A surface in pycortex is any file specifying the triangular mesh geometry of a subject. Surfaces generally have three variables associated:

    * **Subject** : a unique identifier for the subject whom this surface belongs,
    * **Type** : the identifier for the type of geometry. These generally fall in three categories: Fiducial, inflated, and flat.
    * **Hemisphere** : which hemisphere the surface belongs to.

The surface files for a specific subject and hemisphere must have the same number of vertices across all the different types. Without this information, the mapping from fiducial to flatmap is not preserved, and there is no way to display data on the flatmap. Freesurfer_ surfaces preserve this relationship, and can be automatically imported into the database. pycortex does not check the validity of surfaces, and will break in unexpected ways if the number of vertices do not match! It is your job to make sure that all surfaces are valid.

In order to plot cortical data for a subject, at least the fiducial and flat geometries must be available for that subject. Surfaces must be stored in VTK v. 1 format (also known as the ASCII format).

.. _marching cubes: http://en.wikipedia.org/wiki/Marching_cubes
.. _Caret: http://brainvis.wustl.edu/wiki/index.php/Main_Page
.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu/

Accessing surfaces
^^^^^^^^^^^^^^^^^^
Two methods exist for accessing the surface data once they are committed to the database: direct command access, or via a convienient tab complete interface.

Command access
""""""""""""""
For the direct command access, there are two call signatures::

    from cortex import surfs
    pts, polys = surfs.getSurf('AH', 'fiducial', merge=True)

This returns the points and polygons of the given subject and surface type. Hemisphere defaults to "both", and since merge is true, they are vertically stacked **left, then right**. The polygon indices are shifted up for the right hemisphere to make a single unified geometry.

With merge=False, the return looks different::

    left, right = surfs.getSurf('AH', 'fiducial', merge=False)
    lpts, lpolys = left
    rpts, rpolys = right

If you only specify hemisphere="left" or "right", only one hemisphere will be returned, and the return will again be only points, polygons, and normals.

Tab interface
"""""""""""""
An alternate way to browse the database is using ipython_ and its tab completion feature. If you type the following::

    In [1]: from cortex import surfs
    In [2]: surfs.

Then press tab, a list of subjects will appear. For example::

    In [2]: surfs.
    surfs.AH         surfs.getFiles   surfs.JG         surfs.MO         surfs.TC
    surfs.AV         surfs.getVTK     surfs.loadVTK    surfs.NB         surfs.TN
    surfs.DS         surfs.getXfm     surfs.loadXfm    surfs.NB1        surfs.WH
    surfs.getCoords  surfs.JG         surfs.ML         surfs.SN         surfs.WH1

Selecting the subject **AH** and tabbing gives you additional choices::

    In [3]: surfs.AH.
    surfs.AH.anatomical  surfs.AH.surfaces    surfs.AH.transforms

    In [4]: surfs.AH.surfaces.
    surfs.AH.surfaces.ellipsoid      surfs.AH.surfaces.inflated
    surfs.AH.surfaces.fiducial       surfs.AH.surfaces.raw
    surfs.AH.surfaces.flat           surfs.AH.surfaces.superinflated

Selecting "surfaces" gives you a list of all surface types associated with that subject. Here, we see that the subject "AH" has surfaces from almost every stage of flattening: raw, fiducial, inflated, superinflated, ellipsoid, and flat.

Finally, selecting one surface type will give you two new functions: get, and show::
    
    In [5]: left, right = surfs.AH.surfaces.fiducial.get()
    In [6]: surfs.AH.surfaces.fiducial.show()

.. _ipython: http://ipython.org/

Adding new surfaces
^^^^^^^^^^^^^^^^^^^
Surface management is implemented through your file manager. To add a new surface to an existing subject, copy the surface file into ``{$FILESTORE}/{$SUBJECT}/surfaces/`` with the format ``{type}_{hemi}.{format}``, where hemi is lh or rh, and format is one of **OFF**, **VTK**, or an **npz** file with keys 'pts' and 'polys'. If you have a python session with pycortex imported already, please reload the session. The new surfaces should be accessible via the given interfaces immediately.

In order to adequately utilize all the functions in pycortex, please add the fiducial, inflated, and flat geometries for both hemispheres. Again, make sure that all the surface types for a given subject and hemisphere have the same number of vertices, otherwise unexpected things may happen!

Transforms
----------
Transformations in pycortex are stored as **affine** matrices encoded in magnet isocenter space, as defined in the nifti headers.

Accessing transforms
^^^^^^^^^^^^^^^^^^^^
Similar to the surfaces, transforms can be access through two methods: direct command access, and the tab interface.

Command access looks like this::

    from cortex import surfs
    xfm = surfs.getXfm("AH", "AH_huth")

Tab complete looks like this::

    In [1]: from cortex import surfs
    In [2]: surfs.AH.transforms
    Out[2]: Transforms: [AH_shinji,AH_huth]

    In [3]: surfs.AH.transforms['AH_huth'].coord
    Out[5]: 
    array([[ -0.42912749,   0.00749045,   0.00749159,  48.7721599 ],
           [ -0.00681025,  -0.42757105,   0.03740662,  47.36464665],
           [  0.00457734,   0.0210264 ,   0.24117264,  10.44101855],
           [ -0.        ,   0.        ,   0.        ,   1.        ]])


Adding new transforms
^^^^^^^^^^^^^^^^^^^^^
Transforms from anatomical space to functional space are notoriously tricky. Automated algorithms generally give results optimized for various global energy metrics, but do not attempt to target the alignments for your ROIs. It is highly recommended that you use the included aligner to make your affine transforms. To add a transform, either directly create a transform json in ``{$FILESTORE}/transforms/``, or use this command::

    from cortex import surfs
    surfs.loadXfm(subject, xfmname, xfm, xfmtype='magnet', reference='path_to_functional.nii')

.. _database-masks:

Masks
^^^^^
One of the fundamental reasons for carefully aligning surfaces is to allow the creation and use of cortical masks. This limits the number of voxels you need to model. Traditionally, these masks are created by selecting the set of nearest neighbor voxels for each vertex on the transformed surface. Unfortunately, pycortex's advanced per-pixel mapping precludes the use of this simple mask, since faces could potentially intersect with voxel corners which are not in this simple mask. Thus, the default masks in pycortex use a distance metric to compute mask membership.

Masks were added into pycortex in May 2013, due to previous issues with masked data and the addition of the per-pixel mapping. Masked datasets are further discussed in the Datasets_ page.

Retrieving a mask
"""""""""""""""""
A mask is specified by three variables: subject name, transform name, and mask type. pycortex defines two named masks for each transform by default. These are the ``thick`` and the ``thin`` masks. They correspond to a distance of 8 mm and 2 mm, respectively, from any given cortical vertex. If the subject has both pial and white matter surfaces, all voxels of exactly the cortical thickness distance from each vertex are selected from the fiducial surface. To retrieve the ``thick`` mask::

    from cortex import surfs
    mask = surfs.getMask(subject, xfmname, "thick")

Additionally, masks corresponding to known mapper types (such as 'nearest' and 'trilinear') will also be automatically generated and recorded by the database when requested.

Loading a mask
""""""""""""""
If you use a custom mask for any reason, it is highly recommended that you load it into the database for future reference. It will allow more seamless Datasets_ integration, and will prevent it from being lost. To add a custom mask to the database::

    from cortex import surfs
    surfs.loadMask(subject, xfmname, masktype, mask)

Database details
----------------
pycortex implements a simple flat-file database to store transforms and surfaces. By default, the filestore is in ``INSTALL_DATA/share/pycortex/``. This location can be customized in your options.cfg file.

Within the filestore, each subject has their own directory containing all associated data. Each subject has a few subdirectories:

    * surfaces: formatted as ``{type}_{hemi}.{format}``
    * transforms: each subdirectory is a transform. Each transform subdirectory contains two files: matrices.xfm, and reference.nii.gz. Masks are also stored in the transforms directory.
    * 

The ctmcache holds the sequence of files necessary for the webgl viewer. OpenCTM_ is a geometry specification that allows very small files to reduce bandwidth. Files are stored with the format ``{subject}_{transform}_[{types}]_{compression}_{level}.{suffix}``. Each subject and transform is associated with a triplet of files called a "ctmpack". Each ctmpack contains a json file specifying the limits of the data, a ctm file consisting of concatenated left and right hemispheres, and an SVG consisting of the roi's with the data layers deleted. There is a unique ctmpack for each subject, transform, and set of included inflations. Raw CTMs are generated for view.webshow, whereas MG2 CTM's are generated for static WebGL views. These files are considered disposable, and are generated on demand.

The flatcache holds the voxel indices for quickly generating a flatmap. They have the format ``{subject}_{transform}_{height}_{date}.pkl``. A different flatcache must be generated for each datamap height. These files are also disposable and are generated on demand. This cache allows quickflat to satisfy its namesake.

Overlays are stored as SVG_'s. This is where surface ROIs are defined. Since these surface ROIs are invariant to transform, only one ROI map is needed for each subject. These SVGs are automatically created for a subject if you call ``cortex.add_roi``. ROI overlays are created and edited in Inkscape_. For more information, see :module:`svgroi.py`.

References contain the functional scans that are paired with a transform. They are typically in Nifti_ format (*.nii), but can be any format that is understood by nibabel_. These are stored to ensure that we know what the reference for any transform was. This makes it possible to visually verify and tweak alignments, as well as keep a static store of images for future coregistrations.

Surfaces may be stored in any one of OFF, VTK, or npz formats. The highest performance is achieved with NPZ since it is binary and compressed. VTK is also efficient, having a cython module to read files.

Transforms are saved as json-encoded text files. They have the format ``{subject}_{transform}.xfm``. There are four fields in this JSON structure: ``subject``, ``epifile``, ``coord``, ``magnet``. ``epifile`` gives the name of the epi file that served as the reference for this transform. ``coord`` stores the transform from fiducial to coordinate space (for fast index lookups). ``magnet`` stores the transform from the fiducial to the magnet space, as defined in the return of ``nibabel.get_affine()``.

.. _OpenCTM: http://openctm.sourceforge.net/
.. _SVG: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _Inkscape: http://inkscape.org/
.. _Nifti: http://nifti.nimh.nih.gov/nifti-1/
.. _nibabel: http://nipy.sourceforge.net/nibabel/