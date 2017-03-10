Surface Database
================

Pycortex creates and maintains a simple flat-file database store all the data required to plot data on a cortical sheet (surfaces, transforms, masks, regions-of-interest, etc.). By default, the filestore is in ``INSTALL_DATA/share/pycortex/``. This location can be customized in your ``options.cfg`` file. You can find the filestore directory by running::

    import cortex
    cortex.database.default_filestore

Within the filestore, each subject has their own directory containing all associated data.


Anatomical scans
----------------

Each subject must have an anatomical scan.


Cache
-----

The cache holds the sequence of files necessary for the webgl viewer. OpenCTM_ is a geometry specification that allows very small files to reduce bandwidth. Files are stored with the format ``{subject}_{transform}_[{types}]_{compression}_{level}.{suffix}``. Each subject and transform is associated with a triplet of files called a "ctmpack". Each ctmpack contains a json file specifying the limits of the data, a ctm file consisting of concatenated left and right hemispheres, and an SVG_ consisting of the roi's with the data layers deleted. There is a unique ctmpack for each subject, transform, and set of included inflations. Raw CTMs are generated for view.webshow, whereas MG2 CTM's are generated for static WebGL views. These files are considered disposable, and are generated on demand.

The flatcache holds the voxel indices for quickly generating a flatmap. They have the format ``{subject}_{transform}_{height}_{date}.pkl``. A different flatcache must be generated for each datamap height. These files are also disposable and are generated on demand. This cache allows quickflat to satisfy its namesake.


Surfaces
--------

Pycortex fundamentally operates on triangular mesh geometry computed from a subject's anatomy. Surface geometries are usually created from a `marching cubes`_ reconstruction of the segmented cortical sheet. This undistorted reconstruction in the original anatomical space is known as the fiducial surface. The fiducial surface is inflated and cut along anatomical and functional boundaries and is morphed by an energy metric to be on a flattened 2D surface.

Unfortunately, pycortex currently has no way of generating or editing these geometries directly. The recommended software for doing segmentation and flattening is Freesurfer_. Another package which is generally more user-friendly is Caret_. pycortex includes some utility functions to interact with Freesurfer_, documented '''HERE'''.

A surface in pycortex is any file specifying the triangular mesh geometry of a subject. Surfaces may be stored in any one of **OFF**, **VTK**, or **npz** formats. The highest performance is achieved with **npz** since it is binary and compressed. VTK is also efficient, having a `Cython` module to read files. Inside the filestore, surface names are formatted as ``{type}_{hemisphere}.{format}``. Surfaces generally have three variables associated:

    * **Subject** : a unique subject identifier
    * **Type** : the identifier for the type of geometry, **fiducial**, **inflated**, or **flat**
    * **Hemisphere** : the brain hemisphere of the surface, **lh** or **rh**

The surface files for a specific subject and hemisphere must have the same number of vertices across all the different types. Without this information, the mapping from fiducial to flatmap is not preserved, and there is no way to display data on the flatmap. Freesurfer_ surfaces preserve this relationship, and can be automatically imported into the database. pycortex does not check the validity of surfaces, and will break in unexpected ways if the number of vertices do not match! It is your job to make sure that all surfaces are valid.

In order to plot cortical data for a subject, at least the fiducial and flat geometries must be available for that subject. Surfaces must be stored in VTK v. 1 format (also known as the ASCII format).


Accessing surfaces
~~~~~~~~~~~~~~~~~~
Two methods exist for accessing the surface data once they are committed to the database: direct command access, or via a convienient tab completion interface.

Command access
~~~~~~~~~~~~~~
For the direct command access, there are two call signatures::

    import cortex
    pts, polys = cortex.db.get_surf('AH', 'fiducial', merge=True)

This returns the points and polygons of the given subject and surface type. Hemisphere defaults to "both", and since ``merge`` is true, they are vertically stacked **left, then right**. The polygon indices are shifted up for the right hemisphere to make a single unified geometry.

With ``merge=False``, the return looks different::

    left, right = cortex.db.get_surf('AH', 'fiducial', merge=False)
    lpts, lpolys = left
    rpts, rpolys = right

If you only specify ``hemisphere='left'`` or ``'right'``, only one hemisphere will be returned, and the return will again be only points, polygons, and normals.

Tab interface
~~~~~~~~~~~~~
An alternate way to browse the database is using ipython_ and its tab completion feature. If you type the following::

    In [1]: import cortex
    In [2]: cortex.db.

Then press <<TAB>>, a list of subjects will appear. For example::

    In [3]: cortex.db.
     cortex.db.get_anat     cortex.db.get_overlay  cortex.db.get_view     cortex.db.save_view
     cortex.db.get_cache    cortex.db.get_surf     cortex.db.get_xfm      cortex.db.save_xfm
     cortex.db.get_mask     cortex.db.get_surfinfo cortex.db.S1

Selecting the subject **S1** and pressing <<TAB>> gives you additional choices::

    In [4]: cortex.db.S1.
     cortex.db.S1.filestore  cortex.db.S1.surfaces
     cortex.db.S1.subject    cortex.db.S1.transforms

    In [5]: cortex.db.AH.surfaces.
     cortex.db.S1.surfaces.flat     cortex.db.S1.surfaces.pia
     cortex.db.S1.surfaces.inflated cortex.db.S1.surfaces.wm

Selecting "surfaces" gives you a list of all surface types associated with that subject.

Finally, selecting one surface type will give you two new functions: get, and show::
    
    In [6]: left, right = cortex.db.AH.surfaces.inflated.get()
    In [7]: cortex.db.AH.surfaces.fiducial.show()


Adding new surfaces
~~~~~~~~~~~~~~~~~~~
Surface management is implemented through your file manager. To add a new surface to an existing subject, copy the surface file into ``{$FILESTORE}/{$SUBJECT}/surfaces/`` with the format ``{type}_{hemisphere}.{format}``, where ``hemisphere`` is **lh** or **rh**, and format is one of **OFF**, **VTK**, or an **npz** file with keys 'pts' and 'polys'. If you have a python session with pycortex imported already, please reload the session. The new surfaces should be accessible via the given interfaces immediately.

In order to adequately utilize all the functions in pycortex, please add the **fiducial**, **inflated**, and **flat** geometries for both hemispheres. Again, make sure that all the surface types for a given subject and hemisphere have the same number of vertices, otherwise unexpected things may happen!



Transforms
----------

Transformations in pycortex are stored as **affine** matrices encoded in magnet isocenter space, as defined in the Nifti_ headers.

Each transform is stored in its own subdirectory containing two files: ``matrices.xfm``, and ``reference.nii.gz``. Masks are also stored in the transforms directory.

Transforms are saved as JSON-encoded text files. They have the format ``{subject}_{transform}.xfm``. There are four fields in this JSON structure: ``subject``, ``epifile``, ``coord``, ``magnet``. ``epifile`` gives the filename of the functional volume (EPI) that served as the reference for this transform. ``coord`` stores the transform from fiducial to coordinate space (for fast index lookups). ``magnet`` stores the transform from the fiducial to the magnet space, as defined in the return of ``nibabel.get_affine()``.

Reference volumes are typically in Nifti_ format (*.nii), but can be any format that nibabel_ understands. These are stored to ensure that we know what the reference for any transform was. This makes it possible to visually verify and tweak alignments as well as keep a static store of images for future coregistrations.

.. _nibabel: http://nipy.sourceforge.net/nibabel/
.. _Nifti: http://nifti.nimh.nih.gov/nifti-1/


Accessing transforms
^^^^^^^^^^^^^^^^^^^^
Similar to the surfaces, transforms can be access through two methods: direct command access, and the tab interface.

Command access looks like this::

    import cortex
    xfm = cortex.db.get_xfm('AH', 'AH_huth', xfmtype='coord')

Tab complete looks like this::

    In [1]: import cortex
    In [2]: cortex.db.S1.transforms
    Out[2]: Transforms: [fullhead,retinotopy]

    In [3]: cortex.db.S1.transforms['fullhead'].coord.xfm
    Out[3]: 
     [[-0.44486981846094426,
       -0.0021363672818559996,
       -0.03721181986487324,
       46.62686084588364],
      [0.005235315303737166,
       -0.44485768384714863,
       -0.03704886912935894,
       60.165881316857195],
      [-0.02001550497747565,
       -0.020260819840215893,
       0.24044994416882276,
       12.698317611104553],
      [0.0, 0.0, 0.0, 1.0]]


Adding new transforms
^^^^^^^^^^^^^^^^^^^^^
Transforms from anatomical space to functional space are notoriously tricky. Automated algorithms generally give results optimized for various global energy metrics, but do not attempt to target the alignments for your ROIs. It is highly recommended that you use the included aligner to make your affine transforms. To add a transform, either directly create a transform json in ``{$FILESTORE}/transforms/``, or use this command::

    import cortex
    cortex.db.load_xfm(subject, xfmname, xfm, xfmtype='magnet', reference='path_to_functional.nii')

.. _database-masks:

Masks
^^^^^
One of the fundamental reasons for carefully aligning surfaces is to allow the creation and use of cortical masks. This limits the number of voxels you need to model. Traditionally, these masks are created by selecting the set of nearest neighbor voxels for each vertex on the transformed surface. Unfortunately, pycortex's advanced per-pixel mapping precludes the use of this simple mask, since faces could potentially intersect with voxel corners which are not in this simple mask. Thus, the default masks in pycortex use a distance metric to compute mask membership.

Masks were added into pycortex in May 2013, due to previous issues with masked data and the addition of the per-pixel mapping. Masked datasets are further discussed in the datasets page.

Retrieving a mask
"""""""""""""""""
A mask is specified by three variables: **subject**, **transform**, and **mask type**. pycortex defines two named masks for each transform by default. These are the ``'thick'`` and the ``'thin'`` masks. They correspond to a distance of 8 mm and 2 mm, respectively, from any given cortical vertex. Additionally, masks corresponding to known mapper types (such as ``'nearest'`` and ``'trilinear'``) are available. If the subject has both pial and white matter surfaces, all voxels of exactly the cortical thickness distance from each vertex are selected from the fiducial surface. To retrieve the thick mask for S1 using the fullhead transform::

    import cortex
    mask = cortex.db.get_mask('S1', 'fullhead', 'thick')

The first time you load a mask, it will be generated and stored inside the folder for the associated transform.

Loading a mask
""""""""""""""
If you use a custom mask for any reason, it is highly recommended that you load it into the database for future reference. It will allow more seamless integration with ``Datasets``, and will prevent it from being lost. To add a custom mask to the database::

    import cortex
    cortex.db.load_mask(subject, xfmname, masktype, mask)


Surface info
------------

The filestore also manages several important quantifications about the surfaces. These include Tissot's Indicatrix and the flatmap surface distortion. There are stored in the ``/surface_info`` directory.


Views
-----

It is often useful to be able to store, recall, and share specific perspectives onto a 3D model of the brain. The filestore stores these "views" as JSON files containing parameters such as altitude, radius, target, and azimuth. After opening a webgl viewer and manipulating the brain using the browser GUI, a view can be stored by calling::

    viewer = cortex.webgl.show(volume)
    viewer.save_view(subject, name)

Where, ``'subject'`` is the subject identifier and ``'name'`` is a unique name for the stored view. A previously saved view can be applied to a webgl viewer using::

    viewer.get_view(viewer, subject, name)


``overlays.svg``
----------------

Overlays are stored as SVG_'s. This is where surface ROIs are defined. Since these surface ROIs are invariant to transform, only one ROI map is needed for each subject. These SVGs are automatically created for a subject if you call ``cortex.add_roi``. ROI overlays are created and edited in Inkscape_. For more information, see :module:`svgroi.py`.


``rois.svg``
------------


Example subject database entry
------------------------------

Here is an example entry into the filestore...

.. code-block:: shell

    filestore/db
    └── S1
        ├── anatomicals
        │   └── raw.nii.gz
        ├── cache
        │   ├── flatmask_1024.npz
        │   ├── flatpixel_fullhead_1024_nearest_l32.npz
        │   ├── flatverts_1024.npz
        │   └── fullhead_linenn.npz
        ├── overlays.svg
        ├── rois.svg
        ├── surface-info
        │   ├── distortion[dist_type=areal].npz
        │   └── distortion[dist_type=metric].npz
        ├── surfaces
        │   ├── flat_lh.gii
        │   ├── flat_rh.gii
        │   ├── inflated_lh.gii
        │   ├── inflated_rh.gii
        │   ├── pia_lh.gii
        │   ├── pia_rh.gii
        │   ├── wm_lh.gii
        │   └── wm_rh.gii
        ├── transforms
        │   ├── fullhead
        │   │   ├── matrices.xfm
        │   │   └── reference.nii.gz
        │   └── retinotopy
        │       ├── matrices.xfm
        │       └── reference.nii.gz
        └── views


.. _OpenCTM: http://openctm.sourceforge.net/
.. _SVG: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _marching cubes: http://en.wikipedia.org/wiki/Marching_cubes
.. _Caret: http://brainvis.wustl.edu/wiki/index.php/Main_Page
.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu/
.. _ipython: http://ipython.org/
.. _SVG: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _Inkscape: http://inkscape.org/
