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

For a slightly flashier way to view the database immediately::

    surfs.AH.surfaces.fiducial.show()

Surfaces
--------
pycortex fundamentally operates on triangular mesh geometry computed from a subject's anatomy. Surface geometries are usually created from a `marching cubes`_ reconstruction of the segmented cortical sheet. This first reconstruction is known as the fiducial surface. The fiducial surface is inflated and cut along anatomical and functional boundaries, and is morphed by an energy metric to be on a flattened 2D surface.

Unfortunately, pycortex currently has no way of generating or editing these geometries. The recommended software for doing segmentation and flattening is Caret_. Another package which is generally more automated, but tends to fail for some subjects is Freesurfer_. Feel free to explore both options for generating cortical reconstructions.

A surface in pycortex is any VTK file specifying the triangular mesh geometry of a subject. Surfaces generally have three variables associated:

    * **Subject** : a unique identifier for the subject whom this surface belongs,
    * **Type** : the identifier for the type of geometry. These generally fall in three categories: Fiducial, inflated, and flat.
    * **Hemisphere** : which hemisphere the surface belongs to.

The VTK files for a specific subject and hemisphere must have the same number of vertices across all the different types. Without this information, the mapping from fiducial to flatmap is not preserved, and there is no way to display data on the flatmap. Caret_ automatically returns VTK files of this format. pycortex does not check the validity of surfaces, and will break in unexpected ways if the number of vertices do not match! It is your job to make sure that all surfaces are valid.

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
    pts, polys, norms = surfs.getVTK('AH', 'fiducial', merge=True)

This returns the positions, polygons, and normals (if found in the vtk file) of the given subject and surface type. Hemisphere defaults to "both", and since merge is true, they are vertically stacked **left, then right**. The polygon indices are shifted up for the right hemisphere to make a single unified geometry.

With merge=False, the return looks different::

    left, right = surfs.getVTK('AH', 'fiducial', merge=False)
    lpts, lpolys, lnorms = left
    rpts, rpolys, rnorms = right

If you only specify hemisphere="left" or "right", only one hemisphere will be returned, and the return will again be only points, polygons, and normals.

Tab interface
"""""""""""""
An alternate way to browse the database is using ipython_ and its tab completion feature. If you type the following::

    In [1]: from cortex import surfs
    In [2]: surfs.

Then press tab, a list of subjects will appear. For example::

    In [2]: surfs.
    surfs.AH         surfs.getFiles   surfs.JG1        surfs.MO         surfs.TC
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
Adding new surfaces is easy. Simply copy your VTK file into the directory ``{$FILESTORE}/surfaces/`` with a filename format of ``{subject}_{type}_{hemi}.vtk`` where hemi is lh or rh. If you have an anatomical file associated with a subject, also copy it into that directory with format ``{subject}_anatomical_both.{suffix}``. If you have a python session with pycortex imported already, please reload the session. The new surfaces should be accessible via the given interfaces immediately.

In order to adequately utilize all the functions in pycortex, please add the fiducial, inflated, and flat geometries for both hemispheres. Again, make sure that all the surface types for a given subject and hemisphere have the same number of vertices, otherwise unexpected things may happen!

Transforms
----------
Functional data, usually collected by an epi sequence, typically does not have the same scan parameters as the anatomical MPRAGE scan used to generate the surfaces. Additionally, fMRI sequences which are usually optimized for T2* have drastically different and larger distortions than a typical T1 anatomical sequence. While automatic algorithms exist to align these two scan types, they will sometimes fail spectacularly, especially if a partial volume slice prescription is necessary.

pycortex includes a tool based on mayavi_ to do manual **affine** alignments. Please see the :module:`align` module for more information. Alternatively, if an automatic algorithm works well enough, you can also commit your own transform to the database. Transforms in pycortex always go from **fiducial to functional** space. They have four variables associated:

    * **Subject** : name of the subject, must match the surfaces used to create the transform
    * **Name** : A unique identifier for this transform
    * **type** : The type of transform -- from fiducial to functional **magnet** space, or fiducial to **coord** inate space
    * **epifile** : the filename of the functional data that the fiducial is aligned to

Transforms always store the epifile in order to allow visual validation of alignment using the :module:`align` module.

.. _mayavi: http://docs.enthought.com/mayavi/mayavi/

Accessing transforms
^^^^^^^^^^^^^^^^^^^^
Similar to the surfaces, transforms can be access through two methods: direct command access, and the tab interface.

Command access looks like this::

    from cortex import surfs
    xfm, epifile = surfs.getXfm("AH", "AH_huth")

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
    surfs.loadXfm(subject, xfmname, xfm, xfmtype='magnet', epifile='path_to_functional.nii')

Database details
----------------
The "database" is stored in the filestore defined in defaults.json. Within the filestore, there are a set of directories:

    * ctmcache: CTM files for use with the webgl viewer
    * flatcache: Pickled flatmap indices for use with quickflat
    * overlays: SVG-based ROI overlays
    * references: EPI reference images for transforms
    * surfaces: VTK files for surface geometries
    * transforms: JSON-encoded files for affine transforms

The ctmcache holds the sequence of files necessary for the webgl viewer. OpenCTM_ is a geometry specification that allows very small files to reduce bandwidth. Files are stored with the format ``{subject}_{transform}_[{types}]_{compression}_{level}.{suffix}``. Each subject and transform is associated with a triplet of files called a "ctmpack". Each ctmpack contains a json file specifying the limits of the data, a ctm file consisting of concatenated left and right hemispheres, and an SVG consisting of the roi's with the data layers deleted. There is a unique ctmpack for each subject, transform, and set of included inflations. Raw CTMs are generated for view.webshow, whereas MG2 CTM's are generated for static WebGL views. These files are considered disposable, and are generated on demand.

The flatcache holds the voxel indices for quickly generating a flatmap. They have the format ``{subject}_{transform}_{height}_{date}.pkl``. A different flatcache must be generated for each datamap height. These files are also disposable and are generated on demand. This cache allows quickflat to satisfy its namesake.

Overlays are stored as SVG_'s. This is where surface ROIs are defined. Since these surface ROIs are invariant to transform, only one ROI map is needed for each subject. These SVGs are automatically created for a subject if you call ``cortex.add_roi``. ROI overlays are created and edited in Inkscape_. For more information, see :module:`svgroi.py`.

References contain the functional scans that are paired with a transform. They are typically in Nifti_ format (*.nii), but can be any format that is understood by nibabel_. These are stored to ensure that we know what the reference for any transform was. This makes it possible to visually verify and tweak alignments, as well as keep a static store of images for future coregistrations.

Surfaces are stored in the format previously discussed.

Transforms are saved as json-encoded text files. They have the format ``{subject}_{transform}.xfm``. There are four fields in this JSON structure: ``subject``, ``epifile``, ``coord``, ``magnet``. ``epifile`` gives the name of the epi file that served as the reference for this transform. ``coord`` stores the transform from fiducial to coordinate space (for fast index lookups). ``magnet`` stores the transform from the fiducial to the magnet space, as defined in the return of ``nibabel.get_affine()``.

.. _OpenCTM: http://openctm.sourceforge.net/
.. _SVG: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _Inkscape: http://inkscape.org/
.. _Nifti: http://nifti.nimh.nih.gov/nifti-1/
.. _nibabel: http://nipy.sourceforge.net/nibabel/