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

Unfortunately, pycortex currently has no way of generating or editing these geometries directly. The recommended software for doing segmentation and flattening is Freesurfer_. Another package which is generally more user-friendly is Caret_. pycortex includes some utility functions to interact with Freesurfer_, which are documented in :ref:`database-freesurfer-import` below and in the ``cortex.freesurfer`` module.

A surface in pycortex is any file specifying the triangular mesh geometry of a subject. Surfaces may be stored in any one of **OFF**, **VTK**, or **npz** formats. The highest performance is achieved with **npz** since it is binary and compressed. VTK is also efficient, having a `Cython` module to read files. Inside the filestore, surface names are formatted as ``{type}_{hemisphere}.{format}``. Surfaces generally have three variables associated:

    * **Subject** : a unique subject identifier
    * **Type** : the identifier for the type of geometry, **fiducial**, **inflated**, or **flat**
    * **Hemisphere** : the brain hemisphere of the surface, **lh** or **rh**

The surface files for a specific subject and hemisphere must have the same number of vertices across all the different types. Without this information, the mapping from fiducial to flatmap is not preserved, and there is no way to display data on the flatmap. Freesurfer_ surfaces preserve this relationship, and can be automatically imported into the database. pycortex does not check the validity of surfaces, and will break in unexpected ways if the number of vertices do not match! It is your job to make sure that all surfaces are valid.

In order to plot cortical data for a subject, at least the fiducial and flat geometries must be available for that subject. Surfaces must be stored in VTK v. 1 format (also known as the ASCII format).


Accessing surfaces
~~~~~~~~~~~~~~~~~~
Two methods exist for accessing the surface data once they are committed to the database: direct command access, or via a convenient tab completion interface.

Command access
~~~~~~~~~~~~~~
For the direct command access, there are two call signatures::

    import cortex
    pts, polys = cortex.db.get_surf('S1', 'fiducial', merge=True)

This returns the points and polygons of the given subject and surface type. Hemisphere defaults to "both", and since ``merge`` is true, they are vertically stacked **left, then right**. The polygon indices are shifted up for the right hemisphere to make a single unified geometry.

With ``merge=False``, the return looks different::

    left, right = cortex.db.get_surf('S1', 'fiducial', merge=False)
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

    In [5]: cortex.db.S1.surfaces.
     cortex.db.S1.surfaces.flat     cortex.db.S1.surfaces.pia
     cortex.db.S1.surfaces.inflated cortex.db.S1.surfaces.wm

Selecting "surfaces" gives you a list of all surface types associated with that subject.

Finally, selecting one surface type will give you two new functions: get, and show::
    
    In [6]: left, right = cortex.db.S1.surfaces.inflated.get()
    In [7]: cortex.db.S1.surfaces.fiducial.show()

Subject names that are not valid Python identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Attribute access only works for names that happen to be valid Python
identifiers, so a subject called ``S1-test`` cannot be reached as
``cortex.db.S1-test``. Every level of the tab interface also supports item
access, which works for any name::

    In [8]: cortex.db['S1-test']
    In [9]: cortex.db['S1-test'].surfaces['flat'].get()
    In [10]: cortex.db['S1-test'].transforms['fullhead']['coord'].xfm

You can also test for a subject with ``'S1-test' in cortex.db`` and iterate
over the subject names with ``for subject in cortex.db``.


Adding new surfaces
~~~~~~~~~~~~~~~~~~~
Surface management is implemented through your file manager. To add a new surface to an existing subject, copy the surface file into ``{$FILESTORE}/{$SUBJECT}/surfaces/`` with the format ``{type}_{hemisphere}.{format}``, where ``hemisphere`` is **lh** or **rh**, and format is one of **OFF**, **VTK**, or an **npz** file with keys 'pts' and 'polys'. If you have a python session with pycortex imported already, please reload the session. The new surfaces should be accessible via the given interfaces immediately.

In order to adequately utilize all the functions in pycortex, please add the **fiducial**, **inflated**, and **flat** geometries for both hemispheres. Again, make sure that all the surface types for a given subject and hemisphere have the same number of vertices, otherwise unexpected things may happen!



.. _database-freesurfer-import:

Importing a subject from Freesurfer
-----------------------------------

The usual way to create a subject in the pycortex database is to import one that has
already been segmented with Freesurfer_::

    import cortex
    cortex.freesurfer.import_subj(freesurfer_subject, pycortex_subject=None,
                                  freesurfer_subject_dir=None,
                                  whitematter_surf='smoothwm')

This reads from the Freesurfer subject directory,
``$SUBJECTS_DIR/{freesurfer_subject}/``, and writes into the pycortex filestore entry
for the subject, ``{filestore}/{pycortex_subject}/``. Only the files listed below are
copied over. Note that ``import_subj`` overwrites any pre-existing pycortex subject
of the same name, including all blender cuts, masks and transforms, and deletes all
cached files for that subject.

Anatomical volumes are converted with ``mri_convert``:

======================  =============================  ==========================
Freesurfer file         pycortex file                  Contents
======================  =============================  ==========================
``mri/T1.mgz``          ``anatomicals/raw.nii.gz``     T1-weighted anatomical
``mri/aseg.mgz``        ``anatomicals/aseg.nii.gz``    Automatic segmentation
``mri/wm.mgz``          ``anatomicals/raw_wm.nii.gz``  White matter segmentation
======================  =============================  ==========================

Surfaces are imported for both hemispheres (``lh`` and ``rh``) and are converted with
``mris_convert --to-scanner``, so that they are stored in the same coordinate system
as the anatomical volumes rather than in the Freesurfer TKR coordinate system (whose
center is set to FOV/2). As a consequence, the imported surfaces will look misaligned
with the anatomical volumes if you load them in ``freeview``, which expects TKR
coordinates. This is expected: the surfaces in the pycortex database are only meant to
be used by pycortex.

==========================  =============================  ======================
Freesurfer file             pycortex file                  Contents
==========================  =============================  ======================
``surf/?h.smoothwm``        ``surfaces/wm_?h.gii``         White matter surface
``surf/?h.pial``            ``surfaces/pia_?h.gii``        Pial surface
``surf/?h.inflated``        ``surfaces/inflated_?h.gii``   Inflated surface
==========================  =============================  ======================

The surface imported as ``wm`` is whichever surface is named by the
``whitematter_surf`` argument, so the first row above is really
``surf/?h.{whitematter_surf}``. It defaults to ``smoothwm``, but that surface is
smoothed and may not be appropriate for every use; ``white`` is a good alternative.

Surface info files hold one value per vertex. Both hemispheres are stored together in
a single ``.npz`` file, under the keys ``left`` and ``right``. Note that the values are
stored **negated** with respect to the Freesurfer values.

==========================  ================================  ==================
Freesurfer files            pycortex file                     Contents
==========================  ================================  ==================
``surf/?h.sulc``            ``surface-info/sulcaldepth.npz``  Sulcal depth
``surf/?h.thickness``       ``surface-info/thickness.npz``    Cortical thickness
``surf/?h.curv``            ``surface-info/curvature.npz``    Curvature
==========================  ================================  ==================

``import_subj`` also (re-)generates the fiducial surfaces, halfway between the white
matter and the pial surfaces, which are used for cutting and flattening. Those are
written back into the *Freesurfer* subject directory as ``surf/?h.fiducial``, not into
the pycortex filestore.

Nothing else is imported. In particular, flat surfaces are not imported by
``import_subj``; once a surface has been cut and flattened in Freesurfer, import it
separately with::

    cortex.freesurfer.import_flat(fs_subject, patch, hemis=['lh', 'rh'],
                                  cx_subject=None)

which writes ``surfaces/flat_lh.gii`` and ``surfaces/flat_rh.gii``. Labels and
annotations are not imported either; see ``cortex.freesurfer.get_label``.


Transforms
----------

Transformations in pycortex are stored as **affine** matrices encoded in magnet isocenter space, as defined in the Nifti_ headers.

Each transform is stored in its own subdirectory containing two files: ``matrices.xfm``, and ``reference.nii.gz``. Masks are also stored in the transforms directory.

Transforms are saved as JSON-encoded text files. They have the format ``{subject}_{transform}.xfm``. There are four fields in this JSON structure: ``subject``, ``epifile``, ``coord``, ``magnet``. ``epifile`` gives the filename of the functional volume (EPI) that served as the reference for this transform. ``coord`` stores the transform from fiducial to coordinate space (for fast index lookups). ``magnet`` stores the transform from the fiducial to the magnet space, as defined in the return of ``nibabel.affine``.

Reference volumes are typically in Nifti_ format (*.nii), but can be any format that nibabel_ understands. These are stored to ensure that we know what the reference for any transform was. This makes it possible to visually verify and tweak alignments as well as keep a static store of images for future coregistrations.

.. _nibabel: http://nipy.sourceforge.net/nibabel/
.. _Nifti: http://nifti.nimh.nih.gov/nifti-1/


Accessing transforms
~~~~~~~~~~~~~~~~~~~~
Similar to the surfaces, transforms can be access through two methods: direct command access, and the tab interface.

Command access looks like this::

    import cortex
    xfm = cortex.db.get_xfm('S1', 'fullhead', xfmtype='coord')

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
~~~~~~~~~~~~~~~~~~~~~
Transforms from anatomical space to functional space are notoriously tricky. Automated algorithms generally give results optimized for various global energy metrics, but do not attempt to target the alignments for your ROIs. It is highly recommended that you use the included aligner to make your affine transforms. To add a transform, either directly create a transform json in ``{$FILESTORE}/transforms/``, or use this command::

    import cortex
    cortex.db.load_xfm(subject, xfmname, xfm, xfmtype='magnet', reference='path_to_functional.nii')

.. _database-masks:

Masks
~~~~~

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

The filestore also manages several important quantifications about the surfaces. These include Tissot's Indicatrix and the flatmap surface distortion. There are stored in the ``/surface-info`` directory. This is also where the per-vertex curvature, sulcal depth and thickness imported from Freesurfer_ are stored (see :ref:`database-freesurfer-import`). Each file is an ``.npz`` file holding one array per hemisphere, under the keys ``left`` and ``right``, and can be loaded with ``cortex.db.get_surfinfo``.


Views
-----

It is often useful to be able to store, recall, and share specific perspectives onto a 3D model of the brain. The filestore stores these "views" as JSON files containing parameters such as altitude, radius, target, and azimuth. After opening a webgl viewer and manipulating the brain using the browser GUI, a view can be stored by calling::

    viewer = cortex.webgl.show(volume)
    viewer.save_view(subject, name)

Where, ``'subject'`` is the subject identifier and ``'name'`` is a unique name for the stored view. A previously saved view can be applied to a webgl viewer using::

    viewer.get_view(subject, name)

Default views
~~~~~~~~~~~~~

Every subject is offered a standard set of views whether or not anything has been saved for it, so a freshly imported subject already has the usual orientations one click away under **camera > views**:

=========================  ====================================================
View                       Shows
=========================  ====================================================
``dorsal``                 From above, frontal lobe towards the top of the
                           image, the subject's right on the right.
``ventral``                From below, frontal lobe towards the top.
``lateral_left``           From the left, brain upright.
``lateral_right``          From the right, brain upright.
``*_inflated``             The same four angles on the inflated surface.
``flat``                   The flattened surface. Omitted for a subject with
                           no flat surface, which also puts the ``_inflated``
                           views at full inflation rather than half.
=========================  ====================================================

They are built from the same tables ``cortex.export.save_views`` uses for :func:`save_3d_views`, and can be inspected from python::

    from cortex.export.save_views import default_subject_views
    default_subject_views()["dorsal"]

**A view saved in the filestore under one of these names replaces the default.** So if a subject's anatomy wants a different angle, or you prefer a different framing, save your own view under that name and it is used instead — for that subject only, leaving every other default in place::

    viewer.save_view(subject, "dorsal", is_overwrite=True)

The ``flat`` view uses the viewer's standard flatmap preset. It names no camera angle, because a flattened surface ignores one: the controls hold the camera square-on to the flatmap and discard whatever azimuth and altitude they are given (unless the surface's ``allow_tilt`` is on). Leaving them out is what keeps an animation from spinning the brain as it flattens, and from overwriting the folded angle it returns to when it unfolds again. A flat view or keyframe captured in the viewer leaves them out for the same reason.

It carries no zoom of its own either. Clicking it in the browser frames the flatmap the way ``quickflat`` frames the image it writes — the camera looks at the middle of the flatmap from the distance at which the field of view spans it — so that rendered at the pixel size quickflat uses, the frame is the png ``cortex.quickflat.make_png`` writes, same position and same scale. (The perspective camera is no obstacle: a plane square-on to it projects as a uniform scaling, which is all quickflat's mapping of the flat surface onto the bounds of the image amounts to.)

That framing is applied on request, never behind your back. Setting the flat view from python with ``_set_view`` leaves the camera where it is, so :func:`cortex.export.save_3d_views` and anything else driving the viewer render exactly as they always did; ask for it with ``handle.fit_flat_view()``, and ``getImage`` then re-frames it for the image it is about to write, whatever the shape of the window. In the animation panel, tick **match quickflat size** in the render form: the size fields are filled with the subject's quickflat size and flat keyframes are framed for it, including any already laid down. Type another size over it and flat keyframes are framed for that one instead.

A view saved in the filestore under the name ``flat`` replaces all of this, framing included, since a saved view records the camera distance it was saved with.

The camera keeps two targets — the point it orbits and looks at — one for the folded brain and one for the flatmap, and moves between them as the surface unfolds. Views store them separately, as ``camera.target`` (folded) and ``camera.flat_target``, so an animation from a folded pose into the flat view leaves the folded target where it was, and unfolding again returns the brain exactly to its starting place. The flat target starts at the middle of the flatmap, so flattening lands centred without the flat view naming one. A flat view saved before ``camera.flat_target`` existed stores its flat target as ``camera.target``, and is still read that way: a flat view that carries ``camera.target`` but no ``camera.flat_target`` sets the flat target.

Saved views in the browser
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every view stored for the subject(s) a viewer displays is loaded when the viewer starts, and appears as a button under **camera > views** in the browser controls. Clicking one applies it. This works in static viewers made with ``cortex.webgl.make_static`` as well.

The **save view** button in the same menu captures the current view under a name you choose. These stay in the browser rather than being written to the filestore, so that you can experiment freely; retrieve them from python with::

    new_views = viewer.retrieve_new_views()

which returns a dict mapping each name to a dict of view parameters, in the same format as ``viewer._capture_view()``. To keep them, write them into the subject's ``views`` directory::

    viewer.save_new_views()

This returns a dict mapping each name to the file it was written to. Pass ``subject`` to store them under a subject other than the first one displayed, ``names`` to save only some of them, and ``is_overwrite=True`` to replace views already stored under the same name. A view that has been stored is no longer "new": it moves into the **camera > views** menu of the running viewer and stops being returned by ``retrieve_new_views``, so calling ``save_new_views`` twice will not rewrite the same files.

Animations
~~~~~~~~~~

The **create animation** button opens a panel for building an animation out of keyframes. Set the current frame with the slider (frames that already hold a keyframe are marked with a yellow dot), pose the brain, and press **add keyframe**; the values in between are interpolated. **play animation** previews the result at the chosen frame rate, and **render animation** writes one PNG per frame.

Smoothing
^^^^^^^^^

The **smoothing** dropdown sets how the curve through the keyframes is shaped. Each keyframe carries its own setting, which describes both how the animation arrives at it and how it leaves, so the motion between two keyframes depends on the pair at either end. The dropdown always shows the setting of the keyframe under the playhead; when there is no keyframe there it shows the one new keyframes will be given.

Eight options are available:

=========================  ====================================================
Option                     Behaviour
=========================  ====================================================
bezier (smooth)            Default. A cubic Bezier with automatically placed
                           control points. Velocity carries smoothly through
                           the keyframe and the brain never swings past the
                           pose you set.
cubic hermite (smooth)     As above, using the tangents directly rather than
                           control points. Very nearly the same curve.
linear                     Straight lines between keyframes, which is what the
                           panel did before smoothing was added. The motion
                           changes direction abruptly at each keyframe.
bezier in, hold            Arrive smoothly, then freeze on this pose until the
                           next keyframe.
hermite in, hold           As above, arriving along a Hermite tangent.
linear in, hold            Arrive in a straight line, then freeze.
linear in, bezier out      Arrive in a straight line and leave along it, easing
                           out with a Bezier. May swing past the next pose.
linear in, hermite out     As above with a Hermite. May swing past the next
                           pose.
=========================  ====================================================

The two "smooth" options and the three holds stay within the poses you set. The two "linear in" options carry the incoming speed out of the keyframe and so can overshoot, which is useful for a sense of momentum and unhelpful if you need the camera to stop exactly where you put it.

The same eight modes are available when rendering from python, either for a whole animation::

    viewer.make_movie_views(animation, interpolation="Bezier")

or per keyframe, by giving a keyframe its own ``interpolation`` key — which is what the panel does. The browser and ``cortex.webgl.interpolation`` share their arithmetic, so a movie rendered from python matches the preview played in the viewer. The older whole-animation easings ``"linear"``, ``"smoothstep"`` and ``"smootherstep"`` still work, but ease each pair of keyframes separately and cannot be combined with per-keyframe modes.

**render animation** builds the movie in the browser and downloads it as one file, the way **Save image** does — so it lands on the computer running the browser, wherever that browser saves downloads, and the server writes nothing. It works in static viewers made with :func:`cortex.webgl.make_static` too. Choose the format in the render form:

* **PNG frames (.zip)** — one lossless PNG per frame, transparent outside the brain, named after the animation's frame numbers inside a folder named after the movie (``brainmovie/brainmovie_00000.png``, …). These are the frames to use when they must match a flatmap from ``cortex.quickflat.make_png``. A zip holds at most 65,535 frames and 4 GiB; render a shorter range of frames if a movie is larger than that.
* **MP4 video** — H.264, encoded by the browser itself (WebCodecs). It is lossy and has no transparency, so frames are laid on black, as the viewer shows them. Browsers only offer video encoding in a secure context — a viewer opened on ``localhost``, over ``https``, or from a local file — so the option is unavailable when a viewer is reached over plain http from another machine. The largest size depends on the browser's encoder, and a size it cannot encode is refused before rendering starts; an odd width or height gets one extra row or column of black, since H.264 needs even dimensions.


``overlays.svg``
----------------

Overlays are stored as SVG_'s. This is where surface ROIs are defined. Since these surface ROIs are invariant to transform, only one ROI map is needed for each subject. These SVGs are automatically created for a subject if you call ``cortex.add_roi``. ROI overlays are created and edited in Inkscape_. For more information, see ``svgroi.py``.


``rois.svg``
------------


Example subject database entry
------------------------------

Here is an example entry into the filestore...

.. code-block:: shell

    filestore/db
    └── S1
        ├── anatomicals
        │   ├── aseg.nii.gz
        │   ├── raw.nii.gz
        │   └── raw_wm.nii.gz
        ├── cache
        │   ├── flatmask_1024.npz
        │   ├── flatpixel_fullhead_1024_nearest_l32.npz
        │   ├── flatverts_1024.npz
        │   └── fullhead_linenn.npz
        ├── overlays.svg
        ├── rois.svg
        ├── surface-info
        │   ├── curvature.npz
        │   ├── distortion[dist_type=areal].npz
        │   ├── distortion[dist_type=metric].npz
        │   ├── sulcaldepth.npz
        │   └── thickness.npz
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
