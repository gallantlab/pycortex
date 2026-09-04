Alignments
==========

A functional (EPI) scan and the anatomical scan used to build a subject's
surfaces (see :doc:`segmentation_guide`) are almost never acquired with the
subject in exactly the same position, and EPI sequences (optimized for
T2* contrast) suffer larger and different geometric distortions than the
T1 anatomical sequence surfaces are built from. So even though both scans
are of the same physical brain, their voxel grids don't start out
pointing at the same anatomical locations — the functional data and the
surface are, literally, misregistered. Before any functional value can be
assigned to a point on the cortical surface, pycortex needs an explicit
correction for this: a :term:`transform <xfm / transform>` (see
:doc:`transforms`) that maps one coordinate space onto the other. That
correction is deliberately limited to a **rigid-body** transform (6
degrees of freedom: translation and rotation, no scaling/skewing/warping)
— the two scans are of the same physical object, so allowing more degrees
of freedom would let the alignment step silently absorb what should be
segmentation or acquisition problems instead of just correcting for pose.

Two ways to compute that transform are available: **automatic** alignment
(FreeSurfer's boundary-based registration by default, or FSL) and
**manual** alignment (an interactive GUI, currently built on FreeSurfer's
FreeView). Automatic alignment is fast and should be your first attempt
for any new transform, but purely intensity/boundary-based registration
algorithms optimize a global metric that isn't aware of which regions
matter most for your analysis, and can fail outright on partial-volume
acquisitions or unusual contrast — the tips below put this at "gets you
like 95% of the way." Manual alignment exists as a fallback for the
remaining cases: a person, looking at the actual images, correcting
whatever the automatic step got wrong.

Pycortex can automatically try to align the brain, and there is also a manual mode.
To get started, you need a reference image from the functional run in a nibabel-readable format.
In most cases, this would be the temporal mean image. (You can also use something else like the first image, or whatever.)
Let's say the subject is ``S1``, you are making a transform named ``example-transform``, and the reference image is ``ref-image.nii.gz``.

Automatic Alignment
-------------------

This step creates a new transform folder in your pycortex store, and should be the first step for any alignment.

Call ``cortex.align.automatic`` to align the brain automatically::

	cortex.align.automatic('S1', 'example-transform', './ref-image.nii.gz')

As of pycortex 1.2.8, this uses FreeSurfer's ``mri_coreg`` (for an initial
coarse alignment) followed by ``bbregister`` (boundary-based registration)
by default — you'll see a ``UserWarning`` reminding you of this every time
you call it. If you specifically want the older FSL-based BBR alignment
instead, call ``cortex.align.automatic_fsl`` with the same arguments.

``cortex.align.automatic`` accepts a few more arguments worth knowing about:

* ``init``: how to get the initial, coarse alignment that ``bbregister``
  then refines. Defaults to ``"coreg"`` (FreeSurfer's ``mri_coreg``, best
  in most cases); ``"fsl"`` uses FSL's FLIRT instead, ``"header"`` assumes
  the reference and anatomical are already close (e.g. acquired in the
  same session), and a path to an existing DAT/LTA transform can also be
  passed directly.
* ``epi_mask``: pass ``True`` if the reference wasn't distortion-corrected,
  to mask out areas with spatial distortion during registration.
* ``reference_contrast``: ``"t2"`` (default, for BOLD — gray matter
  brighter than white matter) or ``"t1"`` (white matter brighter than gray
  matter).
* ``intermediate``: a path to a whole-brain image acquired in the same
  session, useful when ``reference`` itself has a small field of view.

When it finishes, ``cortex.align.automatic`` prints a "mincost" quality
score (0 to 1, lower is better; values under ~0.5 indicate a good
registration) — but treat that as a sanity check, not a substitute for
looking at the result yourself in the :ref:`manual aligner <manual-alignment>`.

If you look in the pycortex store in ``S1/transforms/example-transform``, you will find the following files:

* ``matrices.xfm``, which stores the transformation parameters
* ``reference.nii.gz``, the reference image you used

Both ``cortex.align.automatic`` and ``cortex.align.automatic_fsl`` accept a
``noclean`` argument (``bool``, default ``False``). Intermediate files
generated during alignment are written to ``/tmp`` and deleted once
alignment finishes; pass ``noclean=True`` to keep them there instead
(useful for debugging), in which case the function returns the temp
directory's path.


.. _manual-alignment:

Manual Alignment
-----------------

Automatic alignment typically gets you most of the way to a good
alignment, but rarely all the way — the remainder needs a person looking
at the actual images. The current, recommended manual aligner is
``cortex.align.manual``, which opens FreeSurfer's **FreeView** with the
reference image and the subject's white-matter and pial surface contours
overlaid on top of it::

	cortex.align.manual('S1', 'example-transform')

This requires FreeSurfer to be installed with ``freeview`` and
``lta_convert`` on your ``PATH``, and ``$SUBJECTS_DIR`` set correctly in
your environment (pycortex shells out to ``freeview`` directly, using
``$SUBJECTS_DIR`` to find the subject's ``orig.mgz`` and surface files).

Use FreeView's own tools to nudge the registration until the surface
contours hug the reference image's gray/white matter boundary. When
you're done, **save the registration** — FreeView will want to write it
somewhere in a temporary ``fsalign_...`` directory pycortex just created;
save it there under the name pycortex tells you to use (``register.lta``
by default, the ``output_name`` argument) and then close FreeView.
Pycortex then converts that file and saves the result into the database
as a new transform.

A few other arguments:

* ``inspect_only=True`` opens the current alignment for viewing only —
  closing FreeView won't overwrite anything, useful for just checking an
  existing transform.
* ``noclean=True`` keeps pycortex's temporary working directory instead of
  deleting it when FreeView closes (and returns its path) — useful for
  debugging if the save doesn't go as expected.
* ``reference`` only needs to be supplied when creating a transform from
  scratch; to re-open and adjust an existing transform, leave it out and
  the transform's stored reference image will be reused.


Tips for aligning the brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The really deep sulci work great as landmarks to align stuff up.
* To check how well the brain is aligned, make a flatmap out of the
  reference image itself using the new transform. Since the reference
  image is a real EPI volume, adjacent voxels should have similar
  intensities — a good alignment carries that smoothness onto the
  flatmap as a smooth gradient across the cortical surface. A bad
  alignment instead samples the reference at the wrong locations, so
  neighbouring surface vertices end up pulling from unrelated voxels and
  the flatmap looks patchy/speckled instead::

    vol = cortex.Volume('./ref-image.nii.gz', 'S1', 'example-transform')
    cortex.quickshow(vol)

  Smooth gradient across the brain → good alignment. A lot of voxels
  starkly different from their neighbours → revisit the alignment (see
  :ref:`manual alignment <manual-alignment>` below).

  See :ref:`sphx_glr_auto_examples_utils_plot_check_alignment.py` for a
  runnable version of this check (against the bundled ``S1`` subject's
  already-good ``fullhead`` transform, so you can see what "smooth
  gradient" actually looks like before judging one of your own).