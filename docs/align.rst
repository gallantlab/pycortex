Alignments
==========
The ``cortex.align`` module.

Aligning functional data, or finding where the brain is.

To correctly visualise cortical activity, we need to know where cortical surface is in the functional data.
The brain surface in the functional images need to be lined up with the surface mesh made from the high-resolution anatomical scans.
This alignment is a rigid body transform, i.e. 6 degrees of freedom in translation and rotation, but no scaling, skewing, or non-linear warping.

Pycortex can automatically try to align the brain, and there is also a manual mode.
To get started, you need a reference image from the functional run in a nibabel-readable format.
In most cases, this would be the temporal mean image. (You can also use something else like the first image, or whatever.)
Let's say the subject is ``S1``, you are making a transform named ``example-transform``, and the reference image is ``ref-image.nii.gz``.

Automatic Alignment
-------------------

Pycortex can automatically align the brain using FSL.
This step creates a new transform folder in your pycortex store, and should be the first step for any alignment.

To have pycortex automagically align the brain, simply call
::
	cortex.align.automatic('S1', 'example-transform', './ref-image.nii.gz')

And the alignment should be done! This is done using FSL.

You can also use FreeSurfer's boundary-based registration by setting the ``use_fs_bbr`` argument to ``True``.
::
	cortex.align.automatic('S1', 'example-transform', './ref-image.nii.gz',
                           use_fs_bbr=True)

If you look in the pycortex store in ``S1/transforms/example-transform``, you will find the following files:

* ``matrices.xfm``, which stores the transformation parameters
* ``reference.nii.gz``, the reference image you used

There is another argument to ``align.automatic``, ``noclean``, a ``bool`` that defaults to ``true``.
The automatic alignment generates a bunch of intermediate files in ``/tmp``, which are deleted upon completion of the alignment process.
Setting ``noclean`` to ``false`` will keep those files there, which is useful for debugging.

Automatically Tweaking Alignments
---------------------------------
In theory, a pre-existing alignment can be tweaked automatically.
Like the automatic alignment, this is done via FSL.
However, in practice, the search range is too big to be practically useful, and tweaking should be done using manual alignment instead.
::
	cortex.align.autotweak('S1', 'example-transform')

Manual Alignment
----------------

Unfortunately, the automatic alignment only gets you like 95% of the way to a good alignment.
To do the final 5%, you need to manually fix it up.
Pycortex offers a GUI aligner that runs in the browser, built on the WebGL viewer.
``cortex.align.manual`` is an alternative that hands the alignment to FreeSurfer's Freeview.

Aligning in the browser
~~~~~~~~~~~~~~~~~~~~~~~

To start the browser-based aligner for a new transform, pass the reference image
::
	cortex.align.webgl_manual('S1', 'example-transform', reference='./ref-image.nii.gz')

To adjust an existing transform, leave the reference out
::
	cortex.align.webgl_manual('S1', 'example-transform')

A transform you had previously used for things opens the same way: saving deletes the masks cached for it, because they were cut out of the reference volume through the alignment you are replacing.
The page warns about this when it opens, and the save message names the masks it deleted.
Data you had already masked with them has to be masked again from the volumes.
To look at an alignment without saving, pass ``view_only=True``.

The page shows the coronal, axial and sagittal slices of the reference image, and a 3D view of the three slices.
The reference image is drawn on its own voxel grid, so its voxels appear as they are, without resampling, and the pial and white matter surfaces are moved into its space.
In each slice view the surfaces are cut off at the displayed slice, so what you see is their outline on the slice.
You move the surfaces until this outline follows the anatomy in the image.

* In a slice view, a left drag moves the cursor. The cursor sets the slices shown in the other views and is the pivot of rotations. The wheel or ``[`` and ``]`` change the slice, ctrl + wheel zooms, and a middle (or shift + left) drag pans.
* A right drag, the WASD keys or the arrow keys translate the surfaces in the plane of the view under the mouse. A ctrl + right drag or ``q`` and ``e`` rotate them about the cursor, in the plane of the view under the mouse. Holding shift makes the keyboard steps ten times smaller, and ctrl + z undoes. Only rotations and translations are possible; the transform cannot stretch the brain.
* In the 3D view, a left drag rotates, a middle (or shift + left) drag pans, and a right drag or the wheel zooms.

The panel on the right holds the controls.
``display`` chooses what the page shows, with three settings (``m`` steps through them).
``3 ortho + 3D slices`` is the display described above, where the fourth panel holds the three slice planes in space.
``3 ortho + 3D brain`` keeps the slice views as they are and turns the bottom right corner into the viewer, so the mesh can be nudged in the slices while the data on the surface follows.
``data on the surface`` gives that viewer the whole window, framed on the surface the way the WebGL viewer opens on one, where ``unfold`` inflates it and flattens it and ``pivot`` swings its halves apart.
Both of the latter paint the reference data on the surface through the alignment as it currently stands, saved or not, and redraw as you move the mesh, so an alignment can be judged from the pattern the data makes on the cortex before committing it.
Every control stays available in all three.
``image`` sets the colormap, its range (``vmin`` and ``vmax``), ``brightness``, ``contrast`` and ``gamma``, and flips the colormap. The colormap dropdown draws a strip of each colormap beside its name.
``mesh`` sets the color of the surfaces, their ``opacity`` in the 3D view (0 shows only the outlines), which of the two surfaces are shown, the cortical ``depth`` the data is painted at, and the ``unfold`` and ``pivot`` of the data view.
``slices`` selects the slices, and ``steps`` sets the keyboard steps.

The ``transform`` field at the top of the panel holds the name the alignment is saved under, and starts as the transform you opened.
Edit it to save the alignment as a new transform, which leaves the one you opened untouched, along with its masks.
An asterisk on the ``save`` button and in the window title marks an alignment that differs from the one last saved.

To save the alignment, click ``save``.
The transform is stored into the database at once, together with the deletion of any masks cached for it, and the window can then be closed.
The function returns a handle to the running aligner: ``handle.get_xfm()`` returns the current transform as a 4x4 matrix, and ``handle.save()`` saves it.

The initial colormap, color of the surfaces and opacity are set in the ``[webgl_aligner]`` section of the config file.

Tips for aligning the brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Holding down the shift key while using the keyboard controls will let you move the brain in fine-tuned, smaller increments.
* The really deep sulci work great as landmarks to align stuff up.
* Changing the color map, brightness, and contrast really helps highlight the sulci.
* To check how well the brain is aligned, make a flatmap out of the reference image using the transformation. A good alignment results in a smooth color gradient across the brain; bad ones will have a lot of voxels that are starkly different from their neighbours.