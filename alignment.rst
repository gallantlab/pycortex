Alignments
==========
Aligning functional data, or finding where the brain is.

The subject's brain is most likely not going to be in the same place and orientation between experiments, so we need to account for that.
To correctly visualise data, the brain surface in the functional images need to be lined up with the surface mesh made from the high-resolution anatomical scans.
This alignment is a rigid body transform, i.e. 6 degrees of freedom in translation and rotation, but no scaling, skewing, or non-linear warping.

Pycortex can automatically try to align the brain, and there is also a manual mode.
To get started, you need a reference image from the functional run as a nifti file. In most cases, this would be the temporal mean image. (You can also use something else like the first image, or whatever.)
Let's say the subject is ``S1``, this transform is ``example-transform``, and the reference image is ``ref-image.nii.gz``.

Automatic Alignment
-------------------
To have pycortex automagically align the brain, simply call
::
	cortex.align.automatic('S1', 'example-transform', './ref-image.nii.gz')

And the alignment should be done!
If you look in the pycortex store in ``S1/transforms/example-transform``, you will find the following files:
	* ``matrices.xfm``, which stores the transformation parameters
	* ``reference.nii.gz``, the reference image you used

Manual Alignment
----------------
**NOTE**: As of right now, the aligner only works on 14.04. Ubuntu 16.04 changed things up

Unfortunately, the automatic alignment only gets you like 95% of the way to a good alignment. To do the final 5%, you need to manually fix it up.
Pycortex has a built-in manual aligner; to start it, call
::
	cortex.align.manual('S1', 'example-transform')
Not: if you are fixing a transform you had previous used for things, you will need to delete the mask files in the transform's folder.

You will see a window like this pop up:
.. image:: snapshot1.png

There's weird gray blobs - click anywhere to get rid of them.
.. image:: snapshot1.png

Here you see 4 different views, showing the saggital, coronal, and transverse slices, and also the three slices in 3D.
The background image is the reference image, and the mesh that you see is the surface that you will be aligning.
You'll be moving the mesh until it's aligned as much as possible with the reference.

To make things easier to see, the aligner offers different color options.

Changing the views
~~~~~~~~~~~~~~~~~~

You can change the color scale for the images with the color map option:
.. image:: colormap.png

Here, we've set it to the red-blue color map.
.. image:: snapshot4.png

``Fliplut`` can be used to reverse the color map.
.. image:: flipcolor.png

You can also use the ``contrast`` and ``brightness`` sliders to adjust the colors.
.. image:: contrast.png

The ``Outline color`` and ``Outline rep`` can be used to change the surface color, and the surface from a mesh (the default), to points only, to a solid surface.
Also, the sliders can be used to chane line and point weights.
Here, we changed it to a green points only representation, with smaller points.
.. image:: surface.png

You will notice two black lines in each view. You can click anywhere in a view to select a different voxel.
Selecting another voxel will update all the other views to show the slices that particular voxel belongs to.
.. image:: lines1.png
.. image:: snapshot13.png

Use these views to change the slices of the brain that you're looking at, to line things up.

Manually aligning the brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On each view, there is a ball surrounded by a ring. These can be used to adjust the brain using the mouse.
Click and drag the center ball to translate in each view, and use the ball on the ring to rotate and scale.
It will take a few seconds for the aligner to update the mesh position.
.. image::adjring.png
**Note**: you should not use the ring to make adjustments. There is no way to fix the scaling, and the ring will screw the scaling up.

You can also use the keyboard to make adjustments.
Holding down the shift key allows you to make fine adjustments.
The aligner will apply the transformation in whatever view currently under your mouse cursor.
.. image::key-controls.png
**Note**: you shouldn't touch the keys outlined in red.

To save the alignment, just click the ``Save Transform`` button and close the window.
.. image::save.png

Tips for aligning the brain
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The really deep sulci work great as landmarks to align stuff up.
* Changing the color map, brightness, and contrast really helps highlight the sulci.
* To check how well the brain is aligned, make a flatmap out of the reference image using the transformation. A good alignment results in a smooth color gradient across the brain; bad ones will have a lot of voxels that are starkly different from their neighbours.