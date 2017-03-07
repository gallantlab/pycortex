Alignments
==========
Aligning functional data, or finding where the brain is.

The subject's brain is most likely not going to be in the same place and orientation between runs, so we need to account for that.
To correctly visualise data, the brain surface in the functional images need to be lined up with the surface mesh made from the high-resolution anatomical scans.
This alignment is a rigid body transform, i.e. 6 degrees of freedom in translation and rotation, but no scaling, skewing, or non-linear warping.

Pycortex can automatically try to align the brain, and there is also a manual mode.
To get started, you need a reference image from the functional run. In most cases, this would be the temporal mean image.


Automatic Alignment
-------------------

Manual Alignment
----------------
**NOTE**: As of right now, the aligner only works on 14.04. Ubuntu 16.04 changed things up 