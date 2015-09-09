Aligner
=======
This module contains functions for manual and automatic alignment of brain images and cortical surfaces. For each transform, it saves a transform matrix, which maps pixels to voxels.

The automatic() function does epi-to-anat registration using FLIRT with BBR, then inverts this with Transform.from_fsl()
