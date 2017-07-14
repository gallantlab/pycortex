"""
===================================
Transform from Subject to MNI Space
===================================

Pycortex has built-in functionality for linearly transforming data to and from
standard atlas spaces (e.g. MNI-152). This functionality is built on top of FSL.

This example shows how to create a transform from some subject functional space
to MNI space, and how to apply that transform to a dataset.

"""

import cortex

# First let's do this "manually", using cortex.mni
from cortex import mni

import numpy as np
np.random.seed(1234)

# This transform is gonna be from one specific functional space for a subject
# which is defined by the transform (xfm)
s1_to_mni = mni.compute_mni_transform(subject='S1', xfm='fullhead')
# s1_to_mni is a 4x4 array describing the transformation in homogeneous corods

# Transform data from subject to MNI space
# first we will create a dataset to transform
data = cortex.Volume.random('S1', 'fullhead')

# then transform it!
mni_data = mni.transform_to_mni(data, s1_to_mni)
# mni_data is a nibabel Nifti1Image

mni_data_vol = mni_data.get_data() # the actual array, shape=(182,218,182)

# That was the manual method. pycortex can also cache these transforms for you
# if you get them using the pycortex database
s1_to_mni_db = cortex.db.get_mnixfm('S1', 'fullhead')
# this is the same as s1_to_mni, but will return instantly on subsequent calls