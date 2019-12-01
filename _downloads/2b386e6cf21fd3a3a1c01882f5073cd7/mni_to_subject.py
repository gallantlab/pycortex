"""
===================================
Transform from MNI to Subject Space
===================================

Pycortex has built-in functionality for linearly transforming data to and from
standard atlas spaces (e.g. MNI-152). This functionality is built on top of FSL.

This example shows how to create a transform from some subject functional space
to MNI space (the same as in subject_to_mni.py), and how to use that to put data
into subject space from MNI space.

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

# Transform data from MNI to subject space
# first we will create a dataset to transform
# this uses the implicitly created "identity" transform, which is used for data
# in the native anatomical space (i.e. same dims as the base anatomical image,
# and in the same space as the surface)
data = cortex.Volume.random('MNI', 'identity')

# then transform it into the space defined by the 'fullhead' transform for 'S1'
subject_data = mni.transform_mni_to_subject('S1', 'fullhead', 
                                            data.data, s1_to_mni)
# subject_data is a nibabel Nifti1Image

subject_data_vol = mni_data.get_data() # the actual array, shape=(100,100,31)

# That was the manual method. pycortex can also cache these transforms for you
# if you get them using the pycortex database
s1_to_mni_db = cortex.db.get_mnixfm('S1', 'fullhead')
# this is the same as s1_to_mni, but will return instantly on subsequent calls