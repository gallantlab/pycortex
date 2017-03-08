"""
================
Plot Volume Data
================

This plots example data onto an example subject, S1

Instead of the random test data, you can replace this with any numpy array
that is the correct dimensionality for this brain and transform

By changing the parameters vmin and vmax, you get thresholded data 

If you have NaN values within your array, those voxels show up transparent
on the brain
"""

import cortex
import numpy as np
import matplotlib.pyplot as plt

subject = 'S1'
xfm = 'fullhead'

test_data = np.random.randn(31,100,100)

dv = cortex.Volume(test_data, subject, xfm)
cortex.quickshow(dv)
plt.show()

# Can also alter the minimum and maximum values shown on the colorbar
dv_thresh = cortex.Volume(test_data, subject, xfm, vmin=-1, vmax=1)
cortex.quickshow(dv_thresh)
plt.show()

# If you have NaN values, those voxels show up transparent on the brain
test_data[10:15,:,:] = np.nan
dv_nan = cortex.Volume(test_data, subject, xfm)
cortex.quickshow(dv)
plt.show()

