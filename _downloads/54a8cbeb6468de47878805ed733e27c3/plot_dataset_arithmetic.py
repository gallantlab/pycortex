"""
==================
Dataset Arithmetic
==================

This plots example volume data onto an example subject, S1, onto a flatmap
using quickflat. In order for this to run, you have to have a flatmap for
this subject in the pycortex filestore.

Once you have created a cortex.Volume object, you can manipulate it with
normal arithmetic operators like +, -, *, /, and **
"""

import cortex
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

subject = 'S1'
xfm = 'fullhead'

# Creating a random dataset that is the shape for this transform with one
# entry for each voxel
test_data = np.random.randn(31, 100, 100)

# This creates a Volume object for our test dataset for the given subject
# and transform
vol_data = cortex.Volume(test_data, subject, xfm, vmin=-2, vmax=2)
cortex.quickshow(vol_data)
plt.show()

# Now you can do arithmetic with the Volume
vol_plus = vol_data + 1
cortex.quickshow(vol_plus)
plt.show()

# You can also do multiplication
vol_mult = vol_data * 4
cortex.quickshow(vol_mult)
plt.show()
