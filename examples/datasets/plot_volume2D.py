"""
===================
Plot 2D Volume Data 
===================

This plots example volume data onto an example subject, S1, onto a flatmap
using quickflat. In order for this to run, you have to have a flatmap for
this subject in the pycortex filestore.

The cortex.Volume2D object is instantiated with two numpy arrays of the same 
size as the scan for this subject and transform. Here, there are two datasets
that have been generated to look like gradients across the brain, but you can 
replace these with any numpy arrays of the correct dimensionality.

As with a 1D Volume, you can change vmin and vmax to threshold, but here
they can be manipulated individually for the two arrays.
"""

import cortex
import numpy as np
import matplotlib.pyplot as plt

subject = "S1"
xfm = "fullhead"

test_data1 = np.arange(31*100*100).reshape((31,100,100), order='C')
test_data2 = np.arange(31*100*100).reshape((31,100,100), order='F')

dv = cortex.Volume2D(test_data1, test_data2, subject, xfm)
cortex.quickshow(dv, with_colorbar=False)
plt.show()

# Altering vmin and vmax for the two datasets
dv = cortex.Volume2D(test_data1, test_data2, subject, xfm, 
                     vmin=np.mean(test_data1), vmax=np.max(test_data1),
                     vmin2=np.min(test_data2), vmax2=np.mean(test_data2))
cortex.quickshow(dv, with_colorbar=False)
plt.show()

