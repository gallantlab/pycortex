"""
====================
Get ROI Index Volume
====================

Create an index volume (similar to the aseg masks in freesurfer) with a different 
integer index for each ROI. ROIs in the left hemisphere will have negative values,
ROIs in the right hemisphere will have positive values.

"""

import cortex
import numpy as np
import matplotlib.pyplot as plt

subject = "S1"
xfm = "fullhead"

# Get the map of which voxels are inside of our ROI
index_volume, index_keys = cortex.utils.get_roi_masks(subject, xfm, 
                               roi_list=None, # Default (None) gives all available ROIs in overlays.svg
                               gm_sampler='cortical-conservative', # Select only voxels mostly within cortex
                               split_lr=True, # Separate left/right ROIs (this occurs anyway with index volumes)
                               threshold=0.9, # convert probability values to boolean mask for each ROI
                               return_dict=False # return index volume, not dict of masks
                               )

lim = np.max(np.abs(index_volume))
# Plot the mask for one ROI onto a flatmap
roi_data = cortex.Volume(index_volume, subject, xfm, 
                         vmin=-lim, # This is a probability mask, so only
                         vmax=lim, # so scale btw zero and one
                         cmap="RdBu_r", # Shades of blue for L hem, red for R hem ROIs
                         )

cortex.quickflat.make_figure(roi_data,
                             thick=1, # select a single depth (btw white matter & pia)
                             sampler='nearest', # no interpolation
                             with_curvature=True,
                             with_colorbar=True,
                             )
print("Index keys for which ROI is which in `index_volume`:")
print(index_keys)
plt.show()
