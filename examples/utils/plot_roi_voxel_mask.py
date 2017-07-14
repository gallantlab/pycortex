"""
==================
Get ROI Voxel Mask
==================

Select all of the voxels within a named ROI and then plot them onto a
flatmap. In order for this to work, you have to have this ROI in your
svg file for this subject.
"""

# import cortex
# import matplotlib.pyplot as plt

# subject = "S1"
# xfm = "fullhead"
# roi = "EBA"

# # Get the map of which voxels are inside of our ROI
# eba_map = cortex.utils.get_roi_mask(subject, xfm, roi)[roi]
# # And then threshold
# eba_mask = eba_map > 2

# # Now we can just plot this onto a flatmap
# roi_data = cortex.Volume(eba_mask, subject, xfm, cmap="Blues_r")
# cortex.quickshow(roi_data)
# plt.show()
