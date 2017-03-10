"""
==================
Get ROI Voxel Mask
==================

Select all of the voxels within a named ROI and then plot them onto a 
flatmap. In order for this to work, you have to have this ROI in your
svg file for this subject.
"""

import cortex
import matlotlib.pyplot as plt

subject = "S1"
xfm = "fullhead"
roi = "IPS"

# Get the map of which voxels are inside of our ROI
ips_map = cortex.utils.get_roi_mask(subject, xfm, roi)[roi]
# And then threshold
ips_mask = ips_map > 1

# Now we can just plot this onto a flatmap
ips_data = cortex.Volume(ips_mask, subject, xfm, cmap="Blues_r")
cortex.quickshow(ips_data)
plt.show()
