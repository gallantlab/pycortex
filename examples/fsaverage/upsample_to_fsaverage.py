"""
===================
Upsample data from a lower resolution fsaverage template to fsaverage for visualization
===================

This example shows how data in a lower resolution fsaverage template 
(e.g., fsaverage5 or fsaverage6) can be upsampled to the high resolution fsaverage 
template for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

import cortex

subject = "fsaverage"

# First we check if the fsaverage template is already in the pycortex filestore. If not,
# we download the template from the web and add it to the filestore.
if subject not in cortex.db.subjects:
    cortex.download_subject(subject)

# Next we create some data on fsaverage5. Each hemisphere has 10242 vertices.
n_vertices_fsaverage5 = 10242
data_fs5 = np.arange(1, n_vertices_fsaverage5 + 1)
# We concatenate the data to itself to create a vector of length 20484, corresponding to
# the two hemispheres together.
data_fs5 = np.concatenate((data_fs5, data_fs5))
# Finally, we upsample the data to fsaverage.
data_fs7 = cortex.freesurfer.upsample_to_fsaverage(data_fs5, "fsaverage5")

# Now that the data is in the fsaverage template, we can visualize it in PyCortex as any
# other vertex dataset.
vtx = cortex.Vertex(data_fs7, subject, vmin=0, vmax=n_vertices_fsaverage5, cmap="turbo")
cortex.quickshow(vtx, with_curvature=False, with_colorbar=False)
plt.show()
