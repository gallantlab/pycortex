"""
==============================================
Plot which vertices are inside the same voxels
==============================================

Show lines connecting vertices on the flatmap that are actually within the same
voxels in a given scan.

Here, we used advanced compositing to be explicit about display options for the
connecting lines.

"""
import cortex
import numpy as np
import matplotlib.pyplot as plt

# Create an empty pycortex Volume
volume = cortex.Volume.empty(subject='S1', xfmname='retinotopy', value=np.nan)

# Plot a flatmap with the data projected onto the surface
fig = cortex.quickflat.make_figure(volume, with_curvature=True, with_colorbar=False)

# Advanced compositing addition of connected vertices.
# Note that this will not currently resize correctly with a figure.
lines = cortex.quickflat.composite.add_connected_vertices(fig, volume,
            exclude_border_width=None, color=(1.0, 0.5, 0.1, 0.6), linewidth=0.75,
            alpha=0.3, recache=True)
plt.show()
