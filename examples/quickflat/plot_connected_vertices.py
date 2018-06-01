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

np.random.seed(1234)

# Create a random pycortex Volume
volume = cortex.Volume.random(subject='S1', xfmname='retinotopy')

# Plot a flatmap with the data projected onto the surface
fig = cortex.quickflat.make_figure(volume)

lines = cortex.quickflat.composite.add_connected_vertices(fig, volume,
            exclude_border_width=None, color=(1.0, 0.5, 0.1, 0.6), linewidth=2,
            alpha=1.0)
plt.show()
