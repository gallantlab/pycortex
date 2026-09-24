"""
===============================
Plot parcellation contour lines
===============================

Parcellation contour lines can be overlaid on top of data to delineate
region boundaries without obscuring the underlying activation map.

This is useful when you want to show, for example, fMRI activation data
with anatomical or functional parcellation borders drawn on top.

The ``with_contours`` parameter accepts a :class:`cortex.Vertex` (or any
Dataview) whose label boundaries will be drawn as contour lines. You can
customise the line color with ``contour_linecolor`` and the line width
with ``contour_linewidth``.
"""

import cortex
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

np.random.seed(1234)

subject = "S1"
n_verts = cortex.db.get_surf(subject, "fiducial", merge=True)[0].shape[0]

###############################################################################
# Create a random parcellation
# ----------------------------
# We generate a parcellation by growing 30 random seed vertices across the
# mesh using breadth-first search. Each seed becomes a parcel.

n_parcels = 30
_, polys = cortex.db.get_surf(subject, "fiducial", merge=True)
neighbors = cortex.utils._get_neighbors_dict(polys)

parcellation = np.zeros(n_verts, dtype=float)
seeds = np.random.choice(n_verts, n_parcels, replace=False)
for i, s in enumerate(seeds, 1):
    parcellation[s] = float(i)

queue = deque(seeds.tolist())
while queue:
    v = queue.popleft()
    for nb in neighbors.get(v, []):
        if nb < n_verts and parcellation[nb] == 0:
            parcellation[nb] = parcellation[v]
            queue.append(nb)

# Create Vertex objects
parc_vertex = cortex.Vertex(parcellation, subject, cmap="Set1", vmin=0, vmax=n_parcels)
activation = cortex.Vertex(
    np.random.randn(n_verts), subject, cmap="RdBu_r", vmin=-2, vmax=2
)

###############################################################################
# Activation data with parcellation contours
# -------------------------------------------
# Pass a Dataview to ``with_contours`` to draw its label boundaries on top
# of the primary data.

fig = cortex.quickshow(
    activation,
    with_contours=parc_vertex,
    with_curvature=True,
    with_rois=False,
    with_colorbar=True,
    height=1024,
)
fig.suptitle("Activation + parcellation contours", fontsize=14)
plt.show()

###############################################################################
# Custom contour color and width
# ------------------------------
# Use ``contour_linecolor`` (RGBA tuple) and ``contour_linewidth`` (pixels)
# to customise the contour appearance.

fig = cortex.quickshow(
    activation,
    with_contours=parc_vertex,
    contour_linecolor=(1, 0, 0, 1),
    contour_linewidth=3,
    with_curvature=True,
    with_rois=False,
    with_colorbar=False,
    height=1024,
)
fig.suptitle("Red thick contours", fontsize=14)
plt.show()

###############################################################################
# Parcellation contours on curvature
# -----------------------------------
# You can also overlay the contours of the parcellation on its own data
# to see both the filled colors and the borders.

fig = cortex.quickshow(
    parc_vertex,
    with_contours=parc_vertex,
    with_curvature=True,
    with_rois=False,
    with_colorbar=False,
    height=1024,
)
fig.suptitle("Parcellation with contour borders", fontsize=14)
plt.show()
