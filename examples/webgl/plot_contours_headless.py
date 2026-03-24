"""
===============================================
Plot parcellation contours on 3D brain (headless)
===============================================

The WebGL viewer supports contour rendering of parcellation borders on
the 3D cortical surface. When multiple vertex datasets are loaded as a
``cortex.Dataset``, you can overlay one dataset's contour borders on top
of another.

``cortex.export.save_3d_views`` accepts a ``contour_overlay`` parameter
that names the dataset whose borders should be drawn, and a ``contour_mode``
that controls how the contours are rendered:

- 0: off
- 1: contours only (borders on curvature)
- 2: contours + fill (data with solid-colour borders)
- 3: colored contours only (borders coloured by the overlay's colormap)
- 4: colored contours + fill (data with colormap-coloured borders)

Prerequisites
-------------
Install Playwright and download the bundled Chromium binary once::

    pip install playwright
    playwright install chromium

"""

import os
import tempfile
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import cortex
import cortex.export

np.random.seed(1234)

subject = "S1"
n_verts = cortex.db.get_surf(subject, "fiducial", merge=True)[0].shape[0]

###############################################################################
# Create a random parcellation
# ----------------------------
# Grow 30 random seed vertices across the mesh using breadth-first search.

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

activation = np.random.randn(n_verts)

###############################################################################
# Create a Dataset with both activation and parcellation
# -------------------------------------------------------

ds = cortex.Dataset(
    activation=cortex.Vertex(activation, subject, cmap="RdBu_r", vmin=-2, vmax=2),
    parcellation=cortex.Vertex(
        parcellation, subject, cmap="Set1", vmin=0, vmax=n_parcels
    ),
)

###############################################################################
# Render with colored parcellation contour overlay
# --------------------------------------------------
# Use ``save_3d_views`` with ``contour_overlay="parcellation"`` to draw
# the parcellation borders on top of the activation data. ``contour_mode=4``
# uses the parcellation's own colormap to colour the border lines.

base_name = os.path.join(tempfile.mkdtemp(), "contour")

fnames = cortex.export.save_3d_views(
    ds,
    base_name=base_name,
    list_angles=["left"],
    list_surfaces=["inflated"],
    viewer_params=dict(labels_visible=[], overlays_visible=[]),
    size=(1920, 1080),
    trim=True,
    headless=True,
    contour_overlay="parcellation",
    contour_mode=4,  # colored contours + fill
)

for fname in fnames:
    img = plt.imread(fname)
    aspect = img.shape[0] / img.shape[1]
    fig, ax = plt.subplots(figsize=(10, 10 * aspect))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Activation + colored parcellation contours (inflated, left)",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
    plt.show()
