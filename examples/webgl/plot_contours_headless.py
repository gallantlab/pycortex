"""
===============================================
Plot parcellation contours on 3D brain (headless)
===============================================

The WebGL viewer supports contour rendering of parcellation borders on
the 3D cortical surface. When multiple vertex datasets are loaded, you
can overlay one dataset's contour borders on top of another.

This example uses the headless viewer to render a left lateral inflated
view of activation data with parcellation contour borders overlaid.

The contour controls available via ``handle._set_view()``:

- ``surface.{subject}.contours.mode``:
    0=off, 1=contours only, 2=contours+fill,
    3=colored contours, 4=colored+fill
- ``surface.{subject}.contours.threshold``: edge sensitivity (0.001–0.5)
- ``surface.{subject}.contours.overlay``: dataset name or "none"

Prerequisites
-------------
Install Playwright and download the bundled Chromium binary once::

    pip install playwright
    playwright install chromium

"""

import time
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
# Use ``headless_viewer`` to open the WebGL viewer in a headless Chromium
# browser, set the view to left lateral inflated, enable the parcellation
# contour overlay with colored borders, and capture a screenshot.

with cortex.export.headless_viewer(ds, viewer_params={}, timeout=30) as handle:
    time.sleep(5)
    handle._set_view(
        **{
            "surface.{subject}.unfold": 0.5,
            "surface.{subject}.contours.overlay": "parcellation",
        }
    )
    time.sleep(5)
    handle._set_view(
        **{
            "surface.{subject}.contours.mode": 4,  # colored + fill
            "camera.azimuth": 160,
            "camera.altitude": 90,
        }
    )
    time.sleep(3)
    handle.getImage("/tmp/contour_webgl_example.png", (1920, 1080))
    time.sleep(3)

img = plt.imread("/tmp/contour_webgl_example.png")
fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / img.shape[1]))
ax.imshow(img)
ax.axis("off")
ax.set_title("Activation + colored parcellation contours (inflated)", fontsize=14)
fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
plt.show()
