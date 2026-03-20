"""
=============================================
Plot 3D Brain Views Headlessly (No Browser)
=============================================

``cortex.export.save_3d_views`` and ``cortex.export.plot_panels`` can render
and save multiple 3D screenshots of brain data without any manual browser
interaction by passing ``headless=True``.

Under the hood this launches a headless Chromium browser via Playwright, which
connects to the pycortex webviewer and renders the WebGL scene using software
rasterisation. You can use these functions to display and save 3D views of
brain data in scripts and notebooks.

Prerequisites
-------------
Install Playwright and download the bundled Chromium binary once::

    pip install playwright
    playwright install chromium

"""

import os
import tempfile

import numpy as np

import cortex
import cortex.export

np.random.seed(42)

volume = cortex.Volume.random(subject="S1", xfmname="fullhead")

# Choose which angles and surface states to render
# Each entry in ``list_angles`` is paired with the corresponding entry in
# ``list_surfaces``.  Both lists must have the same length.
list_angles = [
    "lateral_pivot",
    "medial_pivot",
    "left",
    "right",
]
list_surfaces = ["inflated"] * len(list_angles)

# Render and save using plot_panels
# Build a list of panels (one panel per angle/surface) and render them into
# a single figure with `cortex.export.plot_panels`.  This uses the same
# headless renderer as `cortex.export.save_3d_views`.
panels: list[cortex.export.PanelParams] = []
n = len(list_angles)
for i, (angle, surface) in enumerate(zip(list_angles, list_surfaces)):
    panels.append(
        {
            "extent": (i / n, 0.0, 1.0 / n, 1.0),
            "view": cortex.export.PanelView(angle=angle, surface=surface),
        }
    )

fig = cortex.export.plot_panels(
    volume,
    panels=panels,
    figsize=(2 * n, 2),
    windowsize=(1024 * 2, 768 * 2),
    viewer_params=dict(labels_visible=[], overlays_visible=["rois"]),
    headless=True,
)

# Save these views to files.
base_name = os.path.join(tempfile.mkdtemp(), 'fig')
fnames = cortex.export.save_3d_views(
    volume,
    base_name="base_name",
    list_angles=list_angles,
    list_surfaces=list_surfaces,
    viewer_params=dict(labels_visible=[], overlays_visible=["rois"]),
    headless=True,
)