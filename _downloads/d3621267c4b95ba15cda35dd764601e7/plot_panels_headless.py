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
import matplotlib.pyplot as plt

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

###############################################################################
# Render and save using plot_panels
# ---------------------------------
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
plt.show()

###############################################################################
# Save individual views to files
# ------------------------------
# ``cortex.export.save_3d_views`` saves each angle as a separate PNG file.

# sphinx_gallery_multi_image_block = "single"

TARGET_WIDTH = 10  # inches — consistent width for uniform title sizing

base_name = os.path.join(tempfile.mkdtemp(), "fig")
fnames = cortex.export.save_3d_views(
    volume,
    base_name=base_name,
    list_angles=list_angles,
    list_surfaces=list_surfaces,
    viewer_params=dict(labels_visible=[], overlays_visible=["rois"]),
    headless=True,
)

for fname, angle in zip(fnames, list_angles):
    img = plt.imread(fname)
    aspect = img.shape[0] / img.shape[1]
    fig, ax = plt.subplots(figsize=(TARGET_WIDTH, TARGET_WIDTH * aspect))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(angle, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0)
    plt.show()

###############################################################################
# Predefined panel layouts
# ------------------------
# ``cortex.export`` ships with several ready-made panel configurations.
# Every public name matching ``params_*`` is a dict that can be passed
# directly to ``cortex.export.plot_panels``.  The loop below discovers
# them automatically, so this gallery stays up-to-date when new presets
# are added.

# sphinx_gallery_multi_image_block = "single"

predefined = {
    name: getattr(cortex.export, name)
    for name in sorted(dir(cortex.export))
    if name.startswith("params_")
}

for name, params in predefined.items():
    fig = cortex.export.plot_panels(volume, headless=True, **params)
    w, h = fig.get_size_inches()
    # Rescale to a consistent width
    new_w = TARGET_WIDTH
    new_h = h * (TARGET_WIDTH / w) + 0.6
    scale = (h * TARGET_WIDTH / w) / new_h
    for ax in fig.get_axes():
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 * scale, pos.width, pos.height * scale])
    fig.set_size_inches(new_w, new_h)
    fig.suptitle(name, fontsize=14, fontweight="bold", y=1.0 - 0.2 / new_h)
    plt.show()
