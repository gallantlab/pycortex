"""
===================================
Display Streamlines on a Surface
===================================

``cortex.Tractogram`` holds a bundle of streamlines -- the output of diffusion
tractography, for instance pyAFQ or DIPY bundles -- and draws them inside the
WebGL viewer on top of the cortical surface.

This example builds streamlines synthetically so that it runs anywhere, with no
diffusion data and no extra dependencies. Real tractograms are normally read
from a TRX file with ``cortex.Tractogram.from_trx``, which needs the optional
``trx-python`` dependency (``pip install "pycortex[tractography]"``).

The renders below are produced headlessly through Playwright, exactly as in
:ref:`sphx_glr_auto_examples_webgl_plot_panels_headless.py`. Streamlines run
inside the white matter, so the cortical surface is made translucent with the
``surface_opacity`` view parameter; with the default opaque surface only the
parts of a streamline that leave the cortex would be visible.

See :doc:`the tractography documentation </tractography>` for loading real data,
aligning it to the surfaces, and driving the viewer's tract panel.
"""

import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import cortex
import cortex.export

subject = "S1"

###############################################################################
# Build some synthetic bundles
# ----------------------------
# A ``Tractogram`` is a flat ``(N, 3)`` array of points in the same millimeter
# space as the subject's fiducial surfaces, plus an offset table marking where
# each streamline starts. ``from_streamlines`` builds that from the more
# convenient list-of-arrays form, so all we need is a way to make a bundle of
# curves: parallel, slightly bowed lines running between two points.


def make_bundle(start, end, bow_direction, n_streamlines=40, n_points=60,
                spread=5.0, bow=10.0, seed=0):
    """A bundle of bowed, jittered curves running from `start` to `end`."""
    rng = np.random.default_rng(seed)
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    unit = axis / np.linalg.norm(axis)

    # Two directions across the bundle, to spread the streamlines out in.
    bow_direction = np.asarray(bow_direction, float)
    across = np.cross(unit, bow_direction)
    across /= np.linalg.norm(across)
    bow_direction = np.cross(across, unit)

    t = np.linspace(0, 1, n_points)[:, None]
    profile = np.sin(np.pi * t)  # zero at both ends, one in the middle
    core = start + t * axis + bow * profile * bow_direction

    return [
        (core + offset[0] * across + offset[1] * bow_direction).astype(np.float32)
        for offset in spread * rng.normal(size=(n_streamlines, 2))
    ]


# Three bundles, each running along one anatomical axis. Coordinates are in
# scanner RAS millimeters (x right, y anterior, z superior), inside S1's brain.
bundles = {
    "transverse": make_bundle((-45, 15, 20), (45, 15, 20), (0, 0, 1), seed=0),
    "longitudinal": make_bundle((-38, -35, 5), (-38, 55, 5), (0, 0, 1), seed=1),
    "vertical": make_bundle((-22, 10, -30), (-22, 10, 45), (0, 1, 0), seed=2),
}

# `groups` names subsets of streamlines by index. A TRX file carries these
# already (pyAFQ writes one group per bundle), and they become the per-bundle
# checkboxes in the viewer's tract panel.
streamlines, groups, start = [], {}, 0
for name, lines in bundles.items():
    groups[name] = np.arange(start, start + len(lines))
    streamlines.extend(lines)
    start += len(lines)

tract = cortex.Tractogram.from_streamlines(streamlines, subject, groups=groups)
print(f"{len(tract)} streamlines, {tract.n_points} points, "
      f"groups: {list(tract.groups)}")

###############################################################################
# Show them with a surface
# ------------------------
# A tractogram cannot be displayed on its own -- the viewer is built around a
# cortical surface -- so it travels in a ``Dataset`` together with at least one
# ``Volume`` or ``Vertex``. Here the accompanying view is the subject's own
# curvature, in grayscale, so that the streamline colors stay readable.

curvature = cortex.db.get_surfinfo(subject, "curvature")
overlay = cortex.Vertex(curvature.data, subject, cmap="gray", vmin=-1, vmax=1)

dataset = cortex.Dataset(overlay=overlay, bundles=tract)

# Streamlines only make sense against the folded surface, so render the
# fiducial surface, and turn it translucent to see inside the brain.
view = {
    "surface.{subject}.unfold": 0,
    "surface.{subject}.surface_opacity": 0.35,
}


# An oblique left view, so that all three bundles are visible at once rather
# than one of them running straight into the screen. `save_3d_views` takes
# either a named angle or a (name, parameters) pair.
ANGLE = ("oblique_left", {"camera.azimuth": 125, "camera.altitude": 70})


def render(data, name, angle=ANGLE):
    path = cortex.export.save_3d_views(
        data,
        base_name=name,
        list_angles=[angle],
        list_surfaces=[view],
        viewer_params=dict(labels_visible=[], overlays_visible=[]),
        size=(1024, 768),
        headless=True,
    )[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(plt.imread(path))
    ax.axis("off")
    return fig, ax


tmpdir = tempfile.mkdtemp()

fig, ax = render(dataset, os.path.join(tmpdir, "orientation"))
ax.set_title("color='orientation' (the default)", fontsize=13)
plt.show()

###############################################################################
# By default streamlines are colored by their local direction -- the standard
# directionally-encoded color scheme, in which left-right runs red,
# anterior-posterior green and inferior-superior blue. That is why the three
# bundles above come out in three different colors.
#
# Color by a scalar instead
# -------------------------
# Tractometry results are one value per point (``dpv``) or per streamline
# (``dps``). Naming one of them as the color maps it through a colormap like
# any other pycortex dataview, which is how a profile along a bundle gets onto
# the streamlines.

# A stand-in for a tractometry measure: position along each streamline.
position = np.concatenate(
    [np.linspace(0, 1, len(line)) for line in tract.streamlines]
).astype(np.float32)

scalar_tract = cortex.Tractogram.from_streamlines(
    streamlines, subject, groups=groups, dpv={"position": position},
    color="dpv:position", cmap="viridis", vmin=0, vmax=1,
)

fig, ax = render(
    cortex.Dataset(overlay=overlay, bundles=scalar_tract),
    os.path.join(tmpdir, "scalar"),
)
ax.set_title("color='dpv:position', cmap='viridis'", fontsize=13)
plt.show()

###############################################################################
# Interactively
# -------------
# In a live viewer the tractograms get their own panel under the dataset box,
# with a visibility checkbox and an opacity slider for each one, plus a
# checkbox per group when the tractogram has groups. The ``surface_opacity``
# slider in the surface controls is what makes the cortex translucent.
#
# The same controls are reachable from Python through the viewer handle::
#
#     handle = cortex.webgl.show(dataset)
#     handle.tracts.bundles.setOpacity(0.5)
#     handle.tracts.bundles.setGroupVisible("vertical", False)
#
# Large tractograms are worth decimating before display -- every point costs
# three floats and three color bytes in the browser. ``subsample`` returns a
# decimated copy rather than modifying the tractogram, so display what it
# returns::
#
#     smaller = tract.subsample(max_streamlines=5000)
#     handle = cortex.webgl.show(cortex.Dataset(overlay=overlay, bundles=smaller))
