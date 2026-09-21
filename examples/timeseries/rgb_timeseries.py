"""
=============================
Timeseries of RGB Vertex Data
=============================

An RGB dataset carries three timecourses per vertex, one per color channel.
With the timeseries panel open (Open Controls -> movie -> timeseries), click
a vertex and the panel plots all three channels, each with its own checkbox
and color picker, above a strip showing the blended color over time - the
color actually painted on the brain at each volume.

The data is synthetic, built for the example subject(S1). Three patches on
the cortical surface pulse in the red, green and blue channels at different
rates, so each patch blinks in its own color and clicking inside one shows
which channel drives it.
"""

import numpy as np
import cortex

np.random.seed(42)
subject = "S1"
n_volumes = 120
t = np.arange(n_volumes)

(lpts, _), (rpts, _) = cortex.db.get_surf(subject, "wm")
pts = np.vstack([lpts, rpts])
n_vertices = len(pts)

def patch(center_vertex, radius_mm=18.0):
    """Gaussian surface patch (0-1) around one vertex, by 3D distance."""
    d2 = ((pts - pts[center_vertex]) ** 2).sum(1)
    return np.exp(-d2 / (2 * radius_mm ** 2)).astype("float32")

centers = np.random.choice(n_vertices, 3, replace=False)
rates = [8.0, 12.0, 4.0]                      # pulse periods
channels = []
for center, period in zip(centers, rates):
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t / period)          # (t,)
    signal = pulse[:, None] * patch(center)[None, :]            # (t, v)
    noise = 0.08 * np.random.rand(n_volumes, n_vertices)
    channels.append((np.clip(signal + noise, 0, 1) * 255).astype(np.uint8))
red, green, blue = channels
alpha = np.ones((n_volumes, n_vertices), dtype="float32")

rgb = cortex.VertexRGB(red, green, blue, subject, alpha=alpha)

cortex.webshow({"rgb": rgb})
# Like the other WebGL examples, run this from ipython/jupyter or with
# `python -i` so the interpreter (and the viewer's server) stays alive.
