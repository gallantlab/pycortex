"""
==============
Bumpy Flatmaps
==============

A flatmap throws away the folding of the cortical surface, which is a shame:
where the gyri and sulci were is genuinely useful context, and once a flatmap is
covered in data there is nothing left to show it. The usual fix is to shade the
flatmap by curvature underneath the data, but that only works where the data is
transparent.

The bumpy flatmap puts the folding back as relief instead of as color. Cortex is
2 to 5 mm thick, so imagine peeling the cortical slab off the white matter,
making the relaxation cuts, and laying it down. The white matter side ends up
flat, but the pial side sits some distance above it -- bunched up thicker over
gyri, which flattening compresses, and stretched thinner over sulci, which
flattening expands.

The obvious way to compute that height is to insist each column of tissue keeps
its folded volume: height is the folded volume over the flattened area. The
problem is the denominator. Flattening compresses some triangles almost to
nothing, and those give enormous heights -- so the naive bumpy flatmap is a
field of spikes, and smoothing it afterwards does not help, because a mean of
ratios is dominated by exactly the outliers you wanted to remove.

`cortex.polyutils.FlatSlab` instead treats the slab as a soft elastic solid,
pins its white matter side to the flatmap, and lets the pial side settle. What
keeps a real piece of tissue from spiking is not gravity -- over a 3 mm slab
that is a couple of percent effect -- but shear: a tall narrow column is
expensive because it shears against its neighbours, and it relieves that by
spreading sideways. So the pial surface is free to move in-plane, and the relief
comes out smooth without anything being smoothed.
"""

import numpy as np
import matplotlib.pyplot as plt

import cortex
from cortex.polyutils import FlatSlab
from cortex.polyutils.bumpy import legacy_js_height, naive_prism_height

subject = "S1"

# The relaxation is cached in the pycortex database, and is generated
# automatically when a flatmap is imported. Ask for it here in case it has not
# been; the first call takes a few minutes per hemisphere.
npz = cortex.db.get_surfinfo(subject, type="bumpy_flatmap")
offsets = np.vstack([npz["bump_left"], npz["bump_right"]])
npz.close()

# The two things it is being compared against, computed per hemisphere.
naive, legacy, thickness, onmap = [], [], [], []
for hemi in ["lh", "rh"]:
    wm, polys = cortex.db.get_surf(subject, "wm", hemi)
    pia, _ = cortex.db.get_surf(subject, "pia", hemi)
    flat, flatpolys = cortex.db.get_surf(subject, "flat", hemi)

    naive.append(naive_prism_height(flat, wm, pia, flatpolys))
    legacy.append(legacy_js_height(wm, pia, polys))
    thickness.append(np.linalg.norm(pia - wm, axis=1))

    mask = np.zeros(len(wm), bool)
    mask[flatpolys.ravel()] = True
    onmap.append(mask)

naive = np.concatenate(naive)
legacy = np.concatenate(legacy)
thickness = np.concatenate(thickness)
onmap = np.concatenate(onmap)
relaxed = offsets[:, 2]

###############################################################################
# The relief itself. The relaxed height is shown alongside the naive
# volume-preserving height on the same color scale, which is what makes the
# difference obvious: the naive map is mostly flat with a scatter of very bright
# spikes, because its range is set by a handful of crushed triangles.

vmax = float(np.percentile(relaxed[onmap], 99))
for name, height in [("relaxed", relaxed), ("naive V/A", naive)]:
    vertex = cortex.Vertex(height, subject, vmin=0, vmax=vmax, cmap="viridis")
    cortex.quickshow(vertex, with_rois=False, with_labels=False,
                     with_curvature=False)
    plt.title("bumpy flatmap height, %s (mm)" % name)

###############################################################################
# Where the spikes are. Plotted as distributions, on a log axis because the
# naive height's problem is entirely in its tail. Cortex is 2-5 mm thick, so a
# sensible bumpy flatmap should sit in roughly that range; the naive height runs
# well past it, and the height the viewer used to compute in javascript is
# systematically too tall because it never actually looked at the flatmap -- its
# denominator was the *folded* white matter area, so it reduces to thickness
# times a function of the folded pial-to-white area ratio.

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.logspace(-1, 2, 120)
for name, height in [("cortical thickness", thickness), ("naive V/A", naive),
                     ("legacy javascript", legacy), ("relaxed", relaxed)]:
    ax.hist(np.clip(height[onmap], 1e-2, None), bins=bins, histtype="step",
            label=name, linewidth=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("height above the flatmap (mm)")
ax.set_ylabel("vertices")
ax.legend()
ax.set_title("the naive height's problem is its tail")

###############################################################################
# How far the pial surface slides sideways. This is the part a vertical-prism
# model cannot represent at all, and it is what lets the relief spread instead
# of spiking: material squeezed out of a compressed column has somewhere to go.

slip = np.linalg.norm(offsets[:, :2], axis=1)
vertex = cortex.Vertex(slip, subject, vmin=0,
                       vmax=float(np.percentile(slip[onmap], 99)),
                       cmap="magma")
cortex.quickshow(vertex, with_rois=False, with_labels=False,
                 with_curvature=False)
plt.title("in-plane slip of the pial surface (mm)")

###############################################################################
# Finally, the check that the relief means what it should: height against mean
# curvature. Gyri, which flattening compresses, end up thicker; sulci end up
# thinner.

curv = cortex.db.get_surfinfo(subject, type="curvature")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hexbin(curv.data[onmap], relaxed[onmap], gridsize=60, bins="log",
          cmap="Blues")
ax.set_xlabel("mean curvature (sulci < 0 < gyri)")
ax.set_ylabel("relaxed height (mm)")
ax.set_title("relief follows the folding")

plt.show()
