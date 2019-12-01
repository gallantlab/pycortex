"""
===================
Tissot's Indicatrix
===================

Creating a flatmap from a folded cortical surface always introduces some
distortion. This is similar to what happens when a map of the globe is flattened
into a 2-D map like a Mercator projection. For the cortical surface the amount
and type of distortion will depend on the curvature of the surface (i.e. whether
it is on a gyrus or a sulcus) and on the distance to the nearest cut.

In general, we recommend examining data both in flattened and original 3-D space
using the interactive webGL viewer, but it is also informative to visualize the
distortion directly.

One method to show distortion is to visualize how geodesic discs, which contain
all of the points within some geodesic distance of a central point, appear on the
flattened cortical surface. 

This technique is traditionally used to characterize and visualize distortions
introduced by flattening the globe onto a map:

.. image::https://upload.wikimedia.org/wikipedia/commons/8/87/Tissot_mercator.png

"""

import cortex
import matplotlib.pyplot as plt

tissot = cortex.db.get_surfinfo("S1", "tissots_indicatrix", radius=10, spacing=30)
tissot.cmap = 'plasma'

cortex.quickshow(tissot, with_labels=False, with_rois=False, with_colorbar=False)

plt.show()