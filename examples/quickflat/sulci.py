"""
=================================================
Display sulci on the flatmap.
=================================================

Setting the parameter `with_sulci` for the `quickflat.make_figure` to `True`
will display the defined sulci.

The sulcis are defined in a sub-layer of the sulci layer in
<filestore>/<subject>/rois.svg
"""
import cortex

# Create a random pycortex Volume
volume = cortex.Volume.random(subject='S1', xfmname='retinotopy')

# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and dropout regions
_ = cortex.quickflat.make_figure(volume,
                                 with_curvature=True,
                                 with_sulci=True)
