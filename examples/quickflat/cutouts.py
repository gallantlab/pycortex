"""
=====================================================
Display a previously defined cutout on the flatmap
=====================================================

Cutouts are manually generated cuts of the cortical surface to highlight
a region of interest.

The option `cutout` for the `quickflat.make_figure` method takes a name of a
flatmap cutout.

This cutout name should be the name of a sub-layer of the 'cutouts' layer
in <filestore>/<subject>/rois.svg

"""
import cortex

# Name of a sub-layer of the 'cutouts' layer in rois.svg file
cutout_name = ""

# Create a random pycortex Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed
_ = cortex.quickflat.make_figure(volume,
                                 with_curvature=True,
                                 cutout=cutout_name)
