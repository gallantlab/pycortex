"""
=====================================================
Create a 2D static flatmap using the quickflat module
=====================================================

quickflat visualizations use matplotlib to generate figure-quality 2D flatmaps.

Similar to webgl, this tool uses pixel-based mapping to project functional data
onto the cortical surfaces.

This demo will use randomly generated data and plot a flatmap. Different
options to visualize the data will be demonstrated.

"""
import cortex

# Create a random pycortex Volume
volume = cortex.Volume.random(subject='S1', xfmname='retinotopy')

# Plot a flatmap with the data projected onto the surface
# By default ROIs and their labels will be overlaid to the plot
# Also a colorbar will be added
_ = cortex.quickflat.make_figure(volume)

# Highlight the curvature
_ = cortex.quickflat.make_figure(volume, with_curvature=True)

# Remove ROI labels from the plot
_ = cortex.quickflat.make_figure(volume,
                                 with_curvature=True,
                                 with_labels=False)

# Remove ROIs from the plot
_ = cortex.quickflat.make_figure(volume,
                                 with_curvature=True,
                                 with_rois=False)

# Remove the colorbar from the plot
_ = cortex.quickflat.make_figure(volume,
                                 with_curvature=True,
                                 with_colorbar=False)
