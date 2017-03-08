"""
=====================================================
Create a 2D static flatmap using the quickflat module
=====================================================

quickflat visualizations use matplotlib to generate figure-quality 2D flatmaps.

Similar to webgl, this tool uses pixel-based mapping to project functional data
onto the cortical surfaces.

This demo will use randomly generated data and plot a flatmap. Different options
to visualize the data will be demonstrated.

"""
import cortex

# Create a random cortex Volume
volume = cortex.Volume.random(subject='S1', xfmname='retinotopy')

# Plot a flatmap
_ = cortex.quickflat.make_figure(volume)

# Plot 
_ = cortex.quickflat.make_figure(volume, with_curvature=True)
