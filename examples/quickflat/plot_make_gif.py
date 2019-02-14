"""
====================================
Animate a series of volumes as a GIF
====================================

A convenient way to compare two flat maps (e.g., prediction performance or
tuning weights) is to flip back and forth between them. This example shows how
to make an animated gif in which each frame is a flatmap.

"""
import cortex
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

################################################################################
# Create several pycortex Volumes
# 

volumes = {'first': cortex.Volume.random(subject='S1', xfmname='fullhead', vmin=-2, vmax=2),
           'second': cortex.Volume.random(subject='S1', xfmname='fullhead', vmin=-2, vmax=2)}

################################################################################
# Plot flat maps individually
#

_ = cortex.quickflat.make_figure(volumes['first'])
_ = cortex.quickflat.make_figure(volumes['second'])
_ = plt.show()


################################################################################
# Generate an animated gif that switches between frames every 1.5 seconds
#

filename = "./flatmap_comparison.gif"
cortex.quickflat.make_gif(filename, volumes, frame_duration=1.5)

################################################################################
# Display gif inline in an IPython notebook
#

import io
from IPython.display import Image

stream = io.BytesIO()
cortex.quickflat.make_gif(stream, volumes, frame_duration=1.5)

Image(stream.read())


################################################################################
# .. image:: ../../flatmap_comparison.gif
