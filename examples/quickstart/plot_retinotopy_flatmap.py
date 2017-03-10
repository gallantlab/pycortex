"""
================================
Plot Example Retinotopy Flatmaps
================================

This demo shows how to plot example retinotopy data onto a subject's brain
on a flatmap. In order for this demo to work, you need to download this
dataset_.

.. _dataset: http://gallantlab.org/pycortex/S1_retinotopy.hdf

S1 is the example subject that comes with pycortex, but if you want to plot
data onto a different subject, you will need to have them in your filestore,
and you will also need a flatmap for them.
"""

import cortex
import matplotlib.pyplot as plt

ret_data = cortex.load("S1_retinotopy.hdf")

# The retinotopy data has to be divided into left and right hemispheres
left_data = ret_data.angle_left
cortex.quickshow(left_data)
plt.show()

right_data = ret_data.angle_right
cortex.quickshow(right_data)
plt.show()
