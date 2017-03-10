"""
=====================================
Plot Example Retinotopy in Web Viewer
=====================================

This demo shows how to plot example retinotopy data onto a subject's brain
in a web viewer. In order for this demo to work, you need to download this
dataset_.

.. _dataset: http://gallantlab.org/pycortex/S1_retinotopy.hdf

S1 is the example subject that comes with pycortex, but if you want to plot
data onto a different subject, you will need to have them in your filestore.
"""

import cortex

ret_data = cortex.load("S1_retinotopy.hdf")
cortex.webshow(ret_data)
