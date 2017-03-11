"""
=====================================
Plot Example Retinotopy in Web Viewer
=====================================

This demo shows how to plot example retinotopy data onto a subject's brain
in a web viewer. In order for this demo to work, you need to download this
dataset_, but that can also be done automatically through the `urllib`
command that is included.

.. _dataset: http://gallantlab.org/pycortex/S1_retinotopy.hdf

S1 is the example subject that comes with pycortex, but if you want to plot
data onto a different subject, you will need to have them in your filestore.

This demo will not actually open the web viewer for you, but if you run it
yourself you will get a viewer showing something like the following.

.. image:: ../../webgl/angle_left.png

"""

import cortex
try: #  python 2
    from urllib import urlretrieve
except ImportError:  # python 3
    from urllib.request import urlretrieve


# Download and load in retinotopy data
_ = urlretrieve("http://gallantlab.org/pycortex/S1_retinotopy.hdf",
                "S1_retinotopy.hdf")
ret_data = cortex.load("S1_retinotopy.hdf")

# Open the webviewer
cortex.webshow(ret_data)
