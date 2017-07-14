"""
========================
Create a 3D WebGL Viewer
========================

A webgl viewer displays a 3D view of brain data in a web browser

"""

import cortex

import numpy as np
np.random.seed(1234)

# gather data Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# create viewer
cortex.webgl.show(data=volume)

# a port number will then be output, for example "Started server on port 39140"
# the viewer can then be accessed in a web browser, in this case at "localhost:39140"
