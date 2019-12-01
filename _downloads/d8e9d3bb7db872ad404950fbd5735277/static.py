"""
======================
Create a static viewer
======================

A static viewer is a brain viewer that exists permanently on a filesystem

The viewer is stored in a directory that stores html, javascript, data, etc

The viewer directory must be hosted by a server such as nginx
"""

import cortex

import numpy as np
np.random.seed(1234)

# gather data Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# select path for static viewer on disk
viewer_path = '/path/to/store/viewer'

# create viewer
cortex.webgl.make_static(outpath=viewer_path, data=volume, recache=True)

# a webserver such as nginx can then be used to host the static viewer
