"""
======================
Create a static viewer
======================

A static viewer is a brain viewer that exists permanently on a filesystem

The viewer is stored in a directory that stores html, javascript, data, etc

The viewer directory must be hosted by a server such as nginx


# NOTE
# I don't know why, but when I create static viewers sometimes I have to create the data subdir like this
# import os
# os.mkdir(os.path.join(viewer_path, 'data'))
"""

import numpy as np
import cortex

data = np.random.randn(31, 100, 100)
# viewer_path = '/auto/k1/storm/www/test/static'
viewer_path = '/path/to/store/viewer'
volume = cortex.Volume(data=data, subject="S1", xfmname="fullhead")
cortex.webgl.make_static(outpath=viewer_path, data=volume)
