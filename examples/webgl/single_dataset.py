"""
========================
Create a 3D WebGL Viewer
========================

A webgl viewer displays a 3D view of brain data in a web browser

"""

import cortex

# gather data Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# create viewer
cortex.webgl.show(data=volume)
