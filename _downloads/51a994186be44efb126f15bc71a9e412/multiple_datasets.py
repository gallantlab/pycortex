"""
===============================================
Create a 3D WebGL Viewer with Multiple Datasets
===============================================

A webgl viewer displays a 3D view of brain data in a web browser

Multiple datasets can be loaded into the same viewer

The `priority` kwarg passed to Volume objects determines the display ordering

Lower values of `priority` are displayed first

In the browser you can switch between datasets with the + and - keys

"""

import cortex

import numpy as np
np.random.seed(1234)

# gather multiple datasets
volume1 = cortex.Volume.random(subject='S1', xfmname='fullhead', priority=1)
volume2 = cortex.Volume.random(subject='S1', xfmname='fullhead', priority=2)
volume3 = cortex.Volume.random(subject='S1', xfmname='fullhead', priority=3)
volumes = {
	'First Dataset': volume1,
	'Second Dataset': volume2,
	'Third Dataset': volume3,
}

# create viewer
cortex.webgl.show(data=volumes)

# a port number will then be output, for example "Started server on port 39140"
# the viewer can then be accessed in a web browser, in this case at "localhost:39140"
