"""
===============================
Save a 2D static flatmap as PNG
===============================

Plot a 2D static flatmap and save it as PNG file.

**Some words on the `recache` parameter before we begin:**

Setting the `recache=True` parameter recaches the flatmap cache located in
<filestore>/<subject>/cache. By default intermediate steps for a flatmap are
cached after the first generation to speed up the process for the future. If
any of the intermediate steps changes, the flatmap generation may fail.
`recache=True` will load these intermediate steps new.
This can be helpful if you think there is no reason that the
`quickflat.make_figure` to fail but it nevertheless fails. Try it, it's magic!

The default background is set to be a transparent image. If you want to change
that use the parameter `bgcolor`.

"""
import cortex
import numpy as np
np.random.seed(1234)

# Create several pycortex Volumes
volumes = {'first': cortex.Volume.random(subject='S1', xfmname='fullhead'),
           'second': cortex.Volume.random(subject='S1', xfmname='fullhead')}

# Generate an animated gif that switches between frames every 1.5 seconds
filename = "./my_flatmap.gif"
cortex.quickflat.make_gif(filename, volumes, frame_duration=1.5)


# Display gif inline in an IPython note
import io
from IPython.display import Image

stream = io.BytesIO()
cortex.quickflat.make_gif(stream, volumes, frame_duration=1.5)

Image(stream.read())
