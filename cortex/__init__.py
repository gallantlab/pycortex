# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set fileencoding=utf-8 ft=python sts=4 ts=4 sw=4 et:
from cortex.dataset import Dataset, Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D, Colors
from cortex import align, volume, quickflat, webgl, segment, options
from cortex.database import db
from cortex.utils import *
from cortex.quickflat import make_figure as quickshow
from cortex.volume import mosaic, unmask
import cortex.export
from cortex.version import __version__, __full_version__

try:
    from cortex import formats
except ImportError:
    raise ImportError("Either are running pycortex from the source directory, or the build is broken. "
                      "If your current working directory is 'cortex', where pycortex is installed, then change this. "
                      "If your current working directory is somewhere else, then you may have to rebuild pycortex.")

load = Dataset.from_file

try:
    from cortex import webgl
    from cortex.webgl import show as webshow
except ImportError:
    pass

try:
    from cortex import anat
except ImportError:
    pass

# Create deprecated interface for database
class dep(object):
    def __getattr__(self, name):
        warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
        return getattr(db, name)
    def __dir__(self):
        warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
        return db.__dir__()

surfs = dep()

import sys
if sys.version_info.major == 2:
    stdout = sys.stdout
    reload(sys)
    sys.setdefaultencoding('utf8')
    sys.stdout = stdout
