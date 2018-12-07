# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set fileencoding=utf-8 ft=python sts=4 ts=4 sw=4 et:
import warnings

from cortex.dataset import Dataset, Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D
from cortex import align, volume, quickflat, webgl, segment, options
from cortex.database import db
from cortex.utils import *
from cortex.quickflat import make_figure as quickshow
from cortex.volume import mosaic, unmask

try:
    from cortex import formats
except ImportError:
    raise ImportError("You are running pycortex from the source directory. Don't do that!")

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
