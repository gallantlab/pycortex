from .dataset import Dataset, Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D
from . import align, volume, quickflat, webgl, segment, options
from .database import db
from .utils import *
from .quickflat import make_figure as quickshow

load = Dataset.from_file

try:
	from . import webgl
	from .webgl import show as webshow
except ImportError:
	pass

try:
	from . import anat
except ImportError:
	pass

# Create deprecated interface for database
import warnings
class dep(object):
	def __getattr__(self, name):
		warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
		return getattr(db, name)
	def __dir__(self):
		warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
		return db.__dir__()
surfs = dep()