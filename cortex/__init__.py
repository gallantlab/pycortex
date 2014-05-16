from .dataset import Dataset, VolumeData, VertexData, DataView, View
from . import align, volume, quickflat, webgl, segment, options
from .database import db
from .utils import *
from .quickflat import make_figure as quickshow

openFile = Dataset.from_file

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
	def __getattr__(self, *args, **kwargs):
		warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
		return db.__getattr__(*args, **kwargs)
	def __dir__(self):
		warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
		return db.__dir__()
surfs = dep()