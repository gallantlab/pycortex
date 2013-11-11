from .dataset import Dataset, VolumeData, VertexData, DataView, View
from . import align, volume, quickflat, webgl, segment, options
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

