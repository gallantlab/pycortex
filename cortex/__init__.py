from . import align, volume, quickflat, webgl
from .utils import *

from .dataset import Dataset, VolumeData, VertexData
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

