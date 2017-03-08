"""Module for representing brain data in both voxel and vertex forms
"""

from .views import Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D, Dataview, _from_hdf_data
from .dataset import Dataset, normalize