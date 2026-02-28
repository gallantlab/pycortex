"""Contains classes for representing brain data in either volumetric or vertex (surface-based) formats for visualization.
"""

from __future__ import annotations
from .views import Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D, Dataview, Dataview2D, _from_hdf_data, Colors
from .dataset import Dataset, normalize
