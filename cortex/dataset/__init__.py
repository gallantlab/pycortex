"""Contains classes for representing brain data in either volumetric or vertex
(surface-based) formats for visualization.

The six public view classes form a 2x3 grid: two spaces (volumetric, surface)
crossed with three channel layouts (scalar + 1D colormap, two channels + 2D
colormap, three channels + alpha).

===========  ===============  ===============  ===============
space        scalar           2D               RGB
===========  ===============  ===============  ===============
volumetric   :class:`Volume`  :class:`Volume2D`  :class:`VolumeRGB`
surface      :class:`Vertex`  :class:`Vertex2D`  :class:`VertexRGB`
===========  ===============  ===============  ===============

The channel layout is the inheritance axis: :class:`Dataview` is the common
root, and :class:`ScalarView`, :class:`Dataview2D` and :class:`DataviewRGB` sit
under it. See ``INHERITANCE.md`` for the map and ``TYPING_ALTERNATIVES.md`` for
the restructuring options that were considered.
"""

from __future__ import annotations

from .braindata import BrainData, VertexData, VolumeData
from .dataset import Dataset, DatasetLike, normalize
from .view2D import Dataview2D, Vertex2D, Volume2D
from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB
from .views import (
    Dataview,
    DataviewJSON,
    ScalarView,
    Vertex,
    Volume,
    _from_hdf_data,
)

__all__ = [
    # the six public view classes
    "Volume",
    "Vertex",
    "Volume2D",
    "Vertex2D",
    "VolumeRGB",
    "VertexRGB",
    # containers and helpers
    "Dataset",
    "DatasetLike",
    "Colors",
    "normalize",
    "DataviewJSON",
    # abstract bases, by both their current and their historical names
    "Dataview",
    "ScalarView",
    "Dataview2D",
    "DataviewRGB",
    "BrainData",
    "VolumeData",
    "VertexData",
]
