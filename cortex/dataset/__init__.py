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

# Import order matters. `views` resolves the circular dependency on `viewRGB` and
# `view2D` with deferred imports at the bottom of its own module, so it has to be
# imported first -- anything that reaches `view2D`/`viewRGB` before then (such as
# `_typing`) would see a partially initialised module.
from .views import (  # isort: skip
    Dataview,
    DataviewJSON,
    ScalarView,
    Vertex,
    Volume,
    _from_hdf_data,
)
from ._space import (
    BrainSpace,
    SurfaceSpace,
    VolumeSpace,
    register_space,
    registered_spaces,
)
from ._typing import (
    Renderable,
    SupportsColormap,
    SurfaceRenderable,
    VolumetricRenderable,
    as_renderable,
    space_of,
)
from .braindata import BrainData, VertexData, VolumeData
from .dataset import Dataset, DatasetLike, normalize
from .view2D import Dataview2D, Vertex2D, Volume2D
from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB

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
    # spaces -- the open axis
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
    "register_space",
    "registered_spaces",
    # structural types describing what renderers need from a view
    "VolumetricRenderable",
    "SurfaceRenderable",
    "Renderable",
    "SupportsColormap",
    "as_renderable",
    "space_of",
]
