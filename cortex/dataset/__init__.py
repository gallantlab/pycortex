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

# `views` first, deliberately. It resolves the circular dependency on `viewRGB`
# and `view2D` with deferred imports at the bottom of its own module, so importing
# either of those first would have them reach back into a partially initialised
# `views`. `_typing` used to be the module that tripped this, via an import it no
# longer needs, so the hazard is currently latent rather than live -- kept ordered,
# and kept out of isort's reach, so that re-introducing such an import fails here
# rather than somewhere less obvious. `test_submodule_can_be_imported_first` pins
# that each submodule also works as the entry point.
from .views import (  # isort: skip
    Dataview,
    DataviewJSON,
    Packable,
    RenderableView,
    ScalarView,
    SurfaceView,
    Vertex,
    Volume,
    VolumetricView,
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
    HasSubject,
    as_renderable,
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
    "Packable",
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
    # the row axis of the grid, and helpers for narrowing it
    "VolumetricView",
    "SurfaceView",
    "RenderableView",
    "HasSubject",
    "as_renderable",
]
