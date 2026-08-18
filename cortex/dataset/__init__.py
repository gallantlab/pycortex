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

# `views` first, and kept out of isort's reach, because the constraint is live:
# `views` closes its circular dependency on `viewRGB`/`view2D` with deferred
# imports at the very bottom of its own module, so if `view2D` is imported before
# `views` has finished, it pulls in `viewRGB`, which re-enters `views`, whose
# bottom then finds `viewRGB` half-built. Hoisting the `.view2D` line above this
# one fails with `ImportError: cannot import name 'Colors' from partially
# initialized module`, which is what that ordering buys.
# `test_submodule_can_be_imported_first` separately pins that each submodule still
# works as the process's entry point.
from .views import (  # isort: skip
    Dataview,
    DataviewJSON,
    HasSubject,
    Packable,
    RenderableView,
    ScalarView,
    SurfaceView,
    as_renderable,
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
    # the spatial axis of the grid, and helpers for narrowing it
    "VolumetricView",
    "SurfaceView",
    "RenderableView",
    "HasSubject",
    "as_renderable",
]
