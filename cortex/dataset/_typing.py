"""Type aliases and narrowing helpers for the ``cortex.dataset`` view classes.

The six public view classes form a 2x3 grid, but only its columns exist as
classes: the channel layout is the inheritance axis, so there is no class meaning
"any volumetric view". Consumers used to reach for ``hasattr(braindata,
"xfmname")``, which no type checker can narrow. This module provides the two
things they actually need:

- **Closed unions** (:data:`VolumeLike`, :data:`VertexLike`) with ``TypeGuard``
  helpers, for the common case of "one of the six built-in views". These narrow
  ``braindata`` itself, which is what callers want.
- **An open test** (:func:`space_of`, plus ``isinstance(view.space, ...)``) for
  code that must keep working when a new space is registered. The unions cannot
  cover a space that did not exist when they were written; ``view.space`` can.

``TypeGuard`` rather than ``TypeIs``: ``TypeIs`` only reached the standard
library in 3.13, and this package supports 3.10. The practical difference is that
``TypeGuard`` narrows the positive branch only -- in the ``else`` of
``if is_volume_view(x)``, ``x`` keeps its declared type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, Union

from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .view2D import Vertex2D, Volume2D
from .viewRGB import VertexRGB, VolumeRGB
from .views import Dataview, ScalarView, Vertex, Volume

if TYPE_CHECKING:
    from .view2D import Dataview2D
    from .viewRGB import DataviewRGB

#: Any of the three built-in volumetric views.
VolumeLike = Union[Volume, Volume2D, VolumeRGB]

#: Any of the three built-in surface views.
VertexLike = Union[Vertex, Vertex2D, VertexRGB]

#: Any view that carries a single array of scalar values plus a 1D colormap.
#: This is the stand-in for the ``BrainData & Dataview`` intersection that
#: several comments in this package used to ask for; it is now a real class.
ScalarLike = ScalarView

_VOLUME_VIEWS: tuple[type, ...] = (Volume, Volume2D, VolumeRGB)
_VERTEX_VIEWS: tuple[type, ...] = (Vertex, Vertex2D, VertexRGB)


def is_volume_view(view: Any) -> TypeGuard[VolumeLike]:
    """Whether ``view`` is one of the three built-in volumetric views.

    Closed over the built-ins. For code that must also accept a newly registered
    space, test the space instead::

        if isinstance(view.space, VolumeSpace): ...
    """
    return isinstance(view, _VOLUME_VIEWS)


def is_vertex_view(view: Any) -> TypeGuard[VertexLike]:
    """Whether ``view`` is one of the three built-in surface views."""
    return isinstance(view, _VERTEX_VIEWS)


def is_scalar_view(view: Any) -> TypeGuard[ScalarView]:
    """Whether ``view`` carries scalar data and a 1D colormap.

    True for :class:`~cortex.dataset.views.Volume` and
    :class:`~cortex.dataset.views.Vertex`, false for the 2D and RGB views. This
    is the typed replacement for ``hasattr(braindata, "cmap")``.
    """
    return isinstance(view, ScalarView)


def is_colormapped(view: Any) -> TypeGuard[Union[ScalarView, "Dataview2D[Any]"]]:
    """Whether ``view`` has ``cmap``/``vmin``/``vmax``.

    True for the scalar and 2D views, false for RGB, which carries its own
    colours. This is the typed replacement for ``hasattr(braindata, "cmap")`` --
    note that test matched 2D views as well as scalar ones, so a scalar-only
    predicate would not be equivalent.
    """
    from .view2D import Dataview2D

    return isinstance(view, (ScalarView, Dataview2D))


def is_2d_view(view: Any) -> TypeGuard["Dataview2D[Any]"]:
    """Whether ``view`` holds two channels under a 2D colormap."""
    from .view2D import Dataview2D

    return isinstance(view, Dataview2D)


def is_rgb_view(view: Any) -> TypeGuard["DataviewRGB[Any]"]:
    """Whether ``view`` carries its own colours rather than a colormap."""
    from .viewRGB import DataviewRGB

    return isinstance(view, DataviewRGB)


def space_of(view: Dataview) -> BrainSpace:
    """The space a view's data lives in.

    Prefer this plus ``isinstance`` on the result over the ``*Like`` unions when
    the code should keep working for spaces added later.
    """
    return view.space


__all__ = [
    "VolumeLike",
    "VertexLike",
    "ScalarLike",
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
    "is_volume_view",
    "is_vertex_view",
    "is_scalar_view",
    "is_colormapped",
    "is_2d_view",
    "is_rgb_view",
    "space_of",
]
