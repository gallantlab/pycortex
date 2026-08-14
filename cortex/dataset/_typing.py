"""Type aliases and narrowing helpers for the ``cortex.dataset`` view classes.

The six public view classes form a 2x3 grid, but only its columns exist as
classes: the channel layout is the inheritance axis, so there is no class meaning
"any volumetric view". Consumers used to reach for ``hasattr(braindata,
"xfmname")``, which no type checker can narrow. This module provides the two
things they actually need.

**Closed unions plus ``TypeIs`` predicates**, for the common case of "one of the
six built-in views". These narrow ``braindata`` itself, in *both* branches --
``if is_vertex_view(v): ... else: ...`` gives ``VertexLike`` and ``VolumeLike``
respectively. That is why they are ``TypeIs`` (PEP 742) rather than ``TypeGuard``
(PEP 647): ``TypeGuard`` narrows the positive branch only, so the ``else`` of a
volume/surface fork kept the full six-member union and every attribute access in
it had to be re-guarded. ``TypeIs`` is the right tool whenever the narrowed type
is a subtype of the parameter type, which is true of every predicate here.

Because the predicates take :data:`BuiltinView` rather than ``Any``, code holding
a bare :class:`~cortex.dataset.views.Dataview` converts once at the boundary with
:func:`as_builtin_view`, which is also where an unrecognised space is rejected
with a clear error instead of a later ``AttributeError``.

**An open test** (:func:`space_of`) for code that must keep working when a new
space is registered. The unions cannot cover a space that did not exist when they
were written; ``view.space`` can::

    if isinstance(space_of(view), VolumeSpace): ...

``TypeIs`` is imported under ``TYPE_CHECKING``. ``TypeIs`` only reached the
standard library in 3.13 and this package supports 3.10, but because
``from __future__ import annotations`` is in force the annotation is never
evaluated at runtime -- so ``typing_extensions`` stays a
``python_version < '3.11'`` dependency rather than becoming unconditional. The
one consequence is that ``typing.get_type_hints()`` on these predicates raises
``NameError``; nothing in pycortex or its docs build calls it on them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .view2D import Vertex2D, Volume2D
from .viewRGB import VertexRGB, VolumeRGB
from .views import Dataview, Vertex, Volume

if TYPE_CHECKING:
    from typing_extensions import TypeIs

#: Any of the three built-in volumetric views.
VolumeLike = Union[Volume, Volume2D, VolumeRGB]

#: Any of the three built-in surface views.
VertexLike = Union[Vertex, Vertex2D, VertexRGB]

#: Any of the six built-in views -- the whole 2x3 grid.
#:
#: Deliberately a closed union of concrete classes rather than the ``Dataview``
#: base: that is what makes the ``TypeIs`` predicates able to subtract in their
#: negative branch. A view in a newly registered space is *not* a member; use
#: :func:`space_of` for code that must handle those.
BuiltinView = Union[VolumeLike, VertexLike]

#: The two views carrying a single array of scalar values plus a 1D colormap.
#: This is the stand-in for the ``BrainData & Dataview`` intersection that
#: several comments in this package used to ask for; it is now a real class,
#: :class:`~cortex.dataset.views.ScalarView`.
ScalarLike = Union[Volume, Vertex]

#: The two views carrying two channels under a 2D colormap.
TwoDLike = Union[Volume2D, Vertex2D]

#: The two views carrying their own colours rather than a colormap.
RGBLike = Union[VolumeRGB, VertexRGB]

#: The four views that have ``cmap``/``vmin``/``vmax``: scalar and 2D, not RGB.
ColormappedLike = Union[Volume, Vertex, Volume2D, Vertex2D]


def as_builtin_view(view: Dataview) -> BuiltinView:
    """Narrow a :class:`~cortex.dataset.views.Dataview` to one of the six built-ins.

    The single boundary between "some view" and "a view this code knows how to
    render". Raises rather than letting an unrecognised space fail later with an
    ``AttributeError`` on ``.xfmname`` or ``.volume``.

    Parameters
    ----------
    view : Dataview
        Any view, typically straight out of :func:`cortex.dataset.normalize`.

    Returns
    -------
    Volume, Vertex, Volume2D, Vertex2D, VolumeRGB or VertexRGB

    Raises
    ------
    TypeError
        If ``view`` lives in a space that is not one of the built-ins.
    """
    # Inline tuple literal, not a module-level constant: a `tuple[type, ...]`
    # loses its members, so mypy would not narrow to the union.
    if isinstance(view, (Volume, Volume2D, VolumeRGB, Vertex, Vertex2D, VertexRGB)):
        return view
    # Naming the space is useful, but reading it must not be able to raise: this
    # is a diagnostic path, and `space` is a property on an unknown class.
    try:
        space_name = type(view.space).__name__
    except Exception:
        space_name = "<unavailable>"
    raise TypeError(
        "%s is not one of the six built-in views (Volume, Vertex, Volume2D, "
        "Vertex2D, VolumeRGB, VertexRGB); its space is %s, which this code path "
        "does not know how to render"
        % (type(view).__name__, space_name)
    )


def is_volume_view(view: BuiltinView) -> TypeIs[VolumeLike]:
    """Whether ``view`` is one of the three volumetric views.

    Narrows both branches, so the ``else`` of this test is :data:`VertexLike`.
    For code that must also accept a newly registered space, test the space
    instead: ``isinstance(view.space, VolumeSpace)``.
    """
    return isinstance(view, (Volume, Volume2D, VolumeRGB))


def is_vertex_view(view: BuiltinView) -> TypeIs[VertexLike]:
    """Whether ``view`` is one of the three surface views.

    Narrows both branches, so the ``else`` of this test is :data:`VolumeLike`.
    """
    return isinstance(view, (Vertex, Vertex2D, VertexRGB))


def is_scalar_view(view: BuiltinView) -> TypeIs[ScalarLike]:
    """Whether ``view`` carries scalar data and a 1D colormap.

    True for :class:`~cortex.dataset.views.Volume` and
    :class:`~cortex.dataset.views.Vertex` only. Note this is *narrower* than
    "has a cmap" -- see :func:`is_colormapped`.
    """
    return isinstance(view, (Volume, Vertex))


def is_colormapped(view: BuiltinView) -> TypeIs[ColormappedLike]:
    """Whether ``view`` has ``cmap``/``vmin``/``vmax``.

    True for the scalar and 2D views, false for RGB, which carries its own
    colours. This is the typed replacement for ``hasattr(braindata, "cmap")`` --
    note that test matched 2D views as well as scalar ones, so
    :func:`is_scalar_view` would not be equivalent. The ``else`` branch of this
    test is :data:`RGBLike`.
    """
    return isinstance(view, (Volume, Vertex, Volume2D, Vertex2D))


def is_2d_view(view: BuiltinView) -> TypeIs[TwoDLike]:
    """Whether ``view`` holds two channels under a 2D colormap."""
    return isinstance(view, (Volume2D, Vertex2D))


def is_rgb_view(view: BuiltinView) -> TypeIs[RGBLike]:
    """Whether ``view`` carries its own colours rather than a colormap."""
    return isinstance(view, (VolumeRGB, VertexRGB))


def space_of(view: Dataview) -> BrainSpace:
    """The space a view's data lives in.

    Prefer this plus ``isinstance`` on the result over the ``*Like`` unions when
    the code should keep working for spaces added later.
    """
    return view.space


__all__ = [
    "VolumeLike",
    "VertexLike",
    "BuiltinView",
    "ScalarLike",
    "TwoDLike",
    "RGBLike",
    "ColormappedLike",
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
    "as_builtin_view",
    "is_volume_view",
    "is_vertex_view",
    "is_scalar_view",
    "is_colormapped",
    "is_2d_view",
    "is_rgb_view",
    "space_of",
]
