"""Structural types describing what renderers need from a brain-data view.

The six public view classes form a 2x3 grid, but only its columns exist as
classes: the channel layout is the inheritance axis, so there is no class meaning
"any volumetric view". Consumers used to reach for ``hasattr(braindata,
"xfmname")``, which no type checker can narrow.

The answer here is a pair of ``Protocol``\\ s describing the *interface* a
renderer actually consumes, rather than a closed union enumerating the classes
that happen to satisfy it today:

- :class:`VolumetricRenderable` -- has a subject, a transform, and a ``volume``
- :class:`SurfaceRenderable` -- has a subject and ``vertices``

Each of the six built-in views satisfies exactly one of them. Two properties of
this design matter:

**It narrows both branches.** ``isinstance`` against a member of a union subtracts
that member from the ``else`` branch, so::

    def f(view: Renderable) -> None:
        if isinstance(view, SurfaceRenderable):
            view.vertices        # SurfaceRenderable
        else:
            view.xfmname         # VolumetricRenderable

Nothing else is needed for this -- no ``TypeGuard``, no ``TypeIs``, and no
predicate helpers. (An inline ``isinstance`` against a union of *concrete*
classes narrows identically; ``TypeIs`` is only required to carry that narrowing
across a function-call boundary, which is why an earlier version of this module
had six predicates and a ``typing_extensions`` question. It needed neither.)

**It is open.** A view in a space registered by third-party code conforms
structurally, with no union to edit and no registry entry to add. That is the
property the closed ``VolumeLike``/``VertexLike`` unions could not have.

Two limits worth knowing:

- ``runtime_checkable`` makes ``isinstance`` check only for the *presence* of the
  named attributes -- it cannot tell a property from a method, or check types. So
  the runtime check is exactly as strong as the ``hasattr`` it replaces; the
  static check is what carries the weight. (Statically mypy does compare types,
  and correctly rejects :class:`~cortex.dataset.views.Vertex` from
  :class:`VolumetricRenderable` because its ``volume`` is a *method*.)
- Protocol members must be declared the way the classes declare them. ``subject``
  is a read-only property on :class:`~cortex.dataset.views.Dataview`, so
  declaring ``subject: str`` in a protocol fails with *"expected settable
  variable, got read-only attribute"*. The colormap members really are mutable
  attributes, so :class:`SupportsColormap` declares them as such.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import numpy.typing as npt

from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .views import Dataview


@runtime_checkable
class VolumetricRenderable(Protocol):
    """A view whose data can be sampled as a 3D/4D volume under a transform."""

    @property
    def subject(self) -> str:
        """Subject identifier. Must exist in the pycortex database."""

    @property
    def xfmname(self) -> str:
        """Transform name. Must exist in the pycortex database."""

    @property
    def volume(self) -> npt.NDArray:
        """The data as a volume, with a leading time axis.

        Scalar for :class:`~cortex.dataset.views.Volume`; uint8 RGBA for the 2D
        and RGB views, whose data has already been colormapped.
        """


@runtime_checkable
class SurfaceRenderable(Protocol):
    """A view whose data can be sampled per-vertex on a cortical surface."""

    @property
    def subject(self) -> str:
        """Subject identifier. Must exist in the pycortex database."""

    @property
    def vertices(self) -> npt.NDArray:
        """The data per vertex, with a leading time axis.

        Scalar for :class:`~cortex.dataset.views.Vertex`; uint8 RGBA for the 2D
        and RGB views, whose data has already been colormapped.
        """


#: Anything the flatmap renderers can draw.
Renderable = Union[SurfaceRenderable, VolumetricRenderable]


@runtime_checkable
class SupportsColormap(Protocol):
    """A view carrying a 1D or 2D colormap, as opposed to its own colours.

    Satisfied by the scalar and 2D views; not by the RGB views, which have no
    ``cmap``. This is the typed replacement for ``hasattr(braindata, "cmap")`` --
    note that test matched 2D views as well as scalar ones, so a scalar-only test
    would not be equivalent.
    """

    cmap: str
    vmin: Optional[float]
    vmax: Optional[float]


def as_renderable(view: Dataview) -> Renderable:
    """Check that ``view`` exposes an interface the renderers can draw.

    The single boundary between "some view" and "a view this code can render".
    Public entry points legitimately accept any
    :class:`~cortex.dataset.views.Dataview` -- including one in a space
    registered by third-party code -- so something has to make the check explicit
    and fail with a useful message rather than dying later on a missing
    ``.xfmname`` or ``.volume``.

    Unlike an enumeration of known classes, this accepts a view in any space, so
    long as it exposes the interface.

    Parameters
    ----------
    view : Dataview
        Any view, typically straight out of :func:`cortex.dataset.normalize`.

    Returns
    -------
    A view satisfying :class:`VolumetricRenderable` or :class:`SurfaceRenderable`.

    Raises
    ------
    TypeError
        If ``view`` exposes neither interface.
    """
    if isinstance(view, (VolumetricRenderable, SurfaceRenderable)):
        return view
    # Naming the space is useful, but reading it must not be able to raise: this
    # is a diagnostic path, and `space` is a property on an unknown class.
    try:
        space_name = type(view.space).__name__
    except Exception:
        space_name = "<unavailable>"
    raise TypeError(
        "%s (space %s) is not renderable: it exposes neither a volumetric "
        "interface (subject, xfmname, volume) nor a surface one "
        "(subject, vertices)" % (type(view).__name__, space_name)
    )


def space_of(view: Dataview) -> BrainSpace:
    """The space a view's data lives in.

    Use this when the question is genuinely about the space rather than about the
    interface -- e.g. checking that two views share a transform. Prefer the
    protocols when the question is "can I render this".
    """
    return view.space


__all__ = [
    "VolumetricRenderable",
    "SurfaceRenderable",
    "Renderable",
    "SupportsColormap",
    "as_renderable",
    "space_of",
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
]
