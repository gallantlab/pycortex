"""Types for narrowing the ``cortex.dataset`` view grid, for consumers.

The six public view classes form a 2x3 grid. Both axes are now real classes, so
every narrowing test is a nominal ``isinstance`` -- sound, and understood by type
checkers in both branches:

======================  ==========================  ==========================
                        volumetric                  surface
======================  ==========================  ==========================
**row** (space)         :class:`VolumetricView`     :class:`SurfaceView`
scalar                  ``Volume``                  ``Vertex``
2D                      ``Volume2D``                ``Vertex2D``
RGB                     ``VolumeRGB``               ``VertexRGB``
**column** (channels)   ``ScalarView`` / ``Dataview2D`` / ``DataviewRGB``
======================  ==========================  ==========================

The columns were always classes; only the rows lacked a type, which is why
consumers had to duck-type ``hasattr(braindata, "xfmname")``. The row ABCs close
that gap, so this module holds only the boundary helper and a couple of aliases
-- there is nothing left for it to work around.

Narrowing needs no machinery. ``isinstance`` against a member of a union
subtracts it from the ``else`` branch::

    def f(view: Renderable) -> None:
        if isinstance(view, SurfaceView):
            view.vertices        # SurfaceView
        else:
            view.xfmname         # VolumetricView

No ``TypeGuard``, no ``TypeIs``, no ``Protocol``, no ``typing_extensions``, and no
predicate helpers. Earlier iterations of this module had all of those; ``TypeIs``
is only needed to carry that narrowing *across a function-call boundary*, i.e.
only if the check lives in a named helper rather than inline.

**Why abstract bases rather than ``Protocol``.** A ``runtime_checkable`` protocol's
``isinstance`` tests only for the *presence* of the member names -- it cannot tell
a property from a method or check any types -- so an object carrying an unrelated
``subject``/``xfmname``/``volume`` satisfies it. Composing several protocols does
not help: the check is still per-name ``hasattr``. Nominal bases give a real class
check, and because the row members are abstract, a view that forgets one cannot be
instantiated. The trade is that conformance is explicit opt-in: a third-party view
must inherit :class:`VolumetricView` or :class:`SurfaceView`, rather than merely
happening to have the right attributes.
"""

from __future__ import annotations

from typing import Any, Union

from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .view2D import Dataview2D
from .views import Dataview, ScalarView, SurfaceView, VolumetricView

#: Anything the flatmap renderers can draw: one row of the grid or the other.
Renderable = Union[VolumetricView, SurfaceView]

#: The views carrying a colormap, as opposed to their own colours -- the scalar
#: and 2D columns, but not RGB. The typed replacement for
#: ``hasattr(braindata, "cmap")``; note that test matched 2D views as well as
#: scalar ones, so a scalar-only test is not equivalent.
#:
#: Test at runtime with an **inline** tuple, ``isinstance(view, (ScalarView,
#: Dataview2D))``. A module-level ``tuple[type, ...]`` constant would read more
#: nicely but cannot narrow: the annotation loses the members, so mypy learns
#: nothing from the check.
ColormappedView = Union[ScalarView, "Dataview2D[Any]"]


def as_renderable(view: Dataview) -> Renderable:
    """Check that ``view`` is one the renderers can draw.

    The single boundary between "some view" and "a view this code can render".
    Public entry points legitimately accept any :class:`Dataview`, so something
    has to make the check explicit and fail with a useful message rather than
    dying later on a missing ``.xfmname`` or ``.volume``.

    Parameters
    ----------
    view : Dataview
        Any view, typically straight out of :func:`cortex.dataset.normalize`.

    Returns
    -------
    VolumetricView or SurfaceView

    Raises
    ------
    TypeError
        If ``view`` is neither.
    """
    if isinstance(view, (VolumetricView, SurfaceView)):
        return view
    # Naming the space is useful, but reading it must not be able to raise: this
    # is a diagnostic path, and `space` is a property on an unknown class.
    try:
        space_name = type(view.space).__name__
    except Exception:
        space_name = "<unavailable>"
    raise TypeError(
        "%s (space %s) is not renderable: it subclasses neither VolumetricView "
        "(xfmname, volume) nor SurfaceView (vertices). A view in a new space "
        "should inherit whichever of the two describes how its data is sampled."
        % (type(view).__name__, space_name)
    )


def space_of(view: Dataview) -> BrainSpace:
    """The space a view's data lives in.

    Use this when the question is genuinely about the space rather than about how
    the data is sampled -- e.g. checking that two views share a transform. For
    "can I render this", use :data:`Renderable` and ``isinstance``.
    """
    return view.space


__all__ = [
    "VolumetricView",
    "SurfaceView",
    "Renderable",
    "ColormappedView",
    "as_renderable",
    "space_of",
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
]
