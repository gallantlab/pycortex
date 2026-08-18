r"""Types for narrowing the ``cortex.dataset`` view grid, for consumers.

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

**Which mechanism for which question.** The two are not interchangeable, and the
split is deliberate:

- **ABCs, for "what kind of view is this?"** -- :class:`VolumetricView` and
  :class:`SurfaceView`. This question is asked at *runtime*, so the check has to be
  sound, and only a nominal base class gives that.
- **Protocols, for "what does this function need?"** -- :class:`HasSubject`.
  A *static* contract on a parameter: it lets a function claim exactly what it touches
  instead of demanding a whole :class:`Dataview`. They are never ``isinstance``\ d,
  so it is deliberately **not** ``runtime_checkable`` -- which makes ``isinstance``
  against it raise ``TypeError``, mechanically preventing the presence-only check
  from creeping back in. A test pins that.

  Protocols compose by inheritance when one is a superset of another --
  ``class Blendable(HasSubject, Protocol)`` rather than re-declaring ``subject``.
  Note the ``Protocol`` base has to be repeated: ``class Blendable(HasSubject)``
  silently becomes an ordinary instantiable class.

**Why abstract bases rather than ``Protocol`` for the rows.** A ``runtime_checkable`` protocol's
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
from .views import (
    Dataview,
    HasSubject,
    Packable,
    RenderableView,
    ScalarView,
    SurfaceView,
    VolumetricView,
)

# `Packable` is re-exported from `.views` (imported above, not redeclared -- see the
# note on `HasSubject` below for why a same-shaped copy is a trap). It is the unit of
# transport: what `Dataview.uniques` yields, and the thing carrying a
# content-addressed `name`. Note it is *not* the row axis, and neither implies the
# other: `Packable` says "this is one addressable array" while `Renderable` says "a
# renderer can sample this", so a 2D view is renderable but not packable, and a bare
# ScalarView subclass is packable but has no row.

#: Anything the flatmap renderers can draw. An alias for the common base of every
#: row, not a union of the rows: a union would have to be edited whenever a row is
#: added, and code branching over it would silently mis-route the new one.
Renderable = RenderableView


# `HasSubject` is re-exported from `.views` rather than declared here, because
# `Dataview` inherits it and the two must be the same class. A same-shaped copy
# in this module type-checked identically -- structural protocols are
# interchangeable -- so the duplicate was invisible both to mypy and to the test
# asserting `Dataview` claims it, while `cortex.dataset.HasSubject` was in fact
# *not* the class any view inherited. Most of the `cortex.quickflat.composite`
# helpers need nothing more than this: they look up a flatmap by subject and draw
# on it, where annotating them `Dataview` claimed far more than they use.

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
    if isinstance(view, RenderableView):
        return view
    # Naming the space is useful, but reading it must not be able to raise: this
    # is a diagnostic path, and `space` is a property on an unknown class.
    try:
        space_name = type(view.space).__name__
    except Exception:
        space_name = "<unavailable>"
    raise TypeError(
        "%s (space %s) is not renderable: it does not subclass RenderableView. A "
        "view in a new space should inherit VolumetricView or SurfaceView, or "
        "RenderableView directly if it is sampled some other way."
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
    "Packable",
    "RenderableView",
    "VolumetricView",
    "SurfaceView",
    "Renderable",
    "HasSubject",
    "ColormappedView",
    "as_renderable",
    "space_of",
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
]
