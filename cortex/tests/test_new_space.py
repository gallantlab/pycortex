"""Extending the package with a new brain space or a new view kind.

Split out of ``test_dataset.py``, which covers the six built-in views' behaviour.
Every test here *defines* a space or a view class that does not ship with
pycortex, and asserts that an extension point holds: ``as_renderable`` accepting a
third-party view and rejecting a structural lookalike, a spatial interface
refusing an incomplete subclass, a third spatial kind reaching the flatmap
renderer unchanged, a new space registering ahead of the catch-all, and the
three-class skeleton in ``cortex/dataset/INHERITANCE.md`` working when filled in.

Several of these are cited by name from that document, so renaming one means
editing it too.
"""

import os
import tempfile

import cortex
import numpy as np
import pytest

from cortex import dataset

subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)


def test_as_renderable_rejects_a_view_with_neither_interface():
    """``as_renderable`` is the one boundary; it must reject, not defer.

    Unlike an enumeration of known classes it accepts a view in *any* space, so
    long as it exposes the interface.
    """
    from cortex.dataset.views import as_renderable

    vol = cortex.Volume(np.ones(volshape), subj, xfmname)
    assert as_renderable(vol) is vol

    class Unrenderable(dataset.Dataview):
        """A view exposing neither a volume nor vertices."""

        @property
        def space(self):
            raise NotImplementedError

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    with pytest.raises(TypeError, match="is not renderable"):
        as_renderable(Unrenderable())


def test_as_renderable_accepts_a_third_party_view_that_inherits_a_spatial_interface():
    """Still open, but by explicit opt-in rather than structurally.

    A view in a space this package has never heard of is renderable as soon as it
    inherits whichever spatial interface describes how its data is sampled -- no entry
    and no union to edit.

    Note what it has to implement: ``spatial_data`` and nothing else about its
    array. ``vertices`` is the surface interface's name for that same array and
    arrives concrete, so a third-party view does not publish it twice.
    """
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import SurfaceView

    class ThirdPartyView(SurfaceView):
        @property
        def space(self):
            raise NotImplementedError

        @property
        def subject(self):
            return subj

        _array = np.zeros((1, 10))

        @property
        def spatial_data(self):
            return self._array

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    view = ThirdPartyView()
    assert as_renderable(view) is view
    assert isinstance(view, SurfaceView)
    assert view.vertices is view.spatial_data


def test_as_renderable_rejects_a_lookalike_that_merely_has_the_attributes():
    """The soundness a runtime_checkable Protocol could not provide.

    A Protocol's isinstance tests only for the *presence* of the member names, so
    this class would satisfy one. A nominal base class rejects it.
    """
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import SurfaceView, VolumetricView

    class Lookalike(dataset.Dataview):
        """Every name a spatial interface wants, none of the meanings."""

        subject = 42
        xfmname = None
        volume = "not an array"
        vertices = "not an array either"

        @property
        def space(self):
            raise NotImplementedError

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    fake = Lookalike()
    assert not isinstance(fake, VolumetricView)
    assert not isinstance(fake, SurfaceView)
    with pytest.raises(TypeError, match="is not renderable"):
        as_renderable(fake)


def test_a_spatial_subclass_must_implement_its_interface():
    """Abstract members mean a forgotten accessor fails at construction."""
    from cortex.dataset.views import VolumetricView

    class Incomplete(VolumetricView):
        @property
        def space(self):
            raise NotImplementedError

        @property
        def subject(self):
            return subj

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_a_third_spatial_kind_needs_no_change_to_the_renderer():
    """Nothing branches on which spatial kind a view is, so a new one just works.

    ``make_flatmap_image`` used to fork on ``isinstance(braindata, SurfaceView)``
    with an ``else`` that silently assumed exactly two kinds existed -- a third
    would have taken the volumetric path and then failed on ``.xfmname``. It now
    reads ``space.xfmname`` (what to sample through) and ``spatial_data`` (what to
    sample), so a spatial kind it has never heard of is drawn by the same code.
    """
    import inspect

    from cortex.dataset._space import BrainSpace
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import RenderableView, SurfaceView, VolumetricView
    from cortex.quickflat import utils as qutils

    class ThirdSpace(BrainSpace):
        """A space that is neither volumetric nor surface-shaped."""

        hdf_key = "third"

        @property
        def xfmname(self):
            return None          # sampled without a transform

        def coerce(self, data):
            return np.zeros((1, 4)) if data is None else data

        def is_movie(self, data):
            return data.ndim > 1

        @property
        def spatial_shape(self):
            return (4,)

        def wrap(self, data, **kwargs):
            raise NotImplementedError

        def wrap_rgb(self, red, green, blue, alpha=None, **kwargs):
            raise NotImplementedError

        def to_json(self):
            return {}

        def write_hdf_attrs(self, h5, node):
            return None

        @classmethod
        def from_hdf(cls, attrs, *, subject, xfmname, mask):
            return None

        @classmethod
        def views(cls):
            raise NotImplementedError

    class ThirdSpatialKind(RenderableView):
        """A spatial kind this package has never heard of."""

        @property
        def space(self):
            return self._space

        def __init__(self):
            self._space = ThirdSpace(subj)

        @property
        def spatial_data(self):
            return np.zeros((1, 4))

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    view = ThirdSpatialKind()
    # it is renderable without being either built-in spatial interface
    assert as_renderable(view) is view
    assert not isinstance(view, (VolumetricView, SurfaceView))
    # and the two facts the renderer needs are both available
    assert view.space.xfmname is None
    assert view.spatial_data.shape == (1, 4)
    # the renderer no longer asks which spatial kind it holds. (It still has
    # isinstance checks for np.ma.MaskedArray -- about the array, not the space.)
    src = inspect.getsource(qutils.make_flatmap_image)
    for banned in ("SurfaceView", "VolumetricView", "Volume2D", "Vertex2D"):
        assert banned not in src, banned
    assert "spatial_data" in src and "space.xfmname" in src


def _third_space_family():
    """Build a third space's three view classes exactly as INHERITANCE.md prescribes.

    Returns ``(MySpace, MyView, MyView2D, MyViewRGB, nthings)``. Kept as a helper so
    the skeleton in the doc has exactly one executable counterpart; if the skeleton
    changes, this changes with it.
    """
    from cortex.dataset._space import BrainSpace, SpaceViews
    from cortex.dataset.view2D import Dataview2D, _resolve_2d_channels
    from cortex.dataset.viewRGB import Colors, DataviewRGB, _resolve_rgb_channels
    from cortex.dataset.views import RenderableView, ScalarView

    nthings = 10

    class MySpatial(RenderableView):
        """A third spatial interface: sampled by index, through no transform."""

    class MySpace(BrainSpace):
        hdf_key = "myspace"
        spec_keys = ("myarg",)

        def __init__(self, subject, myarg="a"):
            super().__init__(subject)
            self._myarg = myarg

        @property
        def myarg(self):
            return self._myarg

        @property
        def xfmname(self):
            return None

        def coerce(self, data):
            if data is None:
                data = np.zeros((nthings,))
            if data.shape[-1] != nthings:
                raise ValueError("bad length")
            return data

        def is_movie(self, data):
            return data.ndim > 1

        @property
        def spatial_shape(self):
            return (nthings,)

        def wrap(self, data, **kw):
            return MyView(data, self.subject, self.myarg, **kw)

        def wrap_rgb(self, r, g, b, a=None, **kw):
            return MyViewRGB(r, g, b, self.subject, self.myarg, a, **kw)

        def to_json(self):
            return {}

        def write_hdf_attrs(self, h5, node):
            node.attrs["myarg"] = self.myarg

        @classmethod
        def from_hdf(cls, attrs, *, subject, xfmname, mask):
            if "myarg" not in attrs:
                return None
            return cls(subject, attrs["myarg"])

        @classmethod
        def views(cls):
            return SpaceViews(scalar=MyView, twod=MyView2D, rgb=MyViewRGB)

    class MyView(ScalarView, MySpatial):
        """A scalar view in MySpace."""

        def __init__(self, data, subject, myarg, cmap=None, vmin=None, vmax=None,
                     description="", state=None, **kwargs):
            super().__init__(data, MySpace(subject, myarg), cmap=cmap, vmin=vmin,
                             vmax=vmax, description=description, state=state,
                             **kwargs)
            self._resolve_percentiles()

        # The doc's skeleton also carries a bare `_space: MySpace` annotation
        # here. Omitted: it is a static-only device with no runtime effect, and
        # inside an untyped test helper mypy would only emit an
        # annotation-unchecked note for it.
        @property
        def space(self):
            return self._space

        @property
        def spatial_data(self):
            return self.data if self.movie else self.data[np.newaxis]

        @property
        def raw(self):
            return self._build_raw()

        @classmethod
        def empty(cls, subject, myarg, value=0, **kwargs):
            shape = MySpace(subject, myarg).template_shape
            return cls(cls._sample(shape, value), subject, myarg, **kwargs)

        @classmethod
        def random(cls, subject, myarg, **kwargs):
            shape = MySpace(subject, myarg).template_shape
            return cls(cls._sample(shape, None), subject, myarg, **kwargs)

        def __repr__(self):
            return "<my data for (%s)>" % self.subject

        @property
        def nthings(self):
            return self.space.spatial_shape[0]

    class MyView2D(Dataview2D[MyView], MySpatial):
        """Two MyViews, jointly colormapped."""

        def __init__(self, dim1, dim2, subject=None, myarg=None, description="",
                     cmap=None, vmin=None, vmax=None, vmin2=None, vmax2=None,
                     **kwargs):
            chan1, chan2 = _resolve_2d_channels(
                dim1, dim2, channel_cls=MyView, space_cls=MySpace, subject=subject,
                spec={"myarg": myarg}, ranges=((vmin, vmax), (vmin2, vmax2)))
            super().__init__(chan1, chan2, description=description, cmap=cmap,
                             vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2,
                             **kwargs)

        @property
        def raw(self):
            return super().raw

        def __repr__(self):
            return "<2D my data for (%s)>" % self.dim1.subject

    class MyViewRGB(DataviewRGB[MyView], MySpatial):
        """Three MyView channels plus alpha."""

        def __init__(self, channel1, channel2, channel3, subject=None, myarg=None,
                     alpha=None, description="", state=None,
                     channel1color=Colors.Red, channel2color=Colors.Green,
                     channel3color=Colors.Blue, max_color_value=None,
                     max_color_saturation=1.0, vmin=None, vmax=None,
                     autorange="individual", priority=1):
            red, green, blue, resolved_alpha = _resolve_rgb_channels(
                (channel1, channel2, channel3), channel_cls=MyView,
                space_cls=MySpace, subject=subject, spec={"myarg": myarg},
                colors=(channel1color, channel2color, channel3color),
                max_color_value=max_color_value,
                max_color_saturation=max_color_saturation,
                vmin=vmin, vmax=vmax, autorange=autorange, alpha=alpha)
            super().__init__(red, green, blue, alpha=resolved_alpha, subject=subject,
                             description=description, state=state, priority=priority)

        @property
        def space(self):
            return self.red.space

        def __repr__(self):
            return "<RGB my data for (%s)>" % self.subject

    return MySpace, MyView, MyView2D, MyViewRGB, nthings


def test_the_documented_skeleton_for_a_new_space_actually_works():
    """Every claim the INHERITANCE.md skeleton makes, exercised.

    The doc tells a reader to copy three class skeletons and fill them in. This is
    that, filled in against a synthetic ten-element space, so the doc cannot rot
    into describing an extension point that no longer exists.
    """
    from cortex.dataset.views import as_renderable

    MySpace, MyView, MyView2D, MyViewRGB, nthings = _third_space_family()
    arr = np.random.randn(nthings)

    # --- the scalar view
    view = MyView(arr, subj, "a")
    assert repr(view) == "<my data for (S1)>"
    assert view.subject == subj and view.nthings == nthings
    assert view.space.xfmname is None
    assert view.spatial_data.shape == (1, nthings)          # frame axis added
    assert MyView(np.random.randn(3, nthings), subj, "a").spatial_data.shape == (3, nthings)
    assert view.name.startswith("__") and len(view.name) == 18
    assert np.allclose((view + 1).data, arr + 1)            # inherited operators
    assert view.vmin is not None and view.vmax is not None  # _resolve_percentiles

    # --- empty / random, which read template_shape off the space
    assert np.all(MyView.empty(subj, "a", 2).data == 2)
    assert MyView.random(subj, "a").data.shape == (nthings,)

    # --- raw, which goes out through space.wrap_rgb and comes back concrete
    raw = view.raw
    assert isinstance(raw, MyViewRGB)
    assert raw.spatial_data.shape == (1, nthings, 4)
    assert raw.spatial_data.dtype == np.uint8

    # --- the 2D view, from raw arrays and from built views
    twod = MyView2D(arr, arr, subj, "a")
    assert repr(twod) == "<2D my data for (S1)>"
    assert twod.spatial_data.shape == (1, nthings, 4)
    assert isinstance(twod.raw, MyViewRGB)
    assert isinstance(MyView2D(view, view).dim1, MyView)
    # the shared resolver enforces the space's spec keys
    with pytest.raises(TypeError, match="myarg"):
        MyView2D(arr, arr, subj)

    # --- the RGB view, single frame and movie
    rgb = MyViewRGB(arr, arr, arr, subj, "a")
    assert repr(rgb) == "<RGB my data for (S1)>"
    assert rgb.spatial_data.shape == (1, nthings, 4)
    movie = MyViewRGB(*([np.random.randn(4, nthings)] * 3), subj, "a")
    assert movie.spatial_data.shape == (4, nthings, 4)

    # --- serialization, with the space contributing no layout keys of its own
    assert set(view.to_json(simple=True)) == {"name", "subject", "min", "max"}
    assert "data" in view.to_json(simple=False)
    assert [type(u).__name__ for u in twod.uniques()] == ["MyView", "MyView"]
    assert [type(u).__name__ for u in rgb.uniques(collapse=True)] == ["MyViewRGB"]

    # --- and it is renderable without being either built-in spatial kind
    from cortex.dataset.views import SurfaceView, VolumetricView

    assert as_renderable(view) is view
    assert not isinstance(view, (VolumetricView, SurfaceView))

    # --- HDF round trip, which needs the space registered so that _detect_space
    # can find it. All three columns must come back as their own classes.
    from cortex.dataset import _space as space_mod

    saved = list(space_mod._SPACES)
    try:
        space_mod.register_space(MySpace)
        fname = os.path.join(tempfile.mkdtemp(), "third.hdf")
        cortex.Dataset(scalar=view, twod=twod, rgb=rgb).save(fname)
        back = cortex.load(fname)
        assert {k: type(back[k]).__name__ for k in sorted(back.views)} == {
            "rgb": "MyViewRGB",
            "scalar": "MyView",
            "twod": "MyView2D",
        }
        assert np.allclose(back["scalar"].data, arr)
        assert back["scalar"].space.myarg == "a"      # written by write_hdf_attrs
    finally:
        space_mod._SPACES[:] = saved


def test_a_third_space_registers_ahead_of_the_catch_all():
    """Registration order is by fallback, not by arrival.

    ``register_space`` used to append, so a space registered by a third party --
    necessarily after ``cortex.dataset`` has registered its own two -- landed
    behind ``SurfaceSpace``, whose ``from_hdf`` accepts any node without a
    transform, and was never reached. ``SurfaceSpace`` claimed the node,
    ``wrap`` built a ``Vertex``, ``coerce`` raised on the vertex count, and
    ``cortex.load`` swallowed that per view and returned an empty Dataset.
    """
    from cortex.dataset import _space as space_mod
    from cortex.dataset._space import SurfaceSpace, VolumeSpace

    # only the catch-all declares itself one, and the built-in order is unchanged
    assert SurfaceSpace.fallback and not VolumeSpace.fallback
    assert [c.__name__ for c in space_mod.registered_spaces()] == [
        "VolumeSpace",
        "SurfaceSpace",
    ]

    MySpace = _third_space_family()[0]
    assert not MySpace.fallback
    saved = list(space_mod._SPACES)
    try:
        space_mod.register_space(MySpace)
        after = [c.__name__ for c in space_mod.registered_spaces()]
        assert after == ["VolumeSpace", "MySpace", "SurfaceSpace"]

        # so the new space claims its own node, rather than the catch-all doing it
        claimed = None
        for cls in space_mod.registered_spaces():
            claimed = cls.from_hdf(
                {"myarg": "a"}, subject=subj, xfmname=None, mask=None
            )
            if claimed is not None:
                break
        assert type(claimed) is MySpace

        # and a further fallback still sorts behind every real space
        class LaterFallback(MySpace):
            fallback = True

        space_mod.register_space(LaterFallback)
        assert [c.__name__ for c in space_mod.registered_spaces()][-2:] == [
            "SurfaceSpace",
            "LaterFallback",
        ]
    finally:
        space_mod._SPACES[:] = saved
