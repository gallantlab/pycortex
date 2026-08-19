"""Brain spaces: where a data array lives, and what geometry describes it.

This is the *open* axis of the package. The six public view classes are two
spaces crossed with three channel layouts, but only the channel layout is an
inheritance axis; the space is a component held by every view as ``view.space``.

Adding a new kind of brain data therefore means adding a :class:`BrainSpace`
subclass plus three thin view subclasses that only forward their space keywords.
All the colormapping, HDF, JSON, NaN and alpha logic is inherited.

A space is *per-view*, not shared or cached: it owns the facts that depend on
both the geometry and the particular array bound to it (which mask a flattened
array corresponds to, which hemisphere a half-length array covered). Those are
filled in by :meth:`BrainSpace.coerce`, which every view calls exactly once
during construction.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Optional, Union

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import h5py
import numpy as np
import numpy.typing as npt

from ..database import db
from ._hdf import _find_mask, _hash, _hdf_write
from ._webgl import MosaicTexture, VertexAttributes, WebGLPayload, no_encoding

if TYPE_CHECKING:
    # Annotation-only, so _space stays importable by views.py without a cycle.
    from .views import DataviewJSON, ScalarView

def _require(value: Optional[str], what: str) -> str:
    """Assert that a space-defining argument was supplied.

    Lives here rather than beside the views because what is mandatory is a
    property of the space: :mod:`~cortex.dataset.view2D` and
    :mod:`~cortex.dataset.viewRGB` each had their own copy, worded differently,
    and applied it by hand to the arguments of whichever space class they were
    about to construct.
    """
    if value is None:
        raise TypeError("%s must be specified with raw data" % what)
    return value


MaskSpec = Union[str, npt.NDArray[np.bool_], None]
"""What the user passed as ``mask=``: a database mask name, an explicit boolean
array, or nothing. Recorded verbatim so it can be round-tripped through HDF."""


class SpaceViews(NamedTuple):
    """The three concrete view classes belonging to one space.

    Used when rebuilding a view from HDF, where the column is known from the
    record's shape but the class for it is not: ``twod`` in
    :func:`~cortex.dataset.views._from_hdf_view`, ``rgb`` in both that and
    :func:`~cortex.dataset.views._from_hdf_data`.
    """

    #: The scalar view class. Unlike its siblings, nothing reads this: every path
    #: that needs a scalar view calls :meth:`BrainSpace.wrap`, which builds one
    #: without having to name the class. Declared anyway so the triple describes a
    #: space's full column set rather than just the parts one caller happens to
    #: need, and so ``views()`` stays the single answer to "which classes are
    #: mine?".
    scalar: type
    #: The 2D view class, read when reconstructing a 2D view from HDF.
    twod: type
    #: The RGB view class, read when reconstructing an RGB view from HDF.
    rgb: type


class BrainSpace(ABC):
    """Where a data array lives: a subject plus the geometry to interpret it."""

    #: A stable, human-readable label for this kind of space.
    #:
    #: Nothing in the package reads it. Detection on load does not use it either,
    #: despite the name: :func:`~cortex.dataset.views._detect_space` walks
    #: :func:`registered_spaces` in order -- fallbacks last, see
    #: :func:`register_space` -- and takes the first space
    #: whose :meth:`from_hdf` returns non-``None``, which is why the built-ins key
    #: off whether a transform name is present rather than off any stored key.
    #:
    #: It is kept as the one place a space states its own name, so that a space
    #: which *does* want a discriminator on disk has an obvious value to write in
    #: :meth:`write_hdf_attrs` and match in :meth:`from_hdf` -- neither built-in
    #: needs one, since legacy files predate the idea and carry no such key.
    hdf_key: ClassVar[str]

    fallback: ClassVar[bool] = False
    """Whether this space claims any node no other space wanted.

    A fallback exists only because legacy files carry no space discriminator:
    :class:`SurfaceSpace` accepts anything without a transform, which is how a
    pre-registry file is recognised. Set it and :func:`register_space` keeps you
    behind every non-fallback space, however many are added later. Leave it False
    -- a new space should test in :meth:`from_hdf` for something it writes itself
    in :meth:`write_hdf_attrs`, and will then be consulted ahead of the catch-all.
    """

    spec_keys: ClassVar[tuple[str, ...]] = ()
    """Constructor arguments besides ``subject`` that identify this space.

    Read by :meth:`from_spec`, so that "a volumetric space is subject plus
    xfmname, and both are mandatory" is stated once here rather than in each of
    the four composite view constructors that had to build one.
    """

    def __init__(self, subject: Union[str, bytes]) -> None:
        self.subject = (
            subject if isinstance(subject, str) else subject.decode("utf-8")
        )

    @property
    @abstractmethod
    def xfmname(self) -> Optional[str]:
        """Transform name for this space, or None if it has no transform.

        The single fact that distinguishes "sample through a transform" from
        "sample on a surface". Both the flatmap cache and HDF slot 7 derive from
        it, so neither has to ask which kind of space it is holding.
        """

    # ------------------------------------------------------------------
    # binding an array
    # ------------------------------------------------------------------
    @classmethod
    def from_spec(cls, subject: Optional[str], **spec: Any) -> Self:
        """Build this space from the arguments a view constructor was handed.

        The composite views need to construct a space when they are given raw
        arrays rather than channel objects. Each used to be passed a ``lambda``
        naming the space class with :func:`_require` applied to each of its
        arguments, *and* a dict of those same keys to validate channel objects
        against -- the same knowledge spelled out twice per constructor, four
        times over.
        """
        name = _require(subject, "Subject")
        for key in cls.spec_keys:
            _require(spec.get(key), key)
        return cls(name, **spec)

    @abstractmethod
    def coerce(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Validate ``data`` against this space and return it, possibly padded.

        ``None`` means "an all-zero array for this geometry", which
        ``cortex.rois`` relies on to avoid an expensive initialisation.

        Called once per view. Implementations may record data-dependent geometry
        on ``self`` (which mask a flat array matches, which hemisphere a
        half-length array covered) -- a space belongs to exactly one view.
        """

    def align(
        self, first: "ScalarView", second: "ScalarView"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Two views' arrays in a layout where position *i* means the same place.

        What a 2D view needs before it can colormap its two dimensions jointly,
        and what any elementwise operation between two views in this space would
        need. The stored arrays serve for any space in which one array position is
        one location, which is why this is concrete; :class:`VolumeSpace`
        overrides it because a flattened array's positions mean something only
        relative to its own mask.

        Raise if the two cannot be aligned at all.
        """
        return first.data, second.data

    @abstractmethod
    def is_movie(self, data: npt.NDArray) -> bool:
        """Whether ``data`` carries a leading time axis in this space."""

    @property
    @abstractmethod
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of a single frame's worth of data in this space.

        Describes an array that has already been bound by :meth:`coerce`, so it
        may be unset on a freshly constructed space. For "what shape should a new
        array be", use :attr:`template_shape`.
        """

    @property
    def template_shape(self) -> tuple[int, ...]:
        """Shape a *fresh* single frame should have, before any data exists.

        This is what :meth:`~cortex.dataset.views.ScalarView.empty` and
        ``random`` need, and it is not the same question as
        :attr:`spatial_shape`: that one reports the geometry of an array already
        bound to this space. For a surface the two coincide, because the vertex
        count comes from the database the moment the space is built. A volume
        only learns its shape from the array it is given, so this default is
        wrong for it and :class:`VolumeSpace` overrides it with a lookup.

        Concrete rather than abstract so that adding a space costs nothing when
        its geometry is known up front, which is the common case.
        """
        return self.spatial_shape

    @abstractmethod
    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        """Build a scalar view over ``data`` in a space like this one.

        Uses ``self`` only as a template for the space's parameters; the new view
        gets its own space, since :meth:`coerce` records per-array state.

        This is what makes the composite views and the HDF factories
        space-agnostic: they never name ``Volume`` or ``Vertex``.
        """

    @abstractmethod
    def wrap_rgb(
        self,
        red: npt.NDArray,
        green: npt.NDArray,
        blue: npt.NDArray,
        alpha: Optional[npt.NDArray] = None,
        **kwargs: Any,
    ) -> Any:
        """Build an RGB view over three channel arrays in a space like this one.

        The counterpart of :meth:`wrap`. Lets a scalar view render itself to RGB
        without naming a concrete class, so the implementation lives in one place
        instead of once per space.
        """

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    @abstractmethod
    def to_json(self) -> DataviewJSON:
        """Space-specific keys to merge into a view's JSON description."""

    def describe_layout(self, data: npt.NDArray) -> DataviewJSON:
        """Keys telling the browser how to read the array it is being sent.

        Merged into ``to_json(simple=True)``, alongside the geometry-independent
        ``name``/``subject``/``min``/``max``. Separate from :meth:`to_json`
        because that one describes the *space* (a transform, say) while this
        describes the particular array bound to it, and only the latter needs the
        array. ``webgl/resources/js/dataset.js`` reads these positionally by name,
        so the keys are a hard interface.

        Empty by default: a space whose arrays need no unpacking hint says
        nothing, and the browser falls back to treating the array as flat.
        """
        from .views import DataviewJSON as _JSON

        return _JSON()

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> WebGLPayload:
        """This space's arrays encoded for the browser.

        The single decision behind what used to be four forks in
        ``webgl/data.py``: the dtype cast, whether alpha is premultiplied, whether
        frames are mosaicked into PNGs or shipped as per-vertex attributes, and
        whether they are permuted into the CTM's vertex order. All four follow
        from the geometry, which is why they belong here.

        ``raw`` says the array is 4-channel uint8 from an RGB view rather than
        scalar floats. Return one of the two encodings in
        :mod:`cortex.dataset._webgl`; a space wanting a third has to add a
        matching branch to ``webgl/resources/js/dataset.js``.

        Not abstract, because a space with no browser representation is a
        legitimate thing to have -- ``quickflat`` needs only ``spatial_data`` --
        and the default says so with a message naming both encodings.
        """
        return no_encoding(self)

    @abstractmethod
    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        """Write whatever this space needs to reconstruct itself onto ``node``."""

    @property
    def view_xfmname(self) -> Any:
        """Value for slot 7 of a ``/views`` record.

        Concrete: derived from :attr:`xfmname` rather than overridden per space,
        which is what it used to be.
        """
        return None if self.xfmname is None else [self.xfmname]

    @classmethod
    @abstractmethod
    def from_hdf(
        cls,
        attrs: dict[str, Any],
        *,
        subject: str,
        xfmname: Optional[str],
        mask: MaskSpec,
    ) -> Optional[Self]:
        """Build a space from HDF node attributes, or None if this isn't the one.

        Consulted in registration order by
        :func:`cortex.dataset.views._detect_space`. Legacy files carry no space
        discriminator, so the built-in implementations key off whether an
        ``xfmname`` is present; a new space should test for something it writes
        itself in :meth:`write_hdf_attrs` and register ahead of the built-ins.
        """

    @classmethod
    @abstractmethod
    def views(cls) -> SpaceViews:
        """The scalar, 2D and RGB view classes for this space."""


_SPACES: list[type[BrainSpace]] = []


def register_space(space: type[BrainSpace]) -> type[BrainSpace]:
    """Register a space so the HDF factories and ``normalize`` can find it.

    Order matters: :func:`registered_spaces` is consulted in order and the first
    space whose :meth:`BrainSpace.from_hdf` returns non-None wins. A space is
    inserted ahead of every :attr:`BrainSpace.fallback` space, so the catch-alls
    stay last however many spaces are added.

    This used to append, which meant a space registered by a third party -- and
    that is necessarily after ``cortex.dataset`` has registered its own two --
    landed *behind* ``SurfaceSpace``, whose ``from_hdf`` accepts any node without
    a transform. It was therefore never reached: ``SurfaceSpace`` claimed the
    node, ``wrap`` built a ``Vertex``, ``coerce`` raised on the vertex count, and
    ``cortex.load`` swallowed that per view and returned an empty Dataset. The
    append semantics were never exercised, because both built-in spaces are
    registered here in an order chosen by hand.
    """
    if space.fallback:
        _SPACES.append(space)
        return space
    for i, existing in enumerate(_SPACES):
        if existing.fallback:
            _SPACES.insert(i, space)
            return space
    _SPACES.append(space)
    return space


def registered_spaces() -> list[type[BrainSpace]]:
    return list(_SPACES)


@register_space
class VolumeSpace(BrainSpace):
    """Voxel data under a pycortex transform.

    Parameters
    ----------
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    mask : str or ndarray, optional
        A database mask name, or a boolean 3D array. Only meaningful for
        flattened (masked) data; ignored for full 3D/4D volumes. If omitted for
        flattened data, a mask with a matching voxel count is looked up in the
        database.
    """

    hdf_key = "volume"
    spec_keys = ("xfmname",)

    def __init__(
        self,
        subject: Union[str, bytes],
        xfmname: Union[str, bytes],
        mask: MaskSpec = None,
    ) -> None:
        super().__init__(subject)
        # A property, not a plain attribute: an abstract property is not
        # satisfied by an instance attribute assigned in __init__.
        self._xfmname = (
            xfmname if isinstance(xfmname, str) else xfmname.decode("utf-8")
        )

        #: Verbatim record of what was passed as ``mask=``, for round-tripping.
        self.mask_spec: MaskSpec = mask
        #: The resolved boolean mask, or None for unmasked (3D/4D) data.
        self.mask: Optional[npt.NDArray[np.bool_]] = None
        #: The database mask name, when the mask came from (or was found in) the db.
        self.mask_name: Optional[str] = None
        #: Whether the bound array is flattened into mask space.
        self.linear = False
        self._shape: tuple[int, ...] = ()

    @property
    def xfmname(self) -> str:
        """Narrowed from ``Optional[str]``: a volume always has a transform."""
        return self._xfmname

    def coerce(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        if data is None:
            raise TypeError("Volumetric data cannot be None")
        if data.ndim not in (1, 2, 3, 4):
            raise ValueError("Invalid data shape")

        self.linear = data.ndim in (1, 2)
        mask = self.mask_spec

        if self.linear:
            if mask is None:
                # Guess the mask from the voxel count.
                nvox: int = data.shape[-1]
                found_name, found_mask = _find_mask(nvox, self.subject, self.xfmname)
                self.mask_name = found_name
                self.mask = found_mask
                self.mask_spec = found_name
            elif isinstance(mask, np.ndarray):
                self.mask = mask > 0
                self.mask_spec = mask > 0
            else:
                self.mask = db.get_mask(self.subject, self.xfmname, mask)
                self.mask_name = mask
                self.mask_spec = mask

            assert self.mask is not None
            self._shape = self.mask.shape
        else:
            self.mask_spec = None
            shape = data.shape
            if self.is_movie(data):
                shape = shape[1:]
            xfm = db.get_xfm(self.subject, self.xfmname)
            if xfm.shape != shape:
                raise ValueError(
                    "Volumetric data (shape %s) is not the same shape as reference "
                    "for transform (shape %s)" % (str(shape), str(xfm.shape))
                )
            self._shape = shape

        return data

    def is_movie(self, data: npt.NDArray) -> bool:
        # (t, v) flattened or (t, z, y, x) volumetric
        return data.ndim in (2, 4)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def template_shape(self) -> tuple[int, ...]:
        """The reference volume's shape, from the transform.

        Unlike :attr:`spatial_shape` this does not wait for data, so it costs a
        database lookup. A masked volume still reports the full 3-D shape here:
        callers building a new array want the unmasked geometry, and
        :meth:`coerce` flattens it afterwards if a mask is in play.
        """
        return db.get_xfm(self.subject, self.xfmname).shape

    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        from .views import Volume

        # Passing the resolved mask spec along means a flattened array does not
        # have to be matched against the database again.
        return Volume(
            data, self.subject, self.xfmname, mask=self.mask_spec, **kwargs
        )

    def wrap_rgb(
        self,
        red: npt.NDArray,
        green: npt.NDArray,
        blue: npt.NDArray,
        alpha: Optional[npt.NDArray] = None,
        **kwargs: Any,
    ) -> Any:
        from .viewRGB import VolumeRGB

        return VolumeRGB(
            red, green, blue, self.subject, self.xfmname, alpha, **kwargs
        )

    def align(
        self, first: "ScalarView", second: "ScalarView"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Stored arrays if both are flattened under the same mask, else volumes.

        Two arrays flattened under the *same* mask already line up, and are far
        smaller than the volumes they came from, so prefer them; under different
        masks -- or with one flattened and one not -- position *i* means different
        voxels in each, and only the unmasked volumes are comparable.

        This lived in ``Volume2D.raw`` as fifteen lines against ``Vertex2D.raw``'s
        two, the whole difference being about masks. Masks are volumetric
        knowledge, and the rest of it is already here.
        """
        other = second.space
        if not isinstance(other, VolumeSpace) or other.xfmname != self.xfmname:
            raise ValueError(
                "Both Volumes must have same xfmname to generate single raw volume"
            )
        if (
            self.linear
            and other.linear
            and self.mask is not None
            and other.mask is not None
            and self.mask.shape == other.mask.shape
            and np.all(self.mask == other.mask)
        ):
            return first.data, second.data
        return first.spatial_data, second.spatial_data

    def unmask(self, data: npt.NDArray, movie: bool) -> npt.NDArray:
        """Expand ``data`` to a full 3D/4D volume, adding the time axis."""
        from cortex import volume

        if self.linear:
            assert self.mask is not None
            expanded = volume.unmask(self.mask, data[:])
        else:
            expanded = data[:]

        if not movie:
            expanded = expanded[np.newaxis]
        return expanded

    def to_json(self) -> DataviewJSON:
        from .views import DataviewJSON as _JSON

        xfm = db.get_xfm(self.subject, self.xfmname, "coord").xfm
        return _JSON(xfm=[list(np.array(xfm).ravel())])

    def describe_layout(self, data: npt.NDArray) -> DataviewJSON:
        """The 3-D grid the mosaic tiles unpack back into."""
        from .views import DataviewJSON as _JSON

        return _JSON(shape=self.spatial_shape)

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> WebGLPayload:
        """A mosaicked PNG texture, which the shader samples through the transform."""
        return MosaicTexture(data, raw=raw)

    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        if self.mask_spec is None:
            return

        mask: Any = self.mask_spec
        if isinstance(self.mask_spec, np.ndarray):
            mgrp = "/subjects/{subj}/transforms/{xfm}/masks/".format(
                subj=self.subject, xfm=self.xfmname
            )
            mname = "__%s" % _hash(self.mask_spec)[:8]
            _hdf_write(h5, self.mask_spec, name=mname, group=mgrp)
            mask = mname

        node.attrs["mask"] = mask

    @classmethod
    def from_hdf(
        cls,
        attrs: dict[str, Any],
        *,
        subject: str,
        xfmname: Optional[str],
        mask: MaskSpec,
    ) -> Optional[Self]:
        if xfmname is None:
            return None
        return cls(subject, xfmname, mask=mask)

    @classmethod
    def views(cls) -> SpaceViews:
        from .view2D import Volume2D
        from .viewRGB import VolumeRGB
        from .views import Volume

        return SpaceViews(scalar=Volume, twod=Volume2D, rgb=VolumeRGB)

    def __repr__(self) -> str:
        return "<VolumeSpace(%s, %s)>" % (self.subject, self.xfmname)


@register_space
class SurfaceSpace(BrainSpace):
    """Vertex data on a subject's cortical surface.

    Registered last, because it accepts any array that has no transform -- which
    is how legacy HDF files, that carry no space discriminator, are detected.

    Parameters
    ----------
    subject : str
        Subject identifier. Must exist in the pycortex database.
    """

    hdf_key = "surface"
    #: Claims any node without a transform, which is how legacy files -- written
    #: before spaces existed, so carrying no discriminator -- are recognised.
    fallback = True

    def __init__(self, subject: Union[str, bytes]) -> None:
        super().__init__(subject)
        try:
            left, right = db.get_surf(self.subject, "wm")
        except IOError:
            left, right = db.get_surf(self.subject, "fiducial")
        self.llen = len(left[0])
        self.rlen = len(right[0])
        #: Which hemispheres the bound array covered: "left", "right" or "both".
        self.hem = "both"

    @property
    def xfmname(self) -> None:
        """A surface space has no transform."""
        return None

    @property
    def nverts(self) -> int:
        return self.llen + self.rlen

    def coerce(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Pad single-hemisphere data with zeros for the other hemisphere."""
        if data is None:
            data = np.zeros((self.nverts,))

        movie = self.is_movie(data)
        given = data.shape[-1]
        if self.llen == given:
            self.hem = "left"
            rshape = list(data.shape)
            rshape[1 if movie else 0] = self.rlen
            return np.hstack([data, np.zeros(rshape, dtype=data.dtype)])
        if self.rlen == given:
            self.hem = "right"
            lshape = list(data.shape)
            lshape[1 if movie else 0] = self.llen
            return np.hstack([np.zeros(lshape, dtype=data.dtype), data])
        if self.nverts == given:
            self.hem = "both"
            return data
        raise ValueError(
            "Invalid number of vertices for subject (given %d, should be %d for "
            "left hem, %d for right hem, or %d for both)"
            % (given, self.llen, self.rlen, self.nverts)
        )

    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        from .views import Vertex

        return Vertex(data, self.subject, **kwargs)

    def wrap_rgb(
        self,
        red: npt.NDArray,
        green: npt.NDArray,
        blue: npt.NDArray,
        alpha: Optional[npt.NDArray] = None,
        **kwargs: Any,
    ) -> Any:
        from .viewRGB import VertexRGB

        return VertexRGB(red, green, blue, self.subject, alpha, **kwargs)

    def is_movie(self, data: npt.NDArray) -> bool:
        # (t, v) versus (v,)
        return data.ndim > 1

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return (self.nverts,)

    def to_json(self) -> DataviewJSON:
        from .views import DataviewJSON as _JSON

        return _JSON()

    def split_hemispheres(
        self, data: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(left, right)`` views of ``data``, cut where the hemispheres meet.

        Where that boundary is, and which axis it lies along, is the space's
        knowledge: :attr:`llen` vertices of left hemisphere followed by
        :attr:`rlen` of right. The vertex axis is the only one for a plain array
        and the second otherwise, which covers a scalar movie ``(t, v)`` as well
        as the ``(frames, v, 4)`` an RGB view ships -- so one rule serves both,
        and the returned slices are views, not copies.
        """
        if data.ndim > 1:
            return data[:, : self.llen], data[:, self.llen :]
        return data[: self.llen], data[self.llen :]

    def describe_layout(self, data: npt.NDArray) -> DataviewJSON:
        """Where the two hemispheres meet, and how many frames there are.

        ``split`` is the index the browser cuts the vertex array at to get one
        buffer per hemisphere. ``frames`` has to come from the array rather than
        the space, since one space serves movies and single frames alike.
        """
        from .views import DataviewJSON as _JSON

        frames = data.shape[0] if self.is_movie(data) else 1
        return _JSON(split=self.llen, frames=frames)

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> WebGLPayload:
        """Raw per-vertex attributes, premultiplied and permuted into CTM order."""
        return VertexAttributes(data, raw=raw)

    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        return None

    @classmethod
    def from_hdf(
        cls,
        attrs: dict[str, Any],
        *,
        subject: str,
        xfmname: Optional[str],
        mask: MaskSpec,
    ) -> Optional[Self]:
        if xfmname is not None:
            return None
        return cls(subject)

    @classmethod
    def views(cls) -> SpaceViews:
        from .view2D import Vertex2D
        from .viewRGB import VertexRGB
        from .views import Vertex

        return SpaceViews(scalar=Vertex, twod=Vertex2D, rgb=VertexRGB)

    def __repr__(self) -> str:
        return "<SurfaceSpace(%s)>" % (self.subject,)
