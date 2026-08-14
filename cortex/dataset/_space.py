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

if TYPE_CHECKING:
    # Annotation-only, so _space stays importable by views.py without a cycle.
    from .views import DataviewJSON

MaskSpec = Union[str, npt.NDArray[np.bool_], None]
"""What the user passed as ``mask=``: a database mask name, an explicit boolean
array, or nothing. Recorded verbatim so it can be round-tripped through HDF."""


class SpaceViews(NamedTuple):
    """The three concrete view classes belonging to one space."""

    scalar: type
    twod: type
    rgb: type


class BrainSpace(ABC):
    """Where a data array lives: a subject plus the geometry to interpret it."""

    #: Stable identifier for this space, used by the HDF detection order.
    hdf_key: ClassVar[str]

    def __init__(self, subject: Union[str, bytes]) -> None:
        self.subject = (
            subject if isinstance(subject, str) else subject.decode("utf-8")
        )

    # ------------------------------------------------------------------
    # binding an array
    # ------------------------------------------------------------------
    @abstractmethod
    def coerce(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Validate ``data`` against this space and return it, possibly padded.

        ``None`` means "an all-zero array for this geometry", which
        ``cortex.rois`` relies on to avoid an expensive initialisation.

        Called once per view. Implementations may record data-dependent geometry
        on ``self`` (which mask a flat array matches, which hemisphere a
        half-length array covered) -- a space belongs to exactly one view.
        """

    @abstractmethod
    def is_movie(self, data: npt.NDArray) -> bool:
        """Whether ``data`` carries a leading time axis in this space."""

    @property
    @abstractmethod
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of a single frame's worth of data in this space."""

    @abstractmethod
    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        """Build a scalar view over ``data`` in a space like this one.

        Uses ``self`` only as a template for the space's parameters; the new view
        gets its own space, since :meth:`coerce` records per-array state.

        This is what makes the composite views and the HDF factories
        space-agnostic: they never name ``Volume`` or ``Vertex``.
        """

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    @abstractmethod
    def to_json(self) -> DataviewJSON:
        """Space-specific keys to merge into a view's JSON description."""

    @abstractmethod
    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        """Write whatever this space needs to reconstruct itself onto ``node``."""

    @property
    @abstractmethod
    def view_xfmname(self) -> Any:
        """Value for slot 7 of a ``/views`` record, or None if not applicable."""

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

    Order matters: :func:`registered_spaces` is consulted in registration
    order and the first space whose :meth:`BrainSpace.from_hdf` returns
    non-None wins. ``SurfaceSpace`` is deliberately last, since it accepts
    anything without a transform.
    """
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

    def __init__(
        self,
        subject: Union[str, bytes],
        xfmname: Union[str, bytes],
        mask: MaskSpec = None,
    ) -> None:
        super().__init__(subject)
        self.xfmname = (
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

    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        from .views import Volume

        # Passing the resolved mask spec along means a flattened array does not
        # have to be matched against the database again.
        return Volume(
            data, self.subject, self.xfmname, mask=self.mask_spec, **kwargs
        )

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

    @property
    def view_xfmname(self) -> Any:
        return [self.xfmname]

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

    def is_movie(self, data: npt.NDArray) -> bool:
        # (t, v) versus (v,)
        return data.ndim > 1

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return (self.nverts,)

    def to_json(self) -> DataviewJSON:
        from .views import DataviewJSON as _JSON

        return _JSON()

    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        return None

    @property
    def view_xfmname(self) -> Any:
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
