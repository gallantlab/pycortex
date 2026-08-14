from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import h5py
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Colormap

from .. import options
from ..database import db
from ._hdf import _find_mask, _hash, _hdf_write

default_cmap = options.config.get("basic", "default_cmap")


def register_cmap(cmap: Colormap) -> None:
    """Register a colormap with matplotlib.

    Requires matplotlib >= 3.5 for ``matplotlib.colormaps``. The old
    ``matplotlib.cm.register_cmap`` was deprecated in 3.7 and removed in 3.9, so
    the fallback that used to sit here could no longer fire on any matplotlib
    that this package can otherwise import.
    """
    from matplotlib import colormaps

    colormaps.register(cmap)


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

MaskSpec = Union[str, npt.NDArray[np.bool_], None]
"""What the user passed as ``mask=``: a database mask name, an explicit boolean
array, or nothing. Recorded verbatim so it can be round-tripped through HDF."""


class ColormapDict(TypedDict):
    cmap: Colormap
    vmin: Optional[float]
    vmax: Optional[float]


class DataviewJSON(TypedDict, total=False):
    """The wire format consumed by ``webgl/resources/js/dataset.js``.

    Every key is optional because which ones are present depends on the view
    kind and on ``simple=``. The JS dispatches on the *shape* of these values --
    ``mosaic`` absent means surface data, a nested list in ``data`` means a 2D
    view, ``raw`` true means 4-channel uint8 -- so the shapes are load-bearing.
    """

    state: Any
    attrs: dict[str, Any]
    desc: str
    cmap: Optional[list[Any]]
    vmin: Optional[list[Any]]
    vmax: Optional[list[Any]]
    name: str
    subject: str
    min: float
    max: float
    raw: bool
    mosaic: tuple[int, int]
    shape: tuple[int, ...]
    data: list[Any]
    xfm: list[Any]
    split: int
    frames: int


def u(s, encoding: str = "utf8"):
    try:
        return s.decode(encoding)
    except AttributeError:
        return s


def _build_cmapdict(
    cmap: Any, vmin: Optional[float], vmax: Optional[float]
) -> ColormapDict:
    """Resolve a colormap name (matplotlib's or pycortex's) into ``imshow`` kwargs.

    Shared by the scalar and 2D views. RGB views carry their own colours and so
    return an empty mapping from :meth:`Dataview.get_cmapdict` instead.
    """
    from matplotlib import colors
    from matplotlib import pyplot as plt

    try:
        # plt.get_cmap accepts:
        # - matplotlib colormap names
        # - pycortex colormap names previously registered in matplotlib
        # - matplotlib.colors.Colormap instances
        resolved = plt.get_cmap(cmap)
    except ValueError:
        # unknown colormap, test whether it's in pycortex colormaps
        cmapdir = options.config.get("webgl", "colormaps")
        colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
        available = {os.path.split(c)[1][:-4]: c for c in colormaps}
        if cmap not in available:
            raise ValueError("Unknown color map %s" % cmap)
        I = plt.imread(available[cmap])
        name = cmap if isinstance(cmap, str) else cmap.name
        resolved = colors.ListedColormap(np.squeeze(I), name=name)
        # Register colormap to matplotlib to avoid loading it again
        register_cmap(resolved)

    return ColormapDict(cmap=resolved, vmin=vmin, vmax=vmax)


class Dataview(ABC):
    """Abstract root of every displayable view.

    Holds only what *every* view has: a subject, display metadata, and the
    ability to render itself to an RGB view. Deliberately does **not** hold
    ``cmap``/``vmin``/``vmax`` -- RGB views have no single colormap, and the
    previous design's workaround was to skip ``Dataview.__init__`` entirely and
    let ``except AttributeError`` carry the control flow.
    """

    #: NaN positions captured before a scalar view was converted to uint8 RGB.
    #: uint8 cannot hold NaN, so this is the side channel that keeps
    #: "NaN implies transparent" working. Only set by ``ScalarView.raw``.
    _nan_mask: Optional[npt.NDArray[np.bool_]] = None

    def __init__(self, description: str = "", state: Any = None, **kwargs: Any) -> None:
        self.state = state
        self.attrs: dict[str, Any] = dict(kwargs)
        if "priority" not in self.attrs:
            self.attrs["priority"] = 1
        self.description = description

    @property
    @abstractmethod
    def subject(self) -> str:
        """Subject identifier. Must exist in the pycortex database."""

    @property
    @abstractmethod
    def raw(self) -> DataviewRGB:
        """This view rendered to 8-bit RGBA channels."""

    @abstractmethod
    def uniques(self, collapse: bool = False) -> Iterator[Dataview]:
        """Yield the distinct data-carrying views inside this one."""

    @property
    def priority(self) -> Any:
        return self.attrs["priority"]

    @priority.setter
    def priority(self, value: Any) -> None:
        self.attrs["priority"] = value

    def get_cmapdict(self) -> Mapping[str, Any]:
        """Colormap arguments suitable for splatting into ``imshow``.

        Empty for views that carry their own colors (RGB); overridden by the
        views that are colormapped.
        """
        return {}

    def to_json(self, simple: bool = False) -> DataviewJSON:
        if simple:
            return DataviewJSON()

        desc = self.description
        if isinstance(desc, bytes):
            desc = desc.decode()
        return DataviewJSON(state=self.state, attrs=self.attrs.copy(), desc=desc)

    # ------------------------------------------------------------------
    # HDF5
    # ------------------------------------------------------------------
    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        """Fill slots 2-4 (cmap, vmin, vmax) of a ``/views`` node.

        The base implementation leaves them as the ``"null"`` written by
        :meth:`_write_view_node`, which is what RGB views need.
        """

    def _write_view_node(
        self,
        h5: Union[h5py.File, h5py.Group],
        name: str = "data",
        data: Any = None,
        xfmname: Any = None,
    ) -> h5py.Dataset:
        """Write the 8-slot ``/views/<name>`` record.

        The slot layout is a hard interface: ``webgl/resources/js/dataset.js``
        reads it structurally (a nested list in slot 0 means a 2D view, a nested
        list in slot 3 means 2D ranges, and so on). Do not reorder or retype.
        """
        views = h5.require_group("/views")
        view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
        view[0] = json.dumps(data)
        view[1] = self.description
        view[2] = "null"
        view[3:5] = "null"
        self._write_cmap_slots(view)
        view[5] = json.dumps(self.state)
        view[6] = json.dumps(self.attrs)
        view[7] = json.dumps(xfmname)
        return view

    @abstractmethod
    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        """Write both the data node(s) and the view node for this view."""

    @staticmethod
    def from_hdf(node: h5py.Dataset, subject: Optional[str] = None) -> Dataview:
        data = json.loads(u(node[0]))
        desc = node[1]
        try:
            cmap = json.loads(u(node[2]))
        except ValueError:
            cmap = u(node[2])
        vmin = json.loads(u(node[3]))
        vmax = json.loads(u(node[4]))
        state = json.loads(u(node[5]))
        attrs = json.loads(u(node[6]))
        try:
            xfmname = json.loads(u(node[7]))
        except ValueError:
            xfmname = None

        if not isinstance(vmin, list):
            vmin = [vmin]
        if not isinstance(vmax, list):
            vmax = [vmax]
        if not isinstance(cmap, list):
            cmap = [cmap]

        if len(data) != 1:
            # Multiview was never implemented; the old code built a `views`
            # list here and then unconditionally raised.
            raise NotImplementedError(
                "Views containing more than one dataview are not supported"
            )

        xfm = None if xfmname is None else xfmname[0]
        return _from_hdf_view(
            node.file,
            data[0],
            xfmname=xfm,
            cmap=cmap[0],
            description=desc,
            vmin=vmin[0],
            vmax=vmax[0],
            state=state,
            subject=subject,
            **attrs,
        )


class ScalarView(Dataview):
    """A single array of scalar values, displayed through a 1D colormap.

    This is the union of what used to be ``BrainData`` and the colormapped half
    of ``Dataview``. Those were separate classes joined only by multiple
    inheritance in ``Volume``/``Vertex``, which is what made ``super()`` calls in
    ``BrainData`` resolve to methods that were nowhere in its own ancestry.
    """

    #: Whether the data array carries a leading time axis. Set by the concrete
    #: subclass once it knows how to interpret the array's dimensionality.
    movie: bool

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(data, str):
            import nibabel

            nib = cast(nibabel.Nifti1Image, nibabel.load(data))
            data = cast(npt.NDArray, nib.get_fdata().T)
        self._data = data
        self._subject = subject if isinstance(subject, str) else subject.decode("utf-8")
        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(description=description, state=state, **kwargs)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    @property
    def subject(self) -> str:
        return self._subject

    @property
    def data(self) -> npt.NDArray:
        if isinstance(self._data, h5py.Dataset):
            return self._data[()]
        return cast(npt.NDArray, self._data)

    @data.setter
    def data(self, data: npt.NDArray) -> None:
        self._data = data

    @property
    def name(self) -> str:
        """Content-addressed name, used as the HDF node name."""
        return "__%s" % _hash(self.data)[:16]

    def __hash__(self) -> int:
        return hash(_hash(self.data))

    def uniques(self, collapse: bool = False) -> Iterator[Dataview]:
        yield self

    @abstractmethod
    def copy(self, data: npt.NDArray) -> Self:
        """A new view of the same kind, over ``data``, sharing this one's space."""

    def exp(self) -> Self:
        """Return copy of this brain data with data exponentiated."""
        return self.copy(np.exp(self.data))

    def _set_display_params(self, other: ScalarView) -> None:
        other.cmap = self.cmap
        other.vmin = self.vmin
        other.vmax = self.vmax

    def _resolve_percentiles(self) -> None:
        """Fill unset ``vmin``/``vmax`` from the 1st/99th data percentiles.

        Keeps ``np.percentile``'s ``np.float64`` rather than converting to a
        Python ``float``. Under NEP 50 a numpy scalar is a "strong" operand, so
        ``float32_channel -= vmin`` computes in float64 and rounds once, whereas
        a weak Python float computes in float32. The difference is a single LSB
        on a handful of voxels, but channel names are content hashes, so it would
        silently change on-disk node identity. ``np.float64`` subclasses
        ``float``, so the annotation still holds.
        """
        if self.vmin is None:
            self.vmin = np.percentile(np.nan_to_num(self.data), 1)
        if self.vmax is None:
            self.vmax = np.percentile(np.nan_to_num(self.data), 99)

    # ------------------------------------------------------------------
    # numpy operators
    #
    # These used to be generated by ``BrainData._add_numpy_methods`` with
    # ``setattr`` at import time, which made ``vol + 1`` unresolvable to any
    # static tool -- and gave ``__neg__``/``__abs__`` the same binary signature
    # as the rest.
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> Self:
        return self.copy(self.data + other)

    def __sub__(self, other: Any) -> Self:
        return self.copy(self.data - other)

    def __mul__(self, other: Any) -> Self:
        return self.copy(self.data * other)

    def __floordiv__(self, other: Any) -> Self:
        return self.copy(self.data // other)

    def __truediv__(self, other: Any) -> Self:
        return self.copy(self.data / other)

    def __pow__(self, other: Any) -> Self:
        return self.copy(self.data**other)

    def __neg__(self) -> Self:
        return self.copy(-self.data)

    def __abs__(self) -> Self:
        return self.copy(abs(self.data))

    # ------------------------------------------------------------------
    # colormapping
    # ------------------------------------------------------------------
    def get_cmapdict(self) -> ColormapDict:
        """Returns a dictionary with cmap information."""
        return _build_cmapdict(self.cmap, self.vmin, self.vmax)

    def _colormap_to_rgba(
        self,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        """Apply this view's colormap, returning ``(rgba_channels, nan_mask)``.

        Was ``Dataview.raw``. It is a private helper, not a view: every concrete
        subclass overrode ``raw`` to return an RGB *object* instead, so the
        property had two incompatible types depending on where you looked.
        """
        from matplotlib import cm, colors

        cmap = self.get_cmapdict()["cmap"]
        # Normalize colors according to vmin, vmax
        norm = colors.Normalize(self.vmin, self.vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        # Capture NaN mask before uint8 conversion (NaN info is lost after)
        nan_mask: npt.NDArray[np.bool_] = np.isnan(self.data)
        color_data = cmapper.to_rgba(self.data.flatten()).reshape(
            self.data.shape + (4,)
        )
        # rollaxis puts the last color dimension first, to allow output of
        # separate channels: r, g, b, a = view._colormap_to_rgba()[0]
        color_data = (np.clip(color_data, 0, 1) * 255).astype(np.uint8)
        color_data[nan_mask, 3] = 0
        return np.rollaxis(color_data, -1), nan_mask

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        sdict = super().to_json(simple=simple)
        if simple:
            sdict.update(
                DataviewJSON(
                    name=self.name,
                    subject=self.subject,
                    min=float(np.nan_to_num(self.data).min()),
                    max=float(np.nan_to_num(self.data).max()),
                )
            )
            return sdict

        sdict.update(
            DataviewJSON(
                cmap=[self.cmap],
                vmin=[
                    self.vmin
                    if self.vmin is not None
                    else np.percentile(np.nan_to_num(self.data), 1)
                ],
                vmax=[
                    self.vmax
                    if self.vmax is not None
                    else np.percentile(np.nan_to_num(self.data), 99)
                ],
            )
        )
        return sdict

    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        view[2] = json.dumps([self.cmap])
        view[3] = json.dumps([self.vmin])
        view[4] = json.dumps([self.vmax])

    def _write_data_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: Optional[str] = None
    ) -> h5py.Dataset:
        """Write only the ``/data`` node for this view, not the view record.

        Composite views (2D, RGB) need exactly this for their channels. That is
        what the old ``self._cls._write_hdf(self.red, h5)`` unbound-dispatch
        trick was reaching for.
        """
        if name is None:
            name = self.name
        dgrp = h5.require_group("/data")

        if name in dgrp and "__%s" % _hash(dgrp[name][()])[:16] == name:
            # don't need to update anything, since it's the same data
            return cast(h5py.Dataset, h5.get("/data/%s" % name))

        node = _hdf_write(h5, self.data, name=name)
        node.attrs["subject"] = self.subject
        return node

    def save(
        self, filename: Union[str, h5py.Group], name: Optional[str] = None
    ) -> None:
        """Save the dataset into the hdf file `filename` with the provided name."""
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5", ".hf5"):
                h5 = h5py.File(filename, "a")
                self._write_hdf(h5, name=name or "data")
                h5.close()
            else:
                raise TypeError("Unknown file type")
        elif isinstance(filename, h5py.Group):
            self._write_hdf(filename, name=name or "data")


class Multiview(Dataview):
    """Unimplemented. Retained only so the name stays importable."""

    def __init__(self, views: Any, description: str = "") -> None:
        raise NotImplementedError

    @property
    def subject(self) -> str:
        raise NotImplementedError

    @property
    def raw(self) -> DataviewRGB:
        raise NotImplementedError

    def uniques(self, collapse: bool = False) -> Iterator[Dataview]:
        raise NotImplementedError

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        raise NotImplementedError


class Volume(ScalarView):
    """
    Encapsulates a 3D volume or 4D volumetric movie. Includes information on how
    the volume should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 3D with shape (z,y,x), 1D with shape (v,) for masked data,
        4D with shape (t,z,y,x), or 2D with shape (t,v). For masked data, if the
        size of the given array matches any of the existing masks in the database,
        that mask will automatically be loaded. If it does not, an error will be
        raised.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    mask : ndarray, optional
        Binary 3D array with shape (z,y,x) showing which voxels are selected.
        If masked data is given, the mask will automatically be loaded if it
        exists in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        xfmname: Union[str, bytes],
        mask: MaskSpec = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data,
            subject,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )
        self.xfmname = (
            xfmname if isinstance(xfmname, str) else xfmname.decode("utf-8")
        )
        self._check_size(mask)
        self.masked: _masker[Volume] = _masker(self)
        self._resolve_percentiles()

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _check_size(self, mask: MaskSpec) -> None:
        if self.data.ndim not in (1, 2, 3, 4):
            raise ValueError("Invalid data shape")

        self.linear = self.data.ndim in (1, 2)
        self.movie = self.data.ndim in (2, 4)

        #: Verbatim record of what was passed as ``mask=``, for round-tripping.
        self._mask_spec: MaskSpec = None
        #: The resolved boolean mask, or None for unmasked (3D/4D) data.
        self.mask: Optional[npt.NDArray[np.bool_]] = None
        #: The database mask name, if the mask came from (or was found in) the db.
        self.mask_name: Optional[str] = None

        if self.linear:
            if mask is None:
                # Guess the mask
                nvox: int = self.data.shape[-1]
                found_name, found_mask = _find_mask(nvox, self.subject, self.xfmname)
                self.mask_name = found_name
                self.mask = found_mask
                self._mask_spec = found_name
            elif isinstance(mask, np.ndarray):
                self.mask = mask > 0
                self._mask_spec = mask > 0
            else:
                self.mask = db.get_mask(self.subject, self.xfmname, mask)
                self.mask_name = mask
                self._mask_spec = mask

            assert self.mask is not None
            self.shape: tuple[int, ...] = self.mask.shape
        else:
            shape = self.data.shape
            if self.movie:
                shape = shape[1:]
            xfm = db.get_xfm(self.subject, self.xfmname)
            if xfm.shape != shape:
                raise ValueError(
                    "Volumetric data (shape %s) is not the same shape as reference "
                    "for transform (shape %s)" % (str(shape), str(xfm.shape))
                )
            self.shape = shape

    @property
    def _mask(self) -> MaskSpec:
        """Deprecated. Use :attr:`mask` for the array or :attr:`mask_name` for the name."""
        return self._mask_spec

    @property
    def volume(self) -> npt.NDArray:
        """Returns a 3D or 4D volume for this Volume, automatically unmasking
        masked data.
        """
        from cortex import volume

        if self.linear:
            assert self.mask is not None
            data = volume.unmask(self.mask, self.data[:])
        else:
            data = self.data[:]

        if not self.movie:
            data = data[np.newaxis]

        return data

    def copy(self, data: npt.NDArray) -> Self:
        new = self.__class__(
            data,
            self.subject,
            self.xfmname,
            mask=self._mask_spec,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            description=self.description,
            state=self.state,
            **self.attrs,
        )
        return new

    def map(self, projection: str = "nearest") -> Vertex:
        """Convert this Volume into a Vertex using the given projection method.

        Parameters
        ----------
        projection : str, optional
            Type of projection to use. Default: nearest.

        Returns
        -------
        Vertex
            Vertex valued version of this Volume.
        """
        from cortex import utils

        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        data = mapper(self)
        self._set_display_params(data)
        return data

    @classmethod
    def empty(cls, subject: str, xfmname: str, value: float = 0, **kwargs: Any) -> Self:
        """
        Create a constant-valued Volume for the given subject and xfmname.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        xfmname : str
            Transform name. Must exist in the pycortex database.
        value : float, optional
            Value that the Volume will be filled with.
        **kwargs
            Other keyword arguments are passed to the init function for this
            class.

        Returns
        -------
        Volume
            A Volume whose data is constant, equal to value.
        """
        xfm = db.get_xfm(subject, xfmname)
        shape = xfm.shape
        return cls(np.ones(shape) * value, subject, xfmname, **kwargs)

    @classmethod
    def random(cls, subject: str, xfmname: str, **kwargs: Any) -> Self:
        """
        Create a random-valued Volume for the given subject and xfmname.
        Random values are from gaussian distribution with mean 0, s.d. 1.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        xfmname : str
            Transform name. Must exist in the pycortex database.
        **kwargs
            Other keyword arguments are passed to the init function for this
            class.

        Returns
        -------
        Volume
            A Volume whose data is random.
        """
        xfm = db.get_xfm(subject, xfmname)
        shape = xfm.shape
        return cls(np.random.randn(*shape), subject, xfmname, **kwargs)

    def save_nii(self, filename: Union[str, os.PathLike]) -> None:
        """Save as a nifti file at the given filename. Nifti headers are
        copied from the reference image for this Volume's transform.
        """
        xfm = db.get_xfm(self.subject, self.xfmname)
        if xfm.reference is None:
            raise IOError(
                "Transform %r for subject %r has no reference image to copy "
                "nifti headers from" % (self.xfmname, self.subject)
            )
        affine = xfm.reference.affine
        import nibabel

        new_nii = nibabel.Nifti1Image(self.volume.T, affine)
        nibabel.save(new_nii, filename)

    def __repr__(self) -> str:
        maskstr = "volumetric"
        if self.linear:
            name: Any = self._mask_spec
            if isinstance(self._mask_spec, np.ndarray):
                name = "custom"
            maskstr = "%s masked" % name
        if self.movie:
            maskstr += " movie"
        maskstr = maskstr[0].upper() + maskstr[1:]
        return "<%s data for (%s, %s)>" % (maskstr, self.subject, self.xfmname)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        if simple:
            sdict = super().to_json(simple=simple)
            sdict["shape"] = self.shape
            return sdict

        xfm = db.get_xfm(self.subject, self.xfmname, "coord").xfm
        sdict = DataviewJSON(
            xfm=[list(np.array(xfm).ravel())], data=[self.name]
        )
        sdict.update(super().to_json())
        return sdict

    def _write_data_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: Optional[str] = None
    ) -> h5py.Dataset:
        node = super()._write_data_hdf(h5, name=name)

        # write the mask into the file, as necessary
        if self._mask_spec is not None:
            mask: Any = self._mask_spec
            if isinstance(self._mask_spec, np.ndarray):
                mgrp = "/subjects/{subj}/transforms/{xfm}/masks/"
                mgrp = mgrp.format(subj=self.subject, xfm=self.xfmname)
                mname = "__%s" % _hash(self._mask_spec)[:8]
                _hdf_write(h5, self._mask_spec, name=mname, group=mgrp)
                mask = mname

            node.attrs["mask"] = mask

        return node

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self._write_data_hdf(h5)
        return self._write_view_node(
            h5, name=name, data=[self.name], xfmname=[self.xfmname]
        )

    @property
    def raw(self) -> VolumeRGB:
        (r, g, b, a), nan_mask = self._colormap_to_rgba()
        result = VolumeRGB(
            r,
            g,
            b,
            self.subject,
            self.xfmname,
            a,
            description=self.description,
            state=self.state,
            priority=self.priority,
        )
        result._nan_mask = nan_mask
        return result


class Vertex(ScalarView):
    """
    Encapsulates a 1D vertex map or 2D vertex movie. Includes information on how
    the data should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 1D with shape (v,), or 2D with shape (t,v). Here, v can
        be the number of vertices in both hemispheres, or the number of vertices
        in either one of the hemispheres. In that case, the data for the other
        hemisphere will be filled with zeros.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data,
            subject,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )
        try:
            left, right = db.get_surf(self.subject, "wm")
        except IOError:
            left, right = db.get_surf(self.subject, "fiducial")
        self.llen = len(left[0])
        self.rlen = len(right[0])
        self._set_data(self._data)
        self._resolve_percentiles()

    def _set_data(self, data: Optional[npt.NDArray]) -> None:
        """
        Stores data for this Vertex, filling the other hemisphere with zeros if
        only one hemisphere's worth of data was given. See __init__ for `data`
        shape possibilities.
        """
        if data is None:
            data = np.zeros((self.llen + self.rlen,))

        self._data = data
        self.movie = self.data.ndim > 1
        self.nverts = self.data.shape[-1]
        if self.llen == self.nverts:
            # Just data for left hemisphere
            self.hem = "left"
            rshape = list(self.data.shape)
            rshape[1 if self.movie else 0] = self.rlen
            self._data = np.hstack(
                [self.data, np.zeros(rshape, dtype=self.data.dtype)]
            )
        elif self.rlen == self.nverts:
            # Just data for right hemisphere
            self.hem = "right"
            lshape = list(self.data.shape)
            lshape[1 if self.movie else 0] = self.llen
            self._data = np.hstack(
                [np.zeros(lshape, dtype=self.data.dtype), self.data]
            )
        elif self.llen + self.rlen == self.nverts:
            # Data for both hemispheres
            self.hem = "both"
        else:
            raise ValueError(
                "Invalid number of vertices for subject (given %d, should be %d for "
                "left hem, %d for right hem, or %d for both)"
                % (self.nverts, self.llen, self.rlen, self.llen + self.rlen)
            )

    def copy(self, data: npt.NDArray) -> Self:
        """
        Return a new Vertex object for the same subject but with data
        replaced by the given `data`.

        This is useful for efficiently creating many Vertex objects, since
        it doesn't require reloading the surfaces from the database to check
        numbers of vertices, etc.
        """
        return self.__class__(
            data,
            self.subject,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            description=self.description,
            state=self.state,
            **self.attrs,
        )

    @property
    def vertices(self) -> npt.NDArray:
        verts = self.data
        if not self.movie:
            verts = verts[np.newaxis]
        return verts

    @property
    def left(self) -> npt.NDArray:
        """Data for only the left hemisphere vertices."""
        if self.movie:
            return self.data[:, : self.llen]
        return self.data[: self.llen]

    @property
    def right(self) -> npt.NDArray:
        """Data for only the right hemisphere vertices."""
        if self.movie:
            return self.data[:, self.llen :]
        return self.data[self.llen :]

    def volume(
        self, xfmname: str, projection: str = "nearest", **kwargs: Any
    ) -> Volume:
        """
        Map this Vertex back to volume space, creating a Volume object.
        This uses the `mapper.backwards` function, which is not particularly
        accurate.

        Parameters
        ----------
        xfmname : str
            Transform name for the volume space that this vertex data will be
            projected into. Must exist in the pycortex database.
        projection : str, optional
            The type of projection method to use. See the docs for `mapper` for
            possibilities. Default: nearest.
        **kwargs
            Other keyword args are passed to the `mapper.backwards` function.

        Returns
        -------
        Volume
            Volume containing the back-projected vertex data.
        """
        warnings.warn("Inverse mapping cannot be accurate")
        from cortex import utils

        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def map(
        self,
        target_subj: str,
        surface_type: str = "fiducial",
        hemi: Literal["lh", "rh", "both"] = "both",
        fs_subj: Optional[str] = None,
        **kwargs: Any,
    ) -> Vertex:
        """Map this data from this surface to another surface

        Calls `cortex.freesurfer.vertex_to_vertex()`  with this
        vertex object as the first argument.

        NOTE: Requires either previous computation of mapping matrices
        (with `cortex.db.get_mri_surf2surf_matrix`) or active
        freesurfer environment.

        Parameters
        ----------
        target_subj : str
            freesurfer subject to which to map

        Other Parameters
        ----------------
        kwargs map to `cortex.freesurfer.vertex_to_vertex()`
        """
        # Input check
        if hemi not in ["lh", "rh", "both"]:
            raise ValueError("`hemi` kwarg must be 'lh', 'rh', or 'both'")
        # lazy load
        from ..database import db

        mats = db.get_mri_surf2surf_matrix(
            self.subject,
            surface_type,
            hemi="both",
            target_subj=target_subj,
            fs_subj=fs_subj,
            **kwargs,
        )
        new_data = [mats[0].dot(self.left), mats[1].dot(self.right)]
        if hemi == "both":
            stacked = np.hstack(new_data)
        elif hemi == "lh":
            stacked = np.hstack(
                [new_data[0], np.nan * np.zeros(new_data[1].shape)]
            )
        else:
            stacked = np.hstack(
                [np.nan * np.zeros(new_data[0].shape), new_data[1]]
            )
        return Vertex(
            stacked, target_subj, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
        )

    @classmethod
    def empty(cls, subject: str, value: float = 0, **kwargs: Any) -> Self:
        """
        Create a constant-valued Vertex for the given subject.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        value : float, optional
            Value that the Vertex will be filled with.
        **kwargs
            Other keyword arguments are passed to the init function for this
            class.

        Returns
        -------
        Vertex
            A Vertex whose data is constant, equal to value.
        """
        nverts = cls._count_verts(subject)
        return cls(np.ones((nverts,)) * value, subject, **kwargs)

    @classmethod
    def random(cls, subject: str, **kwargs: Any) -> Self:
        """
        Create a random-valued Vertex for the given subject.
        Random values are from gaussian distribution with mean 0, s.d. 1.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        **kwargs
            Other keyword arguments are passed to the init function for this
            class.

        Returns
        -------
        Vertex
            A Vertex with random data.
        """
        nverts = cls._count_verts(subject)
        return cls(np.random.randn(nverts), subject, **kwargs)

    @staticmethod
    def _count_verts(subject: str) -> int:
        try:
            left, right = db.get_surf(subject, "wm")
        except IOError:
            left, right = db.get_surf(subject, "fiducial")
        return len(left[0]) + len(right[0])

    def __getitem__(self, idx: Any) -> Self:
        """Get the Vertex for the given time index. Only works for movie (2D)
        vertex data.
        """
        if not self.movie:
            raise TypeError("Cannot index non-movie data")

        return self.copy(self.data[idx])

    def __repr__(self) -> str:
        maskstr = "movie " if self.movie else ""
        return "<Vertex %sdata for %s>" % (maskstr, self.subject)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        if simple:
            sdict = DataviewJSON(
                split=self.llen, frames=self.vertices.shape[0]
            )
            sdict.update(super().to_json(simple=simple))
            return sdict

        sdict = DataviewJSON(data=[self.name])
        sdict.update(super().to_json())
        return sdict

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self._write_data_hdf(h5)
        return self._write_view_node(h5, name=name, data=[self.name])

    @property
    def raw(self) -> VertexRGB:
        (r, g, b, a), nan_mask = self._colormap_to_rgba()
        result = VertexRGB(
            r,
            g,
            b,
            self.subject,
            a,
            description=self.description,
            state=self.state,
            priority=self.priority,
        )
        result._nan_mask = nan_mask
        return result

    def blend_curvature(
        self,
        alpha: npt.NDArray[np.floating],
        threshold: float = 0,
        brightness: float = 0.5,
        contrast: float = 0.25,
        smooth: float = 20,
    ) -> VertexRGB:
        """Blend this map with a curvature map. Deprecated; see
        :func:`_blend_curvature` for the full docstring and the replacement."""
        return _blend_curvature(
            self,
            alpha,
            threshold=threshold,
            brightness=brightness,
            contrast=contrast,
            smooth=smooth,
        )


class SupportsCurvatureBlend(Protocol):
    """What :func:`_blend_curvature` needs from its receiver."""

    @property
    def subject(self) -> str: ...

    @property
    def raw(self) -> VertexRGB: ...


def _blend_curvature(
    view: SupportsCurvatureBlend,
    alpha: npt.NDArray[np.floating],
    threshold: float = 0,
    brightness: float = 0.5,
    contrast: float = 0.25,
    smooth: float = 20,
) -> VertexRGB:
    """Blend the data with a curvature map depending on a transparency map.

    .. deprecated::
        Per-vertex/voxel alpha is now honored directly by both the WebGL
        viewer and ``cortex.quickshow``, so this curvature-blending hack
        is no longer needed. The recommended replacement for scalar data
        with a transparency map is :class:`Vertex2D` (or
        :class:`Volume2D`) with a 2D colormap whose second axis encodes
        alpha (e.g. ``"fire_alpha"``, ``"PU_RdBu_covar_alpha"``)::

            # Was:
            #   blended = vtx.blend_curvature(alpha)
            #   cortex.quickshow(blended)
            # Now:
            v2d = cortex.Vertex2D(vtx.data, alpha, subject,
                                  cmap="fire_alpha",
                                  vmin=vtx.vmin, vmax=vtx.vmax,
                                  vmin2=0, vmax2=1)
            cortex.quickshow(v2d)         # or cortex.webgl.show(v2d)

        The 2D colormap path keeps colormap parameters (``cmap``,
        ``vmin``, ``vmax``) editable on the resulting object, and the
        curvature underlay is composited through automatically by both
        the matplotlib and WebGL renderers.

        For data that is already RGB, pass ``alpha=`` to
        :class:`VertexRGB` / :class:`VolumeRGB` directly instead.

    Vertex objects cannot use transparency as Volume objects. This function
    is a hack to mimic the transparency of Volume objects, blending the
    Vertex data with a curvature map. It returns a VertexRGB object, and the
    colormap parameters (vmin, vmax, cmap, ...) of the original Vertex object
    cannot be changed later on.

    Parameters
    ----------
    view : Vertex, Vertex2D or VertexRGB
        The view to blend.
    alpha : array of shape (n_vertices, )
        Transparency map.
    threshold : float
        Threshold for the curvature map.
    brightness : float
        Brightness of the curvature map.
    contrast : float
        Contrast of the curvature map.
    smooth : float
        Smoothness of the curvature map.

    Returns
    -------
    blended : VertexRGB object
        The original map blended with a curvature map.
    """
    warnings.warn(
        "blend_curvature is deprecated and will be removed in a future "
        "release. Per-vertex/voxel alpha is now honored directly by both "
        "the WebGL viewer and quickshow, so this curvature-blending hack "
        "is no longer needed. For scalar data with a transparency map, "
        "use Vertex2D / Volume2D with a 2D colormap whose second axis "
        "encodes alpha (e.g. 'fire_alpha', 'PU_RdBu_covar_alpha'), e.g. "
        "`Vertex2D(data, alpha, subject, cmap='fire_alpha', vmin=..., "
        "vmax=..., vmin2=0, vmax2=1)`. For data that is already RGB, "
        "pass `alpha=` to VertexRGB / VolumeRGB directly.",
        DeprecationWarning,
        stacklevel=3,
    )
    # prepare curvature map
    curvature = db.get_surfinfo(view.subject, smooth=smooth).data
    curvature = (curvature > threshold).astype("float")
    curvature = curvature * contrast + brightness
    curvature_raw = Vertex(
        curvature, view.subject, vmin=0, vmax=1, cmap="gray"
    ).raw

    # prepare alpha map
    clipped = np.clip(alpha.astype("float"), 0, 1)

    # blend original map with curvature map. VertexRGB.raw returns self, so copy.
    blended = deepcopy(view.raw)
    for channel, curv in (
        ("red", curvature_raw.red),
        ("green", curvature_raw.green),
        ("blue", curvature_raw.blue),
    ):
        chan = getattr(blended, channel)
        chan.data = (chan.data * clipped + (1 - clipped) * curv.data).astype("uint8")

    return blended


# Generic allows us to return the concrete view class, not just ScalarView
T_masker = TypeVar("T_masker", bound=Volume)


class _masker(Generic[T_masker]):
    def __init__(self, dv: T_masker) -> None:
        self.dv = dv

        self.data: Optional[npt.NDArray] = None
        if dv.linear:
            self.data = dv.data

    def __getitem__(self, masktype: str) -> T_masker:
        mask = db.get_mask(self.dv.subject, self.dv.xfmname, masktype)
        return self.dv.copy(self.dv.volume[:, mask].squeeze())


# ----------------------------------------------------------------------
# factories
# ----------------------------------------------------------------------
@overload
def normalize(data: tuple[Any, Any, Any]) -> Union[Volume, VolumeRGB]: ...


@overload
def normalize(data: tuple[Any, Any]) -> Vertex: ...


@overload
def normalize(data: Dataview) -> Dataview: ...


def normalize(
    data: Union[Dataview, tuple],
) -> Union[Volume, VolumeRGB, Vertex, Dataview]:
    if isinstance(data, tuple):
        if len(data) == 3:
            if data[0].dtype == np.uint8:
                return VolumeRGB(
                    data[0][..., 0], data[0][..., 1], data[0][..., 2], *data[1:]
                )
            return Volume(*data)
        elif len(data) == 2:
            return Vertex(*data)
        else:
            raise TypeError("Invalid input for Dataview")
    elif isinstance(data, Dataview):
        return data
    else:
        raise TypeError("Invalid input for Dataview")


#: The subset of view kwargs the RGB constructors accept. They take no
#: cmap/vmin/vmax, so anything else has to be filtered out before splatting.
_RGB_KWARGS = ("description", "state", "priority")


def _from_hdf_data(
    h5: h5py.File,
    name: str,
    xfmname: Optional[str] = None,
    subject: Optional[str] = None,
    **kwargs: Any,
) -> Dataview:
    """Decodes a __hash named node from an HDF file into the
    constituent Vertex or Volume object.

    Returns an RGB view rather than a scalar one for legacy uint8 nodes with a
    trailing channel axis, which is why the return type is the root and not
    ``ScalarView``.
    """
    dnode = h5.get("/data/%s" % name)
    if dnode is None:
        dnode = h5.get(name)

    attrs = {k: u(v) for (k, v) in dnode.attrs.items()}
    if subject is None:
        subject = attrs["subject"]
    # support old style xfmname saving as attribute
    if xfmname is None and "xfmname" in attrs:
        xfmname = attrs["xfmname"]
    mask = None
    if "mask" in attrs:
        if attrs["mask"].startswith("__"):
            mask = h5[
                "/subjects/%s/transforms/%s/masks/%s"
                % (attrs["subject"], xfmname, attrs["mask"])
            ][()]
        else:
            mask = attrs["mask"]

    # support old style RGB volumes
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        alpha = None
        if dnode.shape[-1] == 4:
            alpha = dnode[..., 3]

        rgb_kwargs = {k: v for k, v in kwargs.items() if k in _RGB_KWARGS}

        if xfmname is None:
            return VertexRGB(
                dnode[..., 0],
                dnode[..., 1],
                dnode[..., 2],
                subject,
                alpha=alpha,
                **rgb_kwargs,
            )

        return VolumeRGB(
            dnode[..., 0],
            dnode[..., 1],
            dnode[..., 2],
            subject,
            xfmname,
            alpha=alpha,
            **rgb_kwargs,
        )

    if xfmname is None:
        return Vertex(dnode, subject, **kwargs)

    return Volume(dnode, subject, xfmname, mask=mask, **kwargs)


def _from_hdf_view(
    h5: h5py.File,
    data: Any,
    xfmname: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    subject: Optional[str] = None,
    **kwargs: Any,
) -> Dataview:
    if isinstance(data, str):
        return _from_hdf_data(
            h5, data, xfmname=xfmname, vmin=vmin, vmax=vmax, subject=subject, **kwargs
        )

    # Surface views have no transform, so slot 7 of the view node is null and
    # `xfmname` arrives as None rather than as a per-channel list.
    xfmnames = xfmname if isinstance(xfmname, (list, tuple)) else [xfmname] * len(data)

    if len(data) == 2:
        dim1 = _from_hdf_data(h5, data[0], xfmname=xfmnames[0], subject=subject)
        dim2 = _from_hdf_data(h5, data[1], xfmname=xfmnames[1], subject=subject)
        if isinstance(dim1, Vertex):
            assert isinstance(dim2, Vertex)
            return Vertex2D(
                dim1,
                dim2,
                vmin=vmin[0],
                vmin2=vmin[1],
                vmax=vmax[0],
                vmax2=vmax[1],
                subject=subject,
                **kwargs,
            )
        assert isinstance(dim1, Volume) and isinstance(dim2, Volume)
        return Volume2D(
            dim1,
            dim2,
            vmin=vmin[0],
            vmin2=vmin[1],
            vmax=vmax[0],
            vmax2=vmax[1],
            subject=subject,
            **kwargs,
        )
    elif len(data) == 4:
        red, green, blue = [
            _from_hdf_data(h5, d, xfmname=xfmnames[0], subject=subject)
            for d in data[:3]
        ]
        alpha = None
        if data[3] is not None:
            alpha = _from_hdf_data(h5, data[3], xfmname=xfmnames[0], subject=subject)

        rgb_kwargs = {k: v for k, v in kwargs.items() if k in _RGB_KWARGS}
        if isinstance(red, Vertex):
            assert isinstance(green, Vertex) and isinstance(blue, Vertex)
            assert alpha is None or isinstance(alpha, Vertex)
            return VertexRGB(
                red, green, blue, alpha=alpha, subject=subject, **rgb_kwargs
            )
        assert isinstance(red, Volume)
        assert isinstance(green, Volume) and isinstance(blue, Volume)
        assert alpha is None or isinstance(alpha, Volume)
        return VolumeRGB(red, green, blue, alpha=alpha, subject=subject, **rgb_kwargs)
    else:
        raise ValueError("Invalid Dataview specification")


from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB  # noqa: E402
from .view2D import Dataview2D, Vertex2D, Volume2D  # noqa: E402
