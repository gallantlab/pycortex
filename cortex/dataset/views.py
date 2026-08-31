from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Optional, TypedDict, Union, cast, overload, Literal
if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired

import h5py
from matplotlib.colors import Colormap
import numpy as np
import numpy.typing as npt

from .. import options
from .braindata import VertexData, VolumeData

default_cmap = options.config.get("basic", "default_cmap")

# register_cmap is deprecated in matplotlib > 3.7.0 and replaced by colormaps.register
try:
    from matplotlib import colormaps as cm

    def register_cmap(cmap):
        return cm.register(cmap)
except ImportError:
    from matplotlib.cm import register_cmap


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


@overload
def normalize(data: tuple[Any, Any, Any]) -> Union[Volume, VolumeRGB]: ...


@overload
def normalize(data: tuple[Any, Any]) -> Vertex: ...


@overload
def normalize(data: Dataview) -> Dataview: ...


def normalize(data: Union[Dataview, tuple]) -> Union[Volume, VolumeRGB, Vertex, Dataview]:
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


def _from_hdf_data(h5, name, xfmname=None, subject=None, **kwargs):
    """Decodes a __hash named node from an HDF file into the
    constituent Vertex or Volume object"""
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
            ].value
        else:
            mask = attrs["mask"]

    # support old style RGB volumes
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        alpha = None
        if dnode.shape[-1] == 4:
            alpha = dnode[..., 3]

        # RGB classes don't accept arbitrary kwargs (e.g. cmap, vmin, vmax),
        # so filter to only the parameters they support.
        rgb_kwargs = {
            k: v for k, v in kwargs.items() if k in ("description", "state", "priority")
        }

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
    h5, data, xfmname=None, vmin=None, vmax=None, subject=None, **kwargs
):
    if isinstance(data, str):
        return _from_hdf_data(
            h5, data, xfmname=xfmname, vmin=vmin, vmax=vmax, subject=subject, **kwargs
        )

    if len(data) == 2:
        dim1 = _from_hdf_data(h5, data[0], xfmname=xfmname[0], subject=subject)
        dim2 = _from_hdf_data(h5, data[1], xfmname=xfmname[1], subject=subject)
        cls = Vertex2D if isinstance(dim1, Vertex) else Volume2D
        return cls(
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
            _from_hdf_data(h5, d, xfmname=xfmname, subject=subject) for d in data[:3]
        ]
        alpha = None
        if data[3] is not None:
            alpha = _from_hdf_data(h5, data[3], xfmname=xfmname, subject=subject)

        cls = VertexRGB if isinstance(red, Vertex) else VolumeRGB
        rgb_kwargs = {
            k: v for k, v in kwargs.items() if k in ("description", "state", "priority")
        }
        return cls(red, green, blue, alpha=alpha, subject=subject, **rgb_kwargs)
    else:
        raise ValueError("Invalid Dataview specification")


class ColormapDict(TypedDict):
    cmap: Colormap
    vmin: Optional[float]
    vmax: Optional[float]


class DataviewJSON(TypedDict):
    state: Any
    attrs: dict[str, Any]
    desc: str
    cmap: Optional[list[str]]
    vmin: Optional[list[float]]
    vmax: Optional[list[float]]
    name: NotRequired[str]
    raw: NotRequired[bool]
    mosaic: NotRequired[tuple[int, int]]
    subject: NotRequired[str] # is this actually from BrainData?


class Dataview:
    _nan_mask: Optional[npt.NDArray[np.bool_]]

    def __init__(
        self,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state=None,
        **kwargs,
    ):
        if self.__class__ == Dataview:
            raise TypeError("Cannot directly instantiate Dataview objects")

        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.state = state
        self.attrs = kwargs
        if "priority" not in self.attrs:
            self.attrs["priority"] = 1
        self.description = description

    def copy(self, *args, **kwargs):
        """Create a new instance of this Dataview's class, reusing its
        display settings (cmap, vmin, vmax, description, state, attrs).

        Parameters
        ----------
        *args
            Positional arguments passed to the subclass constructor (e.g.
            new `red`/`green`/`blue` data for a VolumeRGB).
        **kwargs
            Additional keyword arguments; merged with (and overriding) this
            Dataview's own `attrs`.

        Returns
        -------
        Dataview
            A new instance of `self.__class__`.
        """
        kwargs.update(self.attrs)
        return self.__class__(
            *args,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            description=self.description,
            state=self.state,
            **kwargs,
        )

    @property
    def priority(self):
        return self.attrs["priority"]

    @priority.setter
    def priority(self, value):
        self.attrs["priority"] = value

    def to_json(self, simple: bool=False) -> DataviewJSON:
        if simple:
            return dict()

        desc = self.description
        if isinstance(desc, bytes):
            desc = desc.decode()
        sdict = dict(state=self.state, attrs=self.attrs.copy(), desc=desc)
        try:
            sdict.update(
                dict(
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
        except AttributeError:
            pass
        return sdict

    @staticmethod
    def from_hdf(node, subject=None):
        """Reconstruct a Dataview from a `/views/<name>` node in an HDF5
        file previously written by `Dataview._write_hdf`.

        Parameters
        ----------
        node : h5py.Dataset
            The view node to decode.
        subject : str, optional
            Subject to use instead of the one stored in the file (e.g. if
            the subject has been renamed since the file was written).

        Returns
        -------
        Dataview
            The decoded Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, or
            Vertex2D object.
        """
        data = json.loads(u(node[0]))
        desc = node[1]
        try:
            cmap = json.loads(u(node[2]))
        except:
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

        if len(data) == 1:
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
        else:
            views = [
                _from_hdf_view(node.file, d, xfmname=x, subject=subject)
                for d, x in zip(data, xfmname)
            ]
            raise NotImplementedError

    def _write_hdf(self, h5, name="data", data=None, xfmname=None):
        views = h5.require_group("/views")
        view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
        view[0] = json.dumps(data)
        view[1] = self.description
        try:
            view[2] = json.dumps([self.cmap])
            view[3] = json.dumps([self.vmin])
            view[4] = json.dumps([self.vmax])
        except AttributeError:
            # For VolumeRGB/Vertex, there is no cmap/vmin/vmax
            view[2] = "null"
            view[3:5] = "null"
        view[5] = json.dumps(self.state)
        view[6] = json.dumps(self.attrs)
        view[7] = json.dumps(xfmname)
        return view

    def get_cmapdict(self) -> ColormapDict:
        """Returns a dictionary with cmap information."""

        from matplotlib import colors
        from matplotlib import pyplot as plt

        try:
            # plt.get_cmap accepts:
            # - matplotlib colormap names
            # - pycortex colormap names previously registered in matplotlib
            # - matplotlib.colors.Colormap instances
            cmap = plt.get_cmap(self.cmap)
        except ValueError:
            # unknown colormap, test whether it's in pycortex colormaps
            cmapdir = options.config.get("webgl", "colormaps")
            colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
            colormaps = {os.path.split(c)[1][:-4]: c for c in colormaps}
            if self.cmap not in colormaps:
                raise ValueError("Unknown color map %s" % self.cmap)
            I = plt.imread(colormaps[self.cmap])
            name = self.cmap if isinstance(self.cmap, str) else self.cmap.name
            cmap = colors.ListedColormap(np.squeeze(I), name=name)
            # Register colormap to matplotlib to avoid loading it again
            register_cmap(cmap)

        return ColormapDict(cmap=cmap, vmin=self.vmin, vmax=self.vmax)

    @property
    def raw(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        from matplotlib import cm, colors

        cmap = self.get_cmapdict()["cmap"]
        # Normalize colors according to vmin, vmax
        norm = colors.Normalize(self.vmin, self.vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        # Capture NaN mask before uint8 conversion (NaN info is lost after)
        nan_mask: npt.NDArray[np.bool_] = np.isnan(self.data)
        # TODO: self.data relies on BrainData. Would need common inheritance for this to work.
        color_data = cmapper.to_rgba(self.data.flatten()).reshape(
            self.data.shape + (4,)
        )
        # rollaxis puts the last color dimension first, to allow output of separate channels: r,g,b,a = dataset.raw
        color_data = (np.clip(color_data, 0, 1) * 255).astype(np.uint8)
        color_data[nan_mask, 3] = 0
        return np.rollaxis(color_data, -1), nan_mask


class Multiview(Dataview):
    def __init__(self, views, description=""):
        for view in views:
            if not isinstance(view, Dataview):
                raise TypeError("Must be a View object!")
        raise NotImplementedError
        self.views = views

    def uniques(self, collapse=False):
        for view in self.views:
            for sv in view.uniques(collapse=collapse):
                yield sv


class Volume(VolumeData, Dataview):
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
        All additional arguments in kwargs are passed to the VolumeData and Dataview.
            state : untyped
                role unclear
            priority : int (default  = 1)
                controls the order in which datasets are viewed in webgl
            stim : str
                path to stimulus file
            rate : numeric (default 1)
                frame rate for movie/time series playback in webgl
            delay : numeric (default 0)
                delay for movie/time series playback in webgl
            filter : str 
                interpolation filter for volumetric rendering in webgl (nearest/trilinear/nearlin/debug)



    """

    def __init__(
        self,
        data: npt.NDArray,
        subject: str,
        xfmname: str,
        mask: Optional[npt.NDArray] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        **kwargs,
    ):
        super().__init__(
            data,
            subject,
            xfmname,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            **kwargs,
        )
        # set vmin and vmax
        self.vmin: float = (
            self.vmin
            if self.vmin is not None
            else np.percentile(np.nan_to_num(self.data), 1).astype(float) 
        )
        self.vmax: float = (
            self.vmax
            if self.vmax is not None
            else np.percentile(np.nan_to_num(self.data), 99).astype(float)
        )

    def _write_hdf(self, h5, name="data"):
        datanode = VolumeData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(
            self, h5, name=name, data=[self.name], xfmname=[self.xfmname]
        )
        return viewnode

    @property
    def raw(self) -> VolumeRGB:
        """Colormap this Volume's data into an RGBA VolumeRGB.

        Returns
        -------
        VolumeRGB
            This Volume's data, mapped through `cmap`/`vmin`/`vmax` into
            per-voxel RGB colors, with alpha set to 0 for NaN voxels.
        """
        (r, g, b, a), nan_mask = super().raw
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


class Vertex(VertexData, Dataview):
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
        All additional arguments in kwargs are passed to the VolumeData and Dataview
            state : untyped
                role unclear
            priority : int (default  = 1)
                controls the order in which datasets are viewed in webgl
            stim : str
                path to stimulus file
            rate : numeric (default 1)
                frame rate for movie/time series playback in webgl
            delay : numeric (default 0)
                delay for movie/time series playback in webgl
            filter : str
                interpolation filter for volumetric rendering in webgl (nearest/trilinear/nearlin/debug)

    """

    def __init__(
        self,
        data: npt.NDArray,
        subject: str,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        **kwargs,
    ):
        super().__init__(
            data,
            subject,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            **kwargs,
        )
        # set vmin and vmax
        self.vmin = (
            self.vmin
            if self.vmin is not None
            else np.percentile(np.nan_to_num(self.data), 1).astype(float)
        )
        self.vmax = (
            self.vmax
            if self.vmax is not None
            else np.percentile(np.nan_to_num(self.data), 99).astype(float)
        )

    def _write_hdf(self, h5, name="data"):
        datanode = VertexData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(self, h5, name=name, data=[self.name])
        return viewnode

    @property
    def raw(self) -> VertexRGB:
        """Colormap this Vertex's data into an RGBA VertexRGB.

        Returns
        -------
        VertexRGB
            This Vertex's data, mapped through `cmap`/`vmin`/`vmax` into
            per-vertex RGB colors, with alpha set to 0 for NaN vertices.
        """
        (r, g, b, a), nan_mask = super().raw
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

    def map(
        self,
        target_subj: str,
        surface_type: str = "fiducial",
        hemi: Literal["lh", "rh", "both"] = "both",
        fs_subj: Optional[str] = None,
        **kwargs,
    ) -> Vertex:
        """Map this data from this surface to another surface

        Builds a source-to-target vertex mapping matrix via
        `cortex.freesurfer.get_mri_surf2surf_matrix()` and applies it to
        this Vertex's data.

        NOTE: Requires the source and target subjects' registered sphere
        surfaces (`?h.sphere.reg`), produced by Freesurfer's `recon-all`
        pipeline. No active Freesurfer installation is needed at call
        time -- the mapping is computed directly from those files in
        pure Python.

        Parameters
        ----------
        target_subj : str
            freesurfer subject to which to map

        Other Parameters
        ----------------
        kwargs map to `cortex.freesurfer.get_mri_surf2surf_matrix()`

        Returns
        -------
        Vertex
            This data, resampled onto `target_subj`'s vertices, with the
            same `cmap`, `vmin`, and `vmax` as this Vertex.
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
            new_data = np.hstack(new_data)
        elif hemi == "lh":
            new_data = np.hstack([new_data[0], np.nan * np.zeros(new_data[1].shape)])
        elif hemi == "rh":
            new_data = np.hstack([np.nan * np.zeros(new_data[0].shape), new_data[1]])
        vx = Vertex(
            new_data, target_subj, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
        )
        return vx


def u(s, encoding="utf8"):
    try:
        return s.decode(encoding)
    except AttributeError:
        return s


from .viewRGB import Colors, VertexRGB, VolumeRGB
from .view2D import Vertex2D, Volume2D, Dataview2D
