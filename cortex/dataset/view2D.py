from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar, Union, cast

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import h5py
import numpy as np
import numpy.typing as npt

from .. import options
from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .viewRGB import DataviewRGB
from .views import (
    ColormapDict,
    Dataview,
    DataviewJSON,
    Packable,
    ScalarView,
    SurfaceView,
    Vertex,
    VertexRGB,
    Volume,
    VolumeRGB,
    RenderableView,
    VolumetricView,
    _build_cmapdict,
    _require,
)

default_cmap2D = options.config.get("basic", "default_cmap2D")

#: Covariant: the channels are read-only properties, so `Dataview2D[Volume]` is
#: safely usable where `Dataview2D[ScalarView]` is expected. An invariant
#: parameter would reject that, which is the usual generics tax.
ScalarT = TypeVar("ScalarT", bound=ScalarView, covariant=True)


class Dataview2D(RenderableView, Generic[ScalarT]):
    """Abstract base class for 2-dimensional data views.

    Holds two scalar channels displayed through a single 2D colormap. The
    channels are read-only: they are set once at construction, and keeping them
    read-only is what makes it sound to treat this class as covariant in its
    channel type.

    Generic in the channel type, so ``Volume2D.dim1`` is a ``Volume`` and
    ``Vertex2D.dim1`` is a ``Vertex`` without either class re-declaring them.
    """

    def __init__(
        self,
        dim1: ScalarT,
        dim2: ScalarT,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        self._dim1 = dim1
        self._dim2 = dim2
        self.cmap = cmap or default_cmap2D
        # Each axis falls back to its own channel's range. This used to be done
        # twice: the subclasses pre-resolved it from the channels, and this base
        # had a *different* rule (vmin2 falling back to vmin) that could never
        # fire because of that pre-resolution. One rule, in one place.
        self.vmin = dim1.vmin if vmin is None else vmin
        self.vmax = dim1.vmax if vmax is None else vmax
        self.vmin2 = dim2.vmin if vmin2 is None else vmin2
        self.vmax2 = dim2.vmax if vmax2 is None else vmax2
        super().__init__(description=description, state=state, **kwargs)

    @property
    def dim1(self) -> ScalarT:
        return self._dim1

    @property
    def dim2(self) -> ScalarT:
        return self._dim2

    @property
    def space(self) -> BrainSpace:
        return self.dim1.space

    def uniques(self, collapse: bool = False) -> Iterator[Packable]:
        yield self.dim1
        yield self.dim2

    def get_cmapdict(self) -> ColormapDict:
        """Colormap arguments for the *first* axis of the 2D colormap.

        The second axis's range (``vmin2``/``vmax2``) has no place in an
        ``imshow`` call; the 2D colorbar is built separately in
        ``cortex.quickflat.view`` from ``vmin``/``vmax``/``vmin2``/``vmax2``.
        """
        return _build_cmapdict(self.cmap, self.vmin, self.vmax)

    def copy(self) -> Self:
        """A new view of the same kind over the same two channels.

        The RGB and 2D families had no working ``copy()`` at all before: the
        inherited ``Dataview.copy`` splatted ``cmap=``/``vmin=``/``vmax=`` into
        ``self.__class__(...)``, which their constructors do not accept.
        """
        return self.__class__(
            self.dim1,
            self.dim2,
            description=self.description,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            vmin2=self.vmin2,
            vmax2=self.vmax2,
            state=self.state,
            **self.attrs,
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        view[2] = json.dumps([self.cmap])
        view[3] = json.dumps([[self.vmin, self.vmin2]])
        view[4] = json.dumps([[self.vmax, self.vmax2]])

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self.dim1._write_data_hdf(h5)
        self.dim2._write_data_hdf(h5)
        return self._write_view_node(
            h5, name=name, data=[[self.dim1.name, self.dim2.name]]
        )

    def to_json(self, simple: bool = False) -> DataviewJSON:
        # NOTE: `simple` is deliberately ignored, preserving long-standing
        # behaviour. It is never exercised: the webgl packer only calls
        # to_json(simple=True) on the scalar channels yielded by uniques(),
        # never on the 2D view itself.
        sdict = super().to_json(simple=False)

        d1js = self.dim1.to_json()
        d2js = self.dim2.to_json()
        d1vmin, d1vmax = d1js["vmin"], d1js["vmax"]
        d2vmin, d2vmax = d2js["vmin"], d2js["vmax"]
        assert d1vmin is not None and d1vmax is not None
        assert d2vmin is not None and d2vmax is not None

        sdict.update(
            DataviewJSON(
                data=[[self.dim1.name, self.dim2.name]],
                cmap=[self.cmap],
                # `is None`, not truthiness: an explicit vmin=0 is a real bound.
                vmin=[
                    [
                        self.vmin if self.vmin is not None else d1vmin[0],
                        self.vmin2 if self.vmin2 is not None else d2vmin[0],
                    ]
                ],
                vmax=[
                    [
                        self.vmax if self.vmax is not None else d1vmax[0],
                        self.vmax2 if self.vmax2 is not None else d2vmax[0],
                    ]
                ],
            )
        )

        if "xfm" in d1js:
            sdict["xfm"] = [[d1js["xfm"][0], d2js["xfm"][0]]]

        return sdict

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    @property
    def spatial_data(self) -> npt.NDArray[np.uint8]:
        """The colormapped RGBA array, as produced by :attr:`raw`.

        A 2D view has no array of its own -- it is two channels plus a joint
        colormap -- so what a renderer samples is whatever its RGB form ships.
        Published as ``volume`` and ``vertices`` by the spatial interface;
        ``Volume2D.volume`` did not exist at all until recently, which forced
        every consumer to special-case it and reach for ``.raw.volume`` itself.
        """
        return self.raw.spatial_data

    def _to_raw(
        self, data1: npt.NDArray, data2: npt.NDArray
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
    ]:
        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize

        cmapdir = options.config.get("webgl", "colormaps")
        cmap = plt.imread(os.path.join(cmapdir, "%s.png" % self.cmap))
        _warn_non_perceptually_uniform_colormap(self.cmap)

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)

        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1] - 1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32)
        dim2 = np.round(d2 * (cmap.shape[0] - 1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # map r, g, b, a values between 0 and 255 to avoid problems with
        # VolumeRGB when plotting flatmaps with quickflat
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = (
            channel.reshape(dim1.shape) for channel in colored.T
        )
        # Preserve nan values as alpha = 0
        aidx = np.logical_or(np.isnan(data1), np.isnan(data2))
        a = a.copy()
        a[aidx] = 0
        return r, g, b, a

    def _finish_raw(
        self,
        r: npt.NDArray[np.uint8],
        g: npt.NDArray[np.uint8],
        b: npt.NDArray[np.uint8],
        a: npt.NDArray[np.uint8],
    ) -> DataviewRGB:
        """Wrap colormapped channels as an RGB view of this view's space.

        Goes through :meth:`BrainSpace.wrap_rgb`, so neither subclass names a
        concrete RGB class and a new space's 2D view gets ``raw`` for free.
        """
        kws = self._raw_kwargs()
        # An explicit `alpha` attr overrides the colormap's alpha channel.
        alpha = kws.pop("alpha", a)
        return self.space.wrap_rgb(r, g, b, alpha, **kws)

    def _raw_kwargs(self) -> dict[str, Any]:
        """View metadata to forward onto the RGB view built by :attr:`raw`.

        Only the keys the RGB constructors accept; ``attrs`` may hold anything.
        """
        kws: dict[str, Any] = dict(
            state=self.state, description=self.description
        )
        for key in ("priority", "alpha"):
            if key in self.attrs:
                kws[key] = self.attrs[key]
        return kws


class Volume2D(Dataview2D[Volume], VolumetricView):
    """
    Contains two 3D volumes for simultaneous visualization. Includes information
    on how the volumes should be jointly colormapped.

    Parameters
    ----------
    dim1 : ndarray or Volume
        The first volume. Can be a 1D or 3D array (see Volume for details), or
        a Volume.
    dim2 : ndarray or Volume
        The second volume. Can be a 1D or 3D array (see Volume for details), or
        a Volume.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        dim1 must be a Volume from which the subject can be extracted. If dim1
        is a Volume and this is given too, the two must agree.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        dim1 must be a Volume from which the transform can be extracted.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the
        `default_cmap2d` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. Defaults to dim1's own vmin, which
        is the 1st percentile of its data.
    vmax : float, optional
        Maximum value in colormap for dim1. Defaults to dim1's own vmax, which
        is the 99th percentile of its data.
    vmin2 : float, optional
        Minimum value in colormap for dim2. Defaults to dim2's own vmin.
    vmax2 : float, optional
        Maximum value in colormap for dim2. Defaults to dim2's own vmax.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        dim1: Union[npt.NDArray, Volume],
        dim2: Union[npt.NDArray, Volume],
        subject: Optional[str] = None,
        xfmname: Optional[str] = None,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        chan1, chan2 = _resolve_2d_channels(
            dim1,
            dim2,
            channel_cls=Volume,
            fallback_space=lambda: VolumeSpace(
                _require(subject, "Subject"), _require(xfmname, "xfmname")
            ),
            subject=subject,
            space_kwargs={"xfmname": xfmname},
            ranges=((vmin, vmax), (vmin2, vmax2)),
        )

        super().__init__(
            chan1,
            chan2,
            description=description,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            **kwargs,
        )

    def __repr__(self) -> str:
        return "<2D volumetric data for (%s, %s)>" % (
            self.dim1.subject,
            self.dim1.xfmname,
        )

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        viewnode = super()._write_hdf(h5, name)
        viewnode[7] = json.dumps([[self.dim1.xfmname, self.dim2.xfmname]])
        return viewnode

    @property
    def raw(self) -> VolumeRGB:
        """VolumeRGB object containing the colormapped data from this object."""
        if self.dim1.xfmname != self.dim2.xfmname:
            raise ValueError(
                "Both Volumes must have same xfmname to generate single raw volume"
            )

        if (
            (self.dim1.linear and self.dim2.linear)
            and self.dim1.mask is not None
            and self.dim2.mask is not None
            and (self.dim1.mask.shape == self.dim2.mask.shape)
            and np.all(self.dim1.mask == self.dim2.mask)
        ):
            r, g, b, a = self._to_raw(self.dim1.data, self.dim2.data)
        else:
            r, g, b, a = self._to_raw(self.dim1.volume, self.dim2.volume)

        return cast(VolumeRGB, self._finish_raw(r, g, b, a))


class Vertex2D(Dataview2D[Vertex], SurfaceView):
    """
    Contains two vertex maps for simultaneous visualization. Includes information
    on how the maps should be jointly colormapped.

    Parameters
    ----------
    dim1 : ndarray or Vertex
        The first vertex map. Can be a 1D array (see Vertex for details), or
        a Vertex.
    dim2 : ndarray or Vertex
        The second vertex map. Can be a 1D array (see Vertex for details), or
        a Vertex.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        dim1 must be a Vertex from which the subject can be extracted. If dim1
        is a Vertex and this is given too, the two must agree.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the
        `default_cmap2d` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. Defaults to dim1's own vmin, which
        is the 1st percentile of its data.
    vmax : float, optional
        Maximum value in colormap for dim1. Defaults to dim1's own vmax, which
        is the 99th percentile of its data.
    vmin2 : float, optional
        Minimum value in colormap for dim2. Defaults to dim2's own vmin.
    vmax2 : float, optional
        Maximum value in colormap for dim2. Defaults to dim2's own vmax.
    **kwargs
        All additional arguments are stored in ``attrs``.
    """

    def __init__(
        self,
        dim1: Union[npt.NDArray, Vertex],
        dim2: Union[npt.NDArray, Vertex],
        subject: Optional[str] = None,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        chan1, chan2 = _resolve_2d_channels(
            dim1,
            dim2,
            channel_cls=Vertex,
            fallback_space=lambda: SurfaceSpace(_require(subject, "Subject")),
            subject=subject,
            space_kwargs={},
            ranges=((vmin, vmax), (vmin2, vmax2)),
        )

        super().__init__(
            chan1,
            chan2,
            description=description,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            **kwargs,
        )

    def __repr__(self) -> str:
        return "<2D vertex data for (%s)>" % self.dim1.subject

    @property
    def raw(self) -> VertexRGB:
        """VertexRGB object containing the colormapped data from this object."""
        r, g, b, a = self._to_raw(self.dim1.data, self.dim2.data)
        return cast(VertexRGB, self._finish_raw(r, g, b, a))



def _resolve_2d_channels(
    dim1: Union[npt.NDArray, ScalarT],
    dim2: Union[npt.NDArray, ScalarT],
    *,
    channel_cls: type[ScalarT],
    fallback_space: Callable[[], BrainSpace],
    subject: Optional[str],
    space_kwargs: dict[str, Any],
    ranges: tuple[
        tuple[Optional[float], Optional[float]],
        tuple[Optional[float], Optional[float]],
    ],
) -> tuple[ScalarT, ScalarT]:
    """Turn the two ``dim`` arguments into a matched pair of scalar views.

    Accepts either two already-built views or two raw arrays, and enforces that
    the two forms are not mixed. Replaces the branch pair that ``Volume2D`` and
    ``Vertex2D`` each carried separately.
    """
    kind = channel_cls.__name__
    if isinstance(dim1, channel_cls):
        if not isinstance(dim2, channel_cls) or dim2.subject != dim1.subject:
            raise TypeError("Invalid data for second dimension")
        if subject is not None and dim1.subject != subject:
            raise ValueError(
                "Subject in %s objects (%r) is different than specified subject (%r)"
                % (kind, dim1.subject, subject)
            )
        for key, value in space_kwargs.items():
            existing = getattr(dim1, key, None)
            if value is not None and existing != value:
                raise ValueError(
                    "%s in %s objects (%r) is different than specified %s (%r)"
                    % (key, kind, existing, key, value)
                )
        return dim1, dim2

    if isinstance(dim2, channel_cls):
        raise TypeError(
            "If dim2 is a %s, dim1 must be a %s as well" % (kind, kind)
        )

    # Wrapping goes through the space, so this never names a concrete view class.
    space = fallback_space()
    (vmin, vmax), (vmin2, vmax2) = ranges
    return (
        cast(ScalarT, space.wrap(np.asarray(dim1), vmin=vmin, vmax=vmax)),
        cast(ScalarT, space.wrap(np.asarray(dim2), vmin=vmin2, vmax=vmax2)),
    )


def _warn_non_perceptually_uniform_colormap(cmap: Any) -> None:
    mapping = {
        "BuOr_2D": "PU_BuOr_covar",
        "RdBu_covar": "PU_RdBu_covar",
        "RdBu_covar2": "PU_BuOr_covar",
        "RdBu_covar_alpha": "PU_RdBu_covar_alpha",
        "RdGn_covar": "PU_RdGn_covar",
        "hot_alpha": "fire_alpha",
    }
    if cmap in mapping:
        warnings.warn(
            "Colormap %r is not perceptually uniform. Consider using"
            " %r instead." % (cmap, mapping[cmap]),
            UserWarning,
        )
