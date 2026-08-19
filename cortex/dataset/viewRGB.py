from __future__ import annotations

import colorsys
import sys
import warnings
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import h5py
import numpy as np
import numpy.typing as npt

from .. import options
from ._hdf import _hash
from ._space import BrainSpace, SurfaceSpace, VolumeSpace
from .views import (
    DataviewJSON,
    Packable,
    RenderableView,
    ScalarView,
    SurfaceView,
    Vertex,
    Volume,
    VolumetricView,
    _require,
)

default_cmap = options.config.get("basic", "default_cmap")


ColorDtype = TypeVar("ColorDtype", int, float)
Color = tuple[ColorDtype, ColorDtype, ColorDtype]  # RGB color

#: A channel argument: either an array of values or an already-built scalar view.
ChannelLike = Union[npt.NDArray, ScalarView]

#: Covariant: the channels are read-only properties, so `DataviewRGB[Volume]` is
#: safely usable where `DataviewRGB[ScalarView]` is expected.
ScalarT = TypeVar("ScalarT", bound=ScalarView, covariant=True)


class Colors:
    """
    Set of known colors
    """

    RoseRed: Color[int] = (237, 35, 96)
    LimeGreen: Color[int] = (141, 198, 63)
    SkyBlue: Color[int] = (0, 176, 218)
    DodgerBlue: Color[int] = (30, 144, 255)
    Red: Color[int] = (255, 000, 000)
    Green: Color[int] = (000, 255, 000)
    Blue: Color[int] = (000, 000, 255)


#: The identity basis. Passing exactly these three means "no colour remapping",
#: which lets the channels be used as-is instead of going through color_voxels.
_RGB_BASIS = (Colors.Red, Colors.Green, Colors.Blue)


def _as_color(color: Union[Color[int], Sequence[int], npt.NDArray]) -> Color[int]:
    """Coerce a 3-sequence to a fixed-length colour tuple.

    ``tuple(color)`` would give ``tuple[int, ...]``, which is not assignable to
    ``Color[int]``; indexing three times is what makes the length static.
    """
    return (int(color[0]), int(color[1]), int(color[2]))


def RGB2HSV(color: Union[Color, npt.NDArray]) -> Color[float]:
    """
    Converts RGB to HSV

    Parameters
    ----------
    color : tuple<uint8, uint8, uint8>
        RGB color value

    Returns
    -------
    tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    """
    hue, saturation, value = colorsys.rgb_to_hsv(
        color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    )
    hue *= 360
    return (int(hue), saturation, value)


def HSV2RGB(color: Union[Color[float], npt.NDArray]) -> Color[int]:
    """
    Converts HSV to RGB

    Parameters
    ----------
    color : tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    Returns
    -------
    tuple<uint8, uint8, uint8>
        RGB color value
    """
    r, g, b = colorsys.hsv_to_rgb(color[0] / 360.0, color[1], color[2])
    return (int(r * 255), int(g * 255), int(b * 255))


class DataviewRGB(Packable, RenderableView, Generic[ScalarT]):
    """Abstract base class for RGB data views.

    Three scalar channels plus an alpha channel, carrying their own colours
    rather than a colormap. Deliberately does not inherit ``cmap``/``vmin``/
    ``vmax``: those live on :class:`~cortex.dataset.views.ScalarView`, so the
    ``except AttributeError`` blocks that used to stand in for "this is an RGB
    view" are gone.

    The channels are read-only. They are set once here, and keeping them
    read-only is what makes it sound to treat this class as covariant in its
    channel type.

    Generic in the channel type, so ``VolumeRGB.red`` and ``VolumeRGB.alpha`` are
    ``Volume`` and ``VertexRGB``'s are ``Vertex``, without either subclass
    re-declaring them. That matters most for ``alpha``: a property's return type
    cannot be narrowed by re-annotation, only by re-implementing the property,
    which is why ``alpha`` used to exist twice.
    """

    def __init__(
        self,
        red: ScalarT,
        green: ScalarT,
        blue: ScalarT,
        alpha: Optional[ChannelLike] = None,
        subject: Optional[str] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
    ) -> None:
        self._red = red
        self._green = green
        self._blue = blue

        if subject is not None and red.subject != subject:
            raise ValueError(
                "Subject in channel objects (%r) is different than specified "
                "subject (%r)" % (red.subject, subject)
            )

        # If movie, make sure each channel has the same number of time points
        if red.movie:
            if not (
                red.data.shape[0] == green.data.shape[0] == blue.data.shape[0]
            ):
                raise ValueError(
                    "For movie data, all three channels have to be the same length"
                )

        self._alpha: Optional[ChannelLike] = alpha
        self._alpha_cache: Optional[ScalarT] = None
        super().__init__(description=description, state=state, priority=priority)

    # ------------------------------------------------------------------
    # channels
    # ------------------------------------------------------------------
    @property
    def red(self) -> ScalarT:
        return self._red

    @property
    def green(self) -> ScalarT:
        return self._green

    @property
    def blue(self) -> ScalarT:
        return self._blue

    @property
    def space(self) -> BrainSpace:
        return self.red.space

    @property
    def movie(self) -> bool:
        return self.red.movie

    def uniques(self, collapse: bool = False) -> Iterator[Packable]:
        if collapse:
            yield self
        else:
            yield self.red
            yield self.green
            yield self.blue
            # `_alpha`, not `alpha`: the property always returns a view, so
            # testing it would always be true and the "no alpha" HDF slot would
            # never be written.
            if self._alpha is not None:
                yield self.alpha

    @property
    def name(self) -> str:
        """Content-addressed name, hashing the RGBA array this view ships.

        Both subclasses carried this, over ``volume`` and ``vertices``
        respectively -- the same array under two names. Hashing
        :attr:`~cortex.dataset.views.RenderableView.spatial_data` is sound here in
        a way it would not be on :class:`~cortex.dataset.views.ScalarView` (see
        :attr:`~cortex.dataset.views.Packable.name`): an RGB view's name is only
        ever a browser key, because its channels are what get written as HDF
        nodes, each under its own name.
        """
        return "__%s" % _hash(self.spatial_data)[:16]

    def __hash__(self) -> int:
        return hash(_hash(self.spatial_data))

    @property
    def raw(self) -> Self:
        return self

    def copy(self) -> Self:
        """A new view of the same kind over the same channels."""
        return self.__class__(
            self.red,
            self.green,
            self.blue,
            alpha=self._alpha,
            description=self.description,
            state=self.state,
            priority=self.priority,
        )

    # ------------------------------------------------------------------
    # alpha
    # ------------------------------------------------------------------
    def _default_alpha(self) -> npt.NDArray:
        """A fully-opaque alpha array shaped like this view's channels.

        Shaped from the channel's *stored* array, which is what a one-frame view
        and a movie differ in, so the alpha grows a frame axis exactly when the
        channels have one and :meth:`_rgba_stack` can stack the four together.
        ``VolumeRGB`` used to size this from ``red.volume`` -- the same values
        with the frame axis already prepended, so the alpha it wrapped came out
        as a one-frame movie. That unmasks back to an identical array, and an
        auto-generated alpha is never written to HDF or shipped on its own, so
        nothing outside this class could tell; it did mean the rule was written
        twice and differently.
        """
        return np.ones(self.red.data.shape)

    def _channel_stack(self) -> npt.NDArray:
        """The three channels stacked, for locating NaNs across them.

        Reads :attr:`~cortex.dataset.views.RenderableView.spatial_data`, so it
        does not matter whether the channels store volumes or vertices. For a
        single-frame surface view that adds a leading axis the stored arrays do
        not have; :meth:`_mask_alpha` already accepts a mask carrying one, which
        is what makes one implementation serve both spaces.
        """
        return np.array(
            [c.spatial_data for c in (self.red, self.green, self.blue)]
        )

    def _rgba_stack(self) -> npt.NDArray[np.uint8]:
        """Stack the four channels into a trailing RGBA axis.

        Each channel is read through
        :attr:`~cortex.dataset.views.RenderableView.spatial_data`, which is why
        this no longer takes the name of the accessor to use: it was passed the
        string ``"volume"`` or ``"vertices"`` by whichever subclass knew which of
        them its space stored. ``moveaxis`` rather than a hand-written transpose
        list: the intent is "put the channel axis last", and spelling that as
        ``[1, 2, 3, 4, 0]`` had to be rewritten per rank and was easy to get
        wrong.
        """
        channels = [
            _to_uint8(dv.spatial_data, dv.vmin, dv.vmax)
            for dv in (self.red, self.green, self.blue, self.alpha)
        ]
        return np.moveaxis(np.array(channels), 0, -1)

    def _wrap_alpha(self, data: npt.NDArray) -> ScalarT:
        """Wrap an alpha array as a scalar view in this view's space."""
        return cast(ScalarT, self.red.space.wrap(data, vmin=0, vmax=1))

    @property
    def alpha(self) -> ScalarT:
        """Alpha transparency, as a scalar view in this view's space.

        Derived lazily and memoized. Unlike the previous implementation this
        never writes into a caller-supplied view: if ``alpha=`` was given as a
        ``Volume``/``Vertex``, it is copied before the NaN mask is applied.
        """
        if self._alpha_cache is None:
            self._alpha_cache = self._resolve_alpha()
        return self._alpha_cache

    @alpha.setter
    def alpha(self, alpha: Optional[ChannelLike]) -> None:
        self._alpha = alpha
        self._alpha_cache = None

    def _resolve_alpha(self) -> ScalarT:
        spec = self._alpha
        view: ScalarT
        if spec is None:
            view = self._wrap_alpha(self._default_alpha())
        elif isinstance(spec, ScalarView):
            # Copy so that reading `.alpha` never mutates the caller's object.
            # cast, not isinstance: ScalarT is a TypeVar, so there is nothing to
            # test against at runtime. The space check below is the real guard.
            view = cast(ScalarT, spec.copy(np.array(spec.data, copy=True)))
        else:
            arr = np.asarray(spec)
            if arr.dtype != np.uint8 and (arr.min() < 0 or arr.max() > 1):
                warnings.warn(
                    "Some alpha values are outside the range of [0, 1]. "
                    "Consider passing a Volume/Vertex object as alpha with "
                    "explicit vmin, vmax keyword arguments.",
                    Warning,
                )
            view = self._wrap_alpha(np.array(arr, copy=True))

        # Channels that still hold NaN mark those positions transparent.
        stack = self._channel_stack()
        self._mask_alpha(view, np.isnan(stack).any(axis=0))
        # ...and positions that held NaN *before* the uint8 conversion, which is
        # unrecoverable from the channels themselves.
        self._mask_alpha(view, self._nan_mask)
        return view

    @staticmethod
    def _mask_alpha(
        view: ScalarView, mask: Optional[npt.NDArray[np.bool_]]
    ) -> None:
        """Drive alpha to its minimum wherever ``mask`` is set.

        Masks arrive in one of two shapes: matching the channel data, or with the
        extra leading axis that ``.volume``/``.vertices`` prepends. A mask that
        matches neither does not describe this alpha and is dropped, which is
        what the previous ``hasattr``-based version did.
        """
        if mask is None:
            return
        fill = view.vmin if view.vmin is not None else 0
        data = view.data
        if mask.shape == data.shape:
            data[mask] = fill
        elif (
            mask.ndim == data.ndim + 1
            and mask.shape[0] == 1
            and mask.shape[1:] == data.shape
        ):
            data[mask[0]] = fill

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self, simple: bool = False) -> DataviewJSON:
        """One implementation for every space; the space supplies its own keys.

        Mirrors :meth:`~cortex.dataset.views.ScalarView.to_json`. ``VolumeRGB``
        and ``VertexRGB`` each overrode this to add ``shape`` or
        ``split``/``frames`` -- the single question "how does the browser unpack
        this array", answered per space -- and ``VolumeRGB`` additionally added
        the ``xfm`` that its space already knows how to emit.
        """
        sdict = super().to_json(simple=simple)

        if simple:
            sdict["name"] = self.name
            sdict["subject"] = self.subject
            sdict["min"] = 0
            sdict["max"] = 255
            sdict.update(self.space.describe_layout(self.spatial_data))
        else:
            sdict["data"] = [self.name]
            sdict["cmap"] = [default_cmap]
            sdict["vmin"] = [0]
            sdict["vmax"] = [255]
            sdict.update(self.space.to_json())
        return sdict

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        """Write the channels as ``/data`` nodes plus one ``/views`` record.

        Slot 7 comes from :attr:`~cortex.dataset._space.BrainSpace.view_xfmname`,
        exactly as in :meth:`~cortex.dataset.views.ScalarView._write_hdf`. This
        was two methods and an ``xfmname`` parameter, the outer one existing only
        so that ``VolumeRGB`` could override it to pass ``[self.xfmname]`` in.
        """
        self.red._write_data_hdf(h5)
        self.green._write_data_hdf(h5)
        self.blue._write_data_hdf(h5)

        alpha_name = None
        if self._alpha is not None:
            alpha = self.alpha
            alpha._write_data_hdf(h5)
            alpha_name = alpha.name

        data = [self.red.name, self.green.name, self.blue.name, alpha_name]
        return self._write_view_node(
            h5, name=name, data=[data], xfmname=self.space.view_xfmname
        )

    @staticmethod
    def color_voxels(
        channel1: ChannelLike,
        channel2: ChannelLike,
        channel3: ChannelLike,
        channel1color: Color[int],
        channel2color: Color[int],
        channel3Color: Color[int],
        value_max: Optional[float],
        saturation_max: float,
        vmin: Optional[Union[float, tuple[float, float, float]]],
        vmax: Optional[Union[float, tuple[float, float, float]]],
        autorange: Literal["shared", "individual"] = "individual",
        alpha: Optional[ChannelLike] = None,
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        ChannelLike,
    ]:
        """
        Colors voxels in 3 color dimensions but not necessarily canonical red,
        green, and blue.

        Parameters
        ----------
        channel1 : ndarray or Volume or Vertex
            voxel values for first channel
        channel2 : ndarray or Volume or Vertex
            voxel values for second channel
        channel3 : ndarray or Volume or Vertex
            voxel values for third channel
        channel1color : tuple<uint8, uint8, uint8>
            color in RGB for first channel
        channel2color : tuple<uint8, uint8, uint8>
            color in RGB for second channel
        channel3Color : tuple<uint8, uint8, uint8>
            color in RGB for third channel
        value_max : float, optional
            Maximum HSV value for voxel colors. If not given, will be the value of
            the average of the three channel colors.
        saturation_max : float [0, 1]
            Maximum HSV saturation for voxel colors.
        vmin : float or tuple of float, optional
            Lower bound(s) that map to 0 in each color channel. If a single float,
            the same lower bound is used for all three channels. If a tuple of
            three floats, each channel uses its respective value. If None, the
            lower bound is auto-determined based on ``autorange``.
        vmax : float or tuple of float, optional
            Upper bound(s) that map to 255 in each color channel. If a single
            float, the same upper bound is used for all three channels. If a tuple
            of three floats, each channel uses its respective value. If None, the
            upper bound is auto-determined based on ``autorange``.
        autorange : 'shared' or 'individual'
            How to auto-determine bounds when vmin or vmax is None. 'shared'
            computes the 1st and 99th percentile across all three channels
            combined. 'individual' computes per-channel 1st and 99th percentiles.
            Overridden when vmin and vmax are both provided.
        alpha : ndarray or Volume or Vertex, optional
            Alpha values for each voxel. If None, alpha is set to 1 for all voxels.

        Returns
        -------
        red : ndarray of channel1.shape
            uint8 array of red values
        green : ndarray of channel1.shape
            uint8 array of green values
        blue : ndarray of channel1.shape
            uint8 array of blue values
        alpha : ndarray or Volume or Vertex
            A copy of the alpha that was passed in (or a fully-opaque uint8 array
            if none was), with NaN positions set to 0. The input is never
            mutated.
        """
        # normalize each channel to [0, 1]
        data1 = _channel_data(channel1).astype(float)
        data2 = _channel_data(channel2).astype(float)
        data3 = _channel_data(channel3).astype(float)

        if (data1.shape != data2.shape) or (data2.shape != data3.shape):
            raise ValueError("Volumes are of different shapes")

        # Create an alpha mask now, before casting nans to 0
        # Voxels with at least one channel equal to NaN will be masked out.
        mask = np.isnan(np.array([data1, data2, data3])).any(axis=0)
        # Now convert NaNs to num for all channels
        data1 = np.nan_to_num(data1)
        data2 = np.nan_to_num(data2)
        data3 = np.nan_to_num(data3)

        channel_vmins = _expand_bounds(vmin)
        channel_vmaxs = _expand_bounds(vmax)

        # Auto-determine any None bounds
        needs_auto_min = any(v is None for v in channel_vmins)
        needs_auto_max = any(v is None for v in channel_vmaxs)

        if needs_auto_min or needs_auto_max:
            if autorange == "shared":
                all_data = np.concatenate(
                    [data1.ravel(), data2.ravel(), data3.ravel()]
                )
                shared_min = float(np.percentile(all_data, 1))
                shared_max = float(np.percentile(all_data, 99))
                channel_vmins = [
                    shared_min if v is None else v for v in channel_vmins
                ]
                channel_vmaxs = [
                    shared_max if v is None else v for v in channel_vmaxs
                ]
            elif autorange == "individual":
                for i, data in enumerate([data1, data2, data3]):
                    if channel_vmins[i] is None:
                        channel_vmins[i] = float(np.percentile(data.ravel(), 1))
                    if channel_vmaxs[i] is None:
                        channel_vmaxs[i] = float(np.percentile(data.ravel(), 99))
            else:
                raise ValueError("autorange must be 'shared' or 'individual'")

        normalized = []
        for channel, (data, channel_min, channel_max) in enumerate(
            zip([data1, data2, data3], channel_vmins, channel_vmaxs), start=1
        ):
            assert channel_min is not None and channel_max is not None
            channel_range = channel_max - channel_min
            if channel_range == 0:
                warnings.warn(
                    "Channel {} has no dynamic range (vmin == vmax) and will be "
                    "zeroed out".format(channel)
                )
                normalized.append(np.zeros_like(data))
            else:
                normalized.append((data - channel_min) / channel_range)
        data1, data2, data3 = (np.clip(d, 0, 1) for d in normalized)

        color1 = np.array(channel1color)
        color2 = np.array(channel2color)
        color3 = np.array(channel3Color)

        averageColor = (color1 + color2 + color3) / 3

        if value_max is None:
            _, _, value_max = RGB2HSV(averageColor)

        red = np.zeros_like(data1, np.uint8)
        green = np.zeros_like(data1, np.uint8)
        blue = np.zeros_like(data1, np.uint8)
        for i in range(data1.size):
            this_color = (
                data1.flat[i] * color1
                + data2.flat[i] * color2
                + data3.flat[i] * color3
            )
            this_color /= 3.0
            if (value_max != 1.0) or (saturation_max != 1.0):
                hue, saturation, value = RGB2HSV(this_color)
                saturation /= saturation_max
                value /= value_max
                if saturation > 1:
                    saturation = 1.0
                if value > 1:
                    value = 1.0
                this_color = np.array(HSV2RGB((hue, saturation, value)))
            red.flat[i] = this_color[0]
            green.flat[i] = this_color[1]
            blue.flat[i] = this_color[2]

        # Now make an alpha volume. Always a copy: the previous version wrote
        # `alpha[mask] = 0` straight into a caller-owned array, and raised
        # TypeError outright for a Volume/Vertex alpha.
        alpha_out: ChannelLike
        if alpha is None:
            alpha_out = np.ones_like(red, np.uint8) * 255
            alpha_out[mask] = 0
        elif isinstance(alpha, ScalarView):
            alpha_out = alpha.copy(np.array(alpha.data, copy=True))
            if mask.shape == alpha_out.data.shape:
                alpha_out.data[mask] = 0
        else:
            alpha_out = np.array(alpha, copy=True)
            if mask.shape == alpha_out.shape:
                alpha_out[mask] = 0

        return red, green, blue, alpha_out


def _channel_data(channel: ChannelLike) -> npt.NDArray:
    return channel.data if isinstance(channel, ScalarView) else np.asarray(channel)


def _expand_bounds(
    bound: Optional[Union[float, tuple[float, float, float]]],
) -> list[Optional[float]]:
    """Expand a scalar / 3-tuple / None bound into a per-channel list."""
    if isinstance(bound, (int, float)):
        return [float(bound)] * 3
    if bound is not None:
        return [float(v) for v in bound]
    return [None, None, None]


def _resolve_rgb_channels(
    channels: tuple[ChannelLike, ChannelLike, ChannelLike],
    *,
    channel_cls: type[ScalarT],
    fallback_space: Callable[[], BrainSpace],
    subject: Optional[str],
    space_kwargs: dict[str, Any],
    colors: tuple[Sequence[int], Sequence[int], Sequence[int]],
    max_color_value: Optional[float],
    max_color_saturation: float,
    vmin: Optional[Union[float, tuple[float, float, float]]],
    vmax: Optional[Union[float, tuple[float, float, float]]],
    autorange: Literal["shared", "individual"],
    alpha: Optional[ChannelLike],
) -> tuple[ScalarT, ScalarT, ScalarT, Optional[ChannelLike]]:
    """Turn three channel arguments into a matched triple of scalar views.

    Replaces the four-branch shape (channel-object vs ndarray, x, identity basis
    vs remap) that ``VolumeRGB`` and ``VertexRGB`` each carried separately -- the
    same logic written out eight times.
    """
    chan1, chan2, chan3 = channels
    kind = channel_cls.__name__
    # Coerced here rather than by each caller: a list or ndarray has to become a
    # fixed-length tuple before it can be compared against the identity basis.
    basis = (_as_color(colors[0]), _as_color(colors[1]), _as_color(colors[2]))

    template: Optional[ScalarT] = None
    if isinstance(chan1, channel_cls):
        template = chan1
        for pos, chan in (("2", chan2), ("3", chan3)):
            if not isinstance(chan, channel_cls):
                raise TypeError(
                    "Data channel %s is not a %s object" % (pos, kind)
                )
            if chan.subject != chan1.subject:
                raise TypeError(
                    "Data channel %s is from a different subject" % pos
                )
        if subject is not None and chan1.subject != subject:
            raise ValueError(
                "Subject in %s objects (%r) is different than specified subject "
                "(%r)" % (kind, chan1.subject, subject)
            )
        for key, value in space_kwargs.items():
            existing = getattr(chan1, key, None)
            if value is not None and existing != value:
                raise ValueError(
                    "%s in %s objects (%r) is different than specified %s (%r)"
                    % (key, kind, existing, key, value)
                )
    else:
        if subject is None:
            raise TypeError("Subject name is required")
        for key, value in space_kwargs.items():
            _require(value, key)
        if not isinstance(chan2, np.ndarray) or not isinstance(chan3, np.ndarray):
            raise TypeError(
                "Data channels must be numpy arrays if channel1 is a numpy array"
            )

    identity_basis = (
        basis == _RGB_BASIS
        and vmin is None
        and vmax is None
        and autorange == "individual"
    )

    # Wrapping goes through the space, so this function never names a concrete
    # view class -- which is what lets a new space reuse it unchanged.
    space = template.space if template is not None else fallback_space()

    def wrap(data: npt.NDArray) -> ScalarT:
        return cast(ScalarT, space.wrap(data))

    if identity_basis:
        # R/G/B basis can be passed straight through.
        if template is not None:
            assert isinstance(chan2, channel_cls)
            assert isinstance(chan3, channel_cls)
            return template, chan2, chan3, alpha
        return (
            wrap(np.asarray(chan1)),
            wrap(np.asarray(chan2)),
            wrap(np.asarray(chan3)),
            alpha,
        )

    red, green, blue, alpha_out = DataviewRGB.color_voxels(
        chan1,
        chan2,
        chan3,
        basis[0],
        basis[1],
        basis[2],
        max_color_value,
        max_color_saturation,
        vmin,
        vmax,
        autorange,
        alpha=alpha,
    )
    return wrap(red), wrap(green), wrap(blue), alpha_out


class VolumeRGB(DataviewRGB[Volume], VolumetricView):
    """
    Contains RGB (or RGBA) colors for each voxel in a volumetric dataset.
    Includes information about the subject and transform for the data.

    Three data channels are mapped into a 3D color set. By default the data
    channels are mapped on to red, green, and blue. They can also be mapped to
    be different colors as specified, and then linearly combined.

    Each data channel is represented as a separate Volume object (these can
    either be supplied explicitly as Volume objects or implicitly as numpy
    arrays). By default, each channel's range is determined independently from
    the data. Use ``vmin``/``vmax`` to specify explicit bounds, or ``autorange``
    to control how bounds are auto-determined.

    Parameters
    ----------
    channel1 : ndarray or Volume
        Array or Volume for the first data channel for each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    channel2 : ndarray or Volume
        Array or Volume for the second data channel for each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    channel3 : ndarray or Volume
        Array or Volume for the third data channel for or each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        channel1 must be a Volume from which the subject can be extracted.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        channel1 must be a Volume from which the transform can be extracted.
    alpha : ndarray or Volume, optional
        Array or Volume that represents the alpha component of the color for each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume. If
        None, all voxels will be assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        Passed through to the webgl viewer.
    channel1color : tuple<uint8, uint8, uint8>
        RGB color to use for the first data channel
    channel2color : tuple<uint8, uint8, uint8>
        RGB color to use for the second data channel
    channel3color : tuple<uint8, uint8, uint8>
        RGB color to use for the third data channel
    max_color_value : float [0, 1], optional
        Maximum HSV value for voxel colors. If not given, will be the value of
        the average of the three channel colors.
    max_color_saturation: float [0, 1]
        Maximum HSV saturation for voxel colors.
    vmin : float or tuple of float, optional
        Lower bound(s) that map to 0 in each color channel. If a single float, the
        same lower bound is used for all three channels. If a tuple of three
        floats, each channel uses its respective value. If None, the lower bound
        is auto-determined based on ``autorange``.
    vmax : float or tuple of float, optional
        Upper bound(s) that map to 255 in each color channel. If a single float,
        the same upper bound is used for all three channels. If a tuple of three
        floats, each channel uses its respective value. If None, the upper bound
        is auto-determined based on ``autorange``.
    autorange : 'shared' or 'individual'
        How to auto-determine bounds when vmin or vmax is None. 'shared' computes
        the 1st and 99th percentile across all three channels combined.
        'individual' computes per-channel 1st and 99th percentiles. Overridden
        when vmin and vmax are both provided. Default is 'individual'.
    priority : int, optional
        Priority for display ordering. Default is 1.
    """

    def __init__(
        self,
        channel1: Union[npt.NDArray, Volume],
        channel2: Union[npt.NDArray, Volume],
        channel3: Union[npt.NDArray, Volume],
        subject: Optional[str] = None,
        xfmname: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, Volume]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple[float, float, float]]] = None,
        vmax: Optional[Union[float, tuple[float, float, float]]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
    ) -> None:
        red, green, blue, resolved_alpha = _resolve_rgb_channels(
            (channel1, channel2, channel3),
            channel_cls=Volume,
            fallback_space=lambda: VolumeSpace(
                _require(subject, "Subject"), _require(xfmname, "xfmname")
            ),
            subject=subject,
            space_kwargs={"xfmname": xfmname},
            colors=(channel1color, channel2color, channel3color),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin,
            vmax=vmax,
            autorange=autorange,
            alpha=alpha,
        )

        if not red.xfmname == green.xfmname == blue.xfmname:
            raise ValueError("Cannot handle different transforms per volume")

        super().__init__(
            red,
            green,
            blue,
            alpha=resolved_alpha,
            subject=subject,
            description=description,
            state=state,
            priority=priority,
        )

    @property
    def xfmname(self) -> str:
        """Transform name, shared by all three channels.

        Derived rather than stored: the constructor already rejects channels with
        differing transforms, so a copy would only be able to disagree. Mirrors
        :attr:`Volume2D.xfmname`, which has always been a property.
        """
        return self.red.xfmname

    @property
    def volume(self) -> npt.NDArray[np.uint8]:
        """5-dimensional volume (t, z, y, x, rgba) with data that has been mapped
        into 8-bit unsigned integers that correspond to colors.
        """
        return self._rgba_stack()

    def __repr__(self) -> str:
        return "<RGB volumetric data for (%s, %s)>" % (
            self.red.subject,
            self.red.xfmname,
        )


class VertexRGB(DataviewRGB[Vertex], SurfaceView):
    """
    Contains RGB (or RGBA) colors for each vertex in a surface dataset.
    Includes information about the subject.

    Three data channels are mapped into a 3D color set. By default the data
    channels are mapped on to red, green, and blue. They can also be mapped to
    be different colors as specified, and then linearly combined.

    Each color channel is represented as a separate Vertex object (these can
    either be supplied explicitly as Vertex objects or implicitly as np
    arrays). By default, each channel's range is determined independently from
    the data. Use ``vmin``/``vmax`` to specify explicit bounds, or ``autorange``
    to control how bounds are auto-determined.

    Parameters
    ----------
    red : ndarray or Vertex
        Array or Vertex that represents the first data channel for each
        vertex. Can be a 1D array (see Vertex for details), or a Vertex.
    green : ndarray or Vertex
        Array or Vertex that represents the second data channel for each
        vertex. Can be a 1D array (see Vertex for details), or a Vertex.
    blue : ndarray or Vertex
        Array or Vertex that represents the third data channel for each
        vertex. Can be a 1D array (see Vertex for details), or a Vertex.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        red must be a Vertex from which the subject can be extracted.
    alpha : ndarray or Vertex, optional
        Array or Vertex that represents the alpha component of the color for each
        vertex. Can be a 1D array (see Vertex for details), or a Vertex. If
        None, all vertices will be assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        Passed through to the webgl viewer.
    channel1color : tuple<uint8, uint8, uint8>
        RGB color to use for the first data channel
    channel2color : tuple<uint8, uint8, uint8>
        RGB color to use for the second data channel
    channel3color : tuple<uint8, uint8, uint8>
        RGB color to use for the third data channel
    max_color_value : float [0, 1], optional
        Maximum HSV value for vertex colors. If not given, will be the value of
        the average of the three channel colors.
    max_color_saturation: float [0, 1]
        Maximum HSV saturation for vertex colors.
    vmin : float or tuple of float, optional
        Lower bound(s) that map to 0 in each color channel. If a single float, the
        same lower bound is used for all three channels. If a tuple of three
        floats, each channel uses its respective value. If None, the lower bound
        is auto-determined based on ``autorange``.
    vmax : float or tuple of float, optional
        Upper bound(s) that map to 255 in each color channel. If a single float,
        the same upper bound is used for all three channels. If a tuple of three
        floats, each channel uses its respective value. If None, the upper bound
        is auto-determined based on ``autorange``.
    autorange : 'shared' or 'individual'
        How to auto-determine bounds when vmin or vmax is None. 'shared' computes
        the 1st and 99th percentile across all three channels combined.
        'individual' computes per-channel 1st and 99th percentiles. Overridden
        when vmin and vmax are both provided. Default is 'individual'.
    priority : int, optional
        Priority for display ordering. Default is 1.
    """

    def __init__(
        self,
        red: Union[npt.NDArray, Vertex],
        green: Union[npt.NDArray, Vertex],
        blue: Union[npt.NDArray, Vertex],
        subject: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, Vertex]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple[float, float, float]]] = None,
        vmax: Optional[Union[float, tuple[float, float, float]]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
    ) -> None:
        r, g, b, resolved_alpha = _resolve_rgb_channels(
            (red, green, blue),
            channel_cls=Vertex,
            fallback_space=lambda: SurfaceSpace(_require(subject, "Subject")),
            subject=subject,
            space_kwargs={},
            colors=(channel1color, channel2color, channel3color),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin,
            vmax=vmax,
            autorange=autorange,
            alpha=alpha,
        )

        super().__init__(
            r,
            g,
            b,
            alpha=resolved_alpha,
            subject=subject,
            description=description,
            state=state,
            priority=priority,
        )

    @property
    def vertices(self) -> npt.NDArray[np.uint8]:
        """3-dimensional array (t, v, rgba) with data that has been mapped
        into 8-bit unsigned integers that correspond to colors.
        """
        return self._rgba_stack()

    @property
    def space(self) -> SurfaceSpace:
        """Narrowed from :class:`BrainSpace`, as on :class:`Vertex`.

        Sound because ``DataviewRGB[Vertex]`` fixes the channel type, so
        ``red.space`` is a :class:`SurfaceSpace`; the base cannot say so because
        it is generic over the channel.
        """
        return self.red.space

    @property
    def left(self) -> npt.NDArray[np.uint8]:
        """Colours for only the left hemisphere vertices."""
        # Asks its own space rather than reaching through `self.red` for the
        # geometry, which was the only reason a channel had to be consulted here.
        return cast(npt.NDArray[np.uint8], self.space.split_hemispheres(self.vertices)[0])

    @property
    def right(self) -> npt.NDArray[np.uint8]:
        """Colours for only the right hemisphere vertices."""
        return cast(npt.NDArray[np.uint8], self.space.split_hemispheres(self.vertices)[1])

    def __repr__(self) -> str:
        return "<RGB vertex data for (%s)>" % (self.subject,)


def _to_uint8(
    data: npt.NDArray, vmin: Optional[float], vmax: Optional[float]
) -> npt.NDArray[np.uint8]:
    """Scale a channel into [0, 255] uint8, honouring explicit bounds.

    Guards the case where exactly one of vmin/vmax is set, which previously
    evaluated ``vmax - None``.
    """
    if data.dtype == np.uint8:
        return data.copy()

    out = data.astype("float32", copy=True)
    lo = vmin
    hi = vmax
    # Numpy scalars, not Python floats: see ScalarView._resolve_percentiles for
    # why the NEP 50 promotion path has to be preserved here.
    if lo is None:
        lo = out.min() if out.min() < 0 else 0.0
    out -= lo
    if hi is None:
        scale = out.max() if out.max() > 1 else 1.0
    else:
        scale = hi - lo
    if scale != 0:
        out /= scale
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
