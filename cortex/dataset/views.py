from __future__ import annotations

import difflib
import glob
import inspect
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import lru_cache
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
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
from ._hdf import _hash, _hdf_write
from ._space import (
    BrainSpace,
    MaskSpec,
    SurfaceSpace,
    VolumeSpace,
    registered_spaces,
)

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


def _json_default(obj: Any) -> Any:
    """Serialise a numpy scalar as its Python equivalent.

    ``json.dumps`` calls this only for objects it cannot handle itself, which is
    exactly the set that needs converting. Of the numpy scalar types only
    ``np.float64`` and ``np.str_`` subclass a builtin json understands (``float``
    and ``str``), so those never arrive here and their output is byte-for-byte what
    it always was; ``np.float32``, ``np.float16``, every integer width and
    ``np.bool_`` all do arrive, since none of them subclasses anything.

    That asymmetry is the whole bug. ``vmin``/``vmax`` hold whatever
    ``np.percentile`` returned, so on float32 data they are ``np.float32`` and
    saving raised ``TypeError: Object of type float32 is not JSON serializable``.
    float64 data only ever worked by the accident of that subclassing.

    Converting here rather than in :meth:`ScalarView._resolve_percentiles` is
    deliberate: the numpy scalar has to survive on the *view*, because under NEP 50
    it is a strong operand and `channel -= vmin` therefore computes in float64.
    Rounding that to a Python float changes a handful of voxels by one LSB, and
    channel names are content hashes -- it would silently rename on-disk nodes.
    See the numerics note in INHERITANCE.md.

    ``np.ndarray`` is not an ``np.generic``, so an array reaching a JSON slot
    still raises. It has no representation in any of these slots, and quietly
    inventing one would hide the mistake.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(
        "Object of type %s is not JSON serializable" % type(obj).__name__
    )


def _dumps(obj: Any) -> str:
    """``json.dumps`` for anything crossing a wire or an HDF slot.

    Every ``json.dumps`` on a write path in this package goes through here, so
    there is one answer to "what happens to a numpy scalar".
    """
    return json.dumps(obj, default=_json_default)


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


class HasSubject(Protocol):
    """Anything that knows which subject it belongs to.

    A *static-only* protocol: deliberately not ``runtime_checkable``, so
    ``isinstance`` against it raises ``TypeError`` rather than doing a
    presence-only ``hasattr`` sweep. :class:`Dataview` claims it explicitly, which
    is what makes the claim visible here and machine-checked -- a subclass that
    failed to provide ``subject`` would be abstract and uninstantiable.

    If another protocol is ever needed, compose by inheritance rather than
    re-declaring ``subject`` -- and note the ``Protocol`` base has to be repeated:
    ``class Blendable(HasSubject, Protocol)``. Writing
    ``class Blendable(HasSubject)`` silently produces an ordinary instantiable
    class instead of a protocol.
    """

    @property
    def subject(self) -> str:
        """Subject identifier. Must exist in the pycortex database."""


#: Similarity above which an unrecognized keyword is treated as a misspelling
#: rather than as metadata. Both bounds are real cases, which is why it is not a
#: round number: ``cmpa``/``cmap`` -- the motivating typo -- scores exactly 0.75,
#: while ``name``/``xfmname`` scores 0.727 and ``name`` is a plausible attr.
_KWARG_TYPO_CUTOFF = 0.75


@lru_cache(maxsize=None)
def _constructor_params(cls: type) -> frozenset[str]:
    """Every named parameter of ``cls``'s constructor chain.

    Walks the MRO because a view's parameters are split across it: ``cmap`` is
    named by ``Volume.__init__``, ``description`` only by ``Dataview.__init__``,
    and a typo of either should be caught. Cached per class -- ``inspect`` is far
    too slow to run per constructed view, and the answer cannot change.
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover - C-level __init__
            continue
        names.update(
            name
            for name, param in params.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        )
    return frozenset(names)


def _reject_misspelled_kwargs(cls: type, kwargs: Mapping[str, Any]) -> None:
    """Raise on a keyword that is a near-miss of one this constructor accepts.

    ``attrs`` is a genuine feature -- ``stim`` is read by ``webgl/view.py``,
    ``alpha`` by ``Dataview2D._raw_kwargs``, and users hang their own metadata
    there -- so unknown keywords cannot simply be rejected. What *can* be
    rejected is a keyword that looks like a misspelling of a real parameter, which
    is the case that used to be silent: ``Volume(data, subj, xfm, cmpa="hot")``
    built a view with the default colormap and an attribute nobody would ever
    read. Anything not resembling a parameter name still lands in ``attrs``.

    Pass such a key through ``attrs=`` if it really is metadata.
    """
    known = _constructor_params(cls)
    for key in kwargs:
        if key in known:
            continue
        close = difflib.get_close_matches(key, known, n=1, cutoff=_KWARG_TYPO_CUTOFF)
        if close:
            raise TypeError(
                "%s() got an unexpected keyword argument %r; did you mean %r? "
                "Unrecognized keywords are stored as `attrs` metadata, so this "
                "would otherwise have been accepted and silently ignored. Pass "
                "attrs={%r: ...} if it is deliberate."
                % (cls.__name__, key, close[0], key)
            )


class Dataview(HasSubject, ABC):
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

    def __init__(
        self,
        description: str = "",
        state: Any = None,
        attrs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        attrs : mapping, optional
            Metadata to carry verbatim, *not* checked against the constructor's
            parameter names. This is how ``copy()`` and the HDF factories pass
            metadata that already lives on a view: it is data at that point, so
            re-validating it would make a file written by an older pycortex
            unloadable. Keywords a caller *types* go through ``**kwargs``, which
            is checked -- see :func:`_reject_misspelled_kwargs`.
        """
        self.state = state
        if kwargs:
            _reject_misspelled_kwargs(type(self), kwargs)
        # Carried metadata wins over a typed keyword on the one key where the two
        # can collide. `priority` is both an attr and a named parameter of the RGB
        # constructors, whose default is indistinguishable from an explicit value
        # -- so ordering it the other way meant `copy()` and an HDF reload both
        # reset a priority of 3 back to 1, since neither passes the parameter.
        self.attrs: dict[str, Any] = dict(kwargs)
        if attrs is not None:
            self.attrs.update(attrs)
        self.attrs.setdefault("priority", 1)
        self.description = description

    @property
    @abstractmethod
    def space(self) -> BrainSpace:
        """Where this view's data lives.

        The space is the *open* axis of the package: adding a new kind of brain
        data means adding a :class:`~cortex.dataset._space.BrainSpace` subclass,
        not reimplementing colormapping, HDF and JSON three more times.

        Read this when the question is genuinely about the space -- checking that
        two views share a transform, say. For "can I render this", use
        ``isinstance(view, RenderableView)``, which is a different question: a
        space says where the data lives, a spatial interface says how it is sampled.
        """

    @property
    def subject(self) -> str:
        """Subject identifier. Must exist in the pycortex database."""
        return self.space.subject

    @property
    @abstractmethod
    def raw(self) -> DataviewRGB:
        """This view rendered to 8-bit RGBA channels."""

    @abstractmethod
    def uniques(self, collapse: bool = False) -> Iterator["Packable"]:
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
        view[0] = _dumps(data)
        view[1] = self.description
        view[2] = "null"
        view[3:5] = "null"
        self._write_cmap_slots(view)
        view[5] = _dumps(self.state)
        view[6] = _dumps(self.attrs)
        view[7] = _dumps(xfmname)
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
        # Slots 2-4 hold the JSON `null` written by `_write_view_node` for an RGB
        # view, which carries no colormap at all. Reading that back as
        # `cmap=None` and passing it on meant the RGB constructors -- which have
        # no `cmap` parameter -- were handed one, and the factories below had to
        # filter their kwargs down to a hardcoded whitelist to get rid of it
        # again. Not inventing the argument in the first place is what lets them
        # forward what they were given.
        colormapped = cmap is not None
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
        cmap_kwargs: dict[str, Any] = {}
        if colormapped:
            cmap_kwargs = dict(cmap=cmap[0], vmin=vmin[0], vmax=vmax[0])
        return _from_hdf_view(
            node.file,
            data[0],
            xfmname=xfm,
            description=desc,
            state=state,
            subject=subject,
            # Metadata off disk, so `attrs=` rather than splatted: a file written
            # by an older pycortex may hold a key that today's constructor would
            # flag as a misspelling, and `cortex.load` swallows per-view
            # exceptions, so raising here would drop the view silently.
            attrs=attrs,
            **cmap_kwargs,
        )


class Packable(Dataview):
    """A view that is one addressable data array, as :meth:`Dataview.uniques` yields.

    This is the *unit of transport*: the thing that gets a single
    content-addressed :attr:`name`, is written as one node under HDF ``/data``,
    and reaches the browser as one texture or vertex-attribute array. Both the
    scalar column and the RGB column are packable -- an RGB view ships as a
    single four-channel array -- but :class:`~cortex.dataset.view2D.Dataview2D`
    is not: a 2D view has no array of its own, only the two channels it
    decomposes into, which is why it has no ``name``.

    It exists because ``uniques()`` used to be annotated ``Iterator[Dataview]``,
    which is wider than the truth and misses the one member every consumer
    immediately reaches for. ``webgl.data.Package`` type-checked only because the
    list it built was ``Any``; nothing warned that ``Dataview`` has no ``name``.

    Note the asymmetry in what ``name`` addresses, which is why this class only
    promises the name and not what it hashes: for a scalar view it is both the
    HDF node name and the browser key, while an RGB view writes its channels as
    four separate HDF nodes and uses its own ``name`` only as the browser key.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Content-addressed identity of this view's data array.

        Content-addressed so that two views over identical data share a node
        instead of duplicating it. Do not reimplement this as a hash of
        :attr:`RenderableView.spatial_data`: for a masked volume that is the
        unmasked 3-D array, not the flat stored one, so it would silently rename
        every existing HDF node.
        """


class RenderableView(Dataview):
    """A view a renderer can sample, whatever space it lives in.

    The common base of every spatial interface. Its purpose is to let callers stop
    enumerating the spatial kinds: :func:`as_renderable` is one
    ``isinstance`` against this, and the flatmap renderer reads
    :attr:`spatial_data` instead of asking which spatial kind it holds.

    That matters because an ``if volumetric / else surface`` fork silently encodes
    "there are exactly two spatial kinds". A third would have taken the ``else`` branch
    and then failed -- or worse, drawn the wrong thing. Nothing branches on the spatial
    kind any more, so a new one works everywhere by implementing this one member (plus
    its space's :attr:`~cortex.dataset._space.BrainSpace.xfmname`).
    """

    @property
    @abstractmethod
    def spatial_data(self) -> npt.NDArray:
        """The array a renderer samples, with a leading time axis.

        This is *the* member a view implements to be renderable. Each spatial
        interface then exposes it under the name its space has always used --
        :attr:`VolumetricView.volume`, :attr:`SurfaceView.vertices` -- rather than
        the reverse, which is how it was first written and which cost a property
        per space per column: an RGB view and a 2D view each had to publish the
        same array twice over, once as ``volume`` and once as ``vertices``, purely
        to satisfy the interface it happened to inherit.

        Paired with ``view.space.xfmname``, which says what to sample *through*,
        and named to match :attr:`BrainSpace.spatial_shape`, which is the shape of
        one frame of it.
        """


class VolumetricView(RenderableView):
    """A view whose data can be sampled as a volume under a transform.

    One of the two *spatial interfaces* of the 2x3 grid. The columns are already classes
    (:class:`ScalarView`, :class:`~cortex.dataset.view2D.Dataview2D`,
    :class:`~cortex.dataset.viewRGB.DataviewRGB`); the spatial axis had no type at all,
    which is why consumers had to duck-type ``hasattr(braindata, "xfmname")``.

    This is a stateless interface: no ``__init__``, no attributes of its own, and
    no cooperative ``super()`` chain. It is therefore *not* a return to the
    multiple inheritance this package was restructured to remove -- that was
    pathological because ``BrainData`` and ``Dataview`` each carried state and
    called ``super()`` methods that resolved only through a subclass's MRO.

    Being a real base class rather than a ``Protocol``, ``isinstance`` against it
    is a genuine class check. A ``runtime_checkable`` protocol can only test for
    the *presence* of attribute names, so an object with an unrelated ``volume``
    would satisfy it; and because the members here are abstract, a subclass that
    forgets one cannot be instantiated at all.
    """

    @property
    def xfmname(self) -> str:
        """Transform name. Must exist in the pycortex database.

        The space owns this -- it is half of what identifies a volumetric space --
        so one implementation serves every volumetric view. It was abstract, and
        answered three times over: ``self.space.xfmname`` on :class:`Volume`,
        ``self.dim1.xfmname`` on ``Volume2D`` and ``self.red.xfmname`` on
        ``VolumeRGB``. Those last two reach through a channel for a value the
        channel itself reads off the shared space.

        The narrowing to ``str`` is the whole reason this cannot just be inherited
        from :attr:`BrainSpace.xfmname`, which is ``Optional`` because a surface
        space has no transform.
        """
        xfmname = self.space.xfmname
        if xfmname is None:
            raise TypeError(
                "%s is volumetric but its space (%r) has no transform"
                % (type(self).__name__, self.space)
            )
        return xfmname

    @property
    def volume(self) -> npt.NDArray:
        """The data as a volume, with a leading time axis.

        The volumetric name for :attr:`RenderableView.spatial_data`, and concrete
        for every volumetric view: scalar and automatically unmasked for
        :class:`Volume`, uint8 RGBA for the 2D and RGB views, whose data has
        already been colormapped.
        """
        return self.spatial_data

    @property
    @abstractmethod
    def raw(self) -> VolumeRGB:
        """Narrowed from :attr:`Dataview.raw`: a volumetric view renders to
        :class:`~cortex.dataset.viewRGB.VolumeRGB`, never to the surface form."""


class SurfaceView(RenderableView):
    """A view whose data can be sampled per-vertex on a cortical surface.

    The other spatial interface. See :class:`VolumetricView` for why this is an
    abstract base rather than a ``Protocol``.

    ``blend_curvature`` lives here as a concrete method, so all three surface
    views inherit one implementation rather than each forwarding to a free
    function.
    """

    @property
    def vertices(self) -> npt.NDArray:
        """The data per vertex, with a leading time axis.

        The surface name for :attr:`RenderableView.spatial_data`, and concrete for
        every surface view: scalar for :class:`Vertex`, uint8 RGBA for the 2D and
        RGB views, whose data has already been colormapped.
        """
        return self.spatial_data

    @property
    @abstractmethod
    def raw(self) -> VertexRGB:
        """Narrowed from :attr:`Dataview.raw`: a surface view renders to
        :class:`~cortex.dataset.viewRGB.VertexRGB`, never to the volumetric form."""

    def blend_curvature(
        self,
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
        curvature = db.get_surfinfo(self.subject, smooth=smooth).data
        curvature = (curvature > threshold).astype("float")
        curvature = curvature * contrast + brightness
        curvature_raw = Vertex(
            curvature, self.subject, vmin=0, vmax=1, cmap="gray"
        ).raw
    
        # prepare alpha map
        clipped = np.clip(alpha.astype("float"), 0, 1)
    
        # blend original map with curvature map. VertexRGB.raw returns self, so copy.
        blended = deepcopy(self.raw)
        for channel, curv in (
            ("red", curvature_raw.red),
            ("green", curvature_raw.green),
            ("blue", curvature_raw.blue),
        ):
            chan = getattr(blended, channel)
            chan.data = (chan.data * clipped + (1 - clipped) * curv.data).astype("uint8")
    
        return blended



#: The scalar view type a composite view's channels have. Covariant, because the
#: channels are read-only properties, so ``DataviewRGB[Volume]`` is safely usable
#: where ``DataviewRGB[ScalarView]`` is expected; an invariant parameter would
#: reject that, which is the usual generics tax. Declared here rather than once
#: per composite column, which is where it was: two identical definitions
#: carrying the same comment.
ScalarT = TypeVar("ScalarT", bound="ScalarView", covariant=True)


def _resolve_channels(
    channels: Sequence[Union[npt.NDArray, "ScalarView"]],
    *,
    channel_cls: type[ScalarT],
    space_cls: type[BrainSpace],
    subject: Optional[str],
    spec: dict[str, Any],
    argnames: Sequence[str],
) -> tuple[BrainSpace, Optional[list[ScalarT]]]:
    """Validate a composite view's channel arguments and find their space.

    Returns the space to build views in, plus the channels as already-built views
    if that is what was passed and ``None`` if they were raw arrays -- the two
    composite columns wrap arrays with different arguments (per-dimension ranges
    against a colour basis), so wrapping is all that is left to them.

    The two forms cannot be mixed: either every channel is a ``channel_cls`` and
    they agree on subject and on whatever else identifies the space, or none is
    and ``subject`` plus the space's :attr:`~cortex.dataset._space.BrainSpace.spec_keys`
    must be given explicitly. ``Dataview2D`` and ``DataviewRGB`` each carried
    this, with the same four checks in the same order and different wording.

    ``argnames`` is only for messages, since one column calls its arguments
    ``dim1``/``dim2`` and the other ``channel1``..``channel3``.
    """
    kind = channel_cls.__name__
    first = channels[0]

    if isinstance(first, channel_cls):
        for name, chan in zip(argnames[1:], channels[1:]):
            if not isinstance(chan, channel_cls):
                raise TypeError(
                    "%s is not a %s object; if %s is one then all of them must be"
                    % (name, kind, argnames[0])
                )
            if chan.subject != first.subject:
                raise TypeError("%s is from a different subject" % name)
        if subject is not None and first.subject != subject:
            raise ValueError(
                "Subject in %s objects (%r) is different than specified subject "
                "(%r)" % (kind, first.subject, subject)
            )
        for key, value in spec.items():
            existing = getattr(first, key, None)
            if value is not None and existing != value:
                raise ValueError(
                    "%s in %s objects (%r) is different than specified %s (%r)"
                    % (key, kind, existing, key, value)
                )
        return first.space, [cast(ScalarT, c) for c in channels]

    for name, chan in zip(argnames[1:], channels[1:]):
        if isinstance(chan, channel_cls):
            raise TypeError(
                "%s is a %s object, so %s must be one as well"
                % (name, kind, argnames[0])
            )
    # Only reached with raw arrays, so the space has to be built from scratch and
    # every argument that identifies it is now mandatory. Which those are is the
    # space's own business; see BrainSpace.from_spec.
    return space_cls.from_spec(subject, **spec), None


def as_renderable(view: Dataview) -> RenderableView:
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

class ScalarView(Packable, RenderableView):
    """A single array of scalar values, displayed through a 1D colormap.

    This is the union of what used to be ``BrainData`` and the colormapped half
    of ``Dataview``. Those were separate classes joined only by multiple
    inheritance in ``Volume``/``Vertex``, which is what made ``super()`` calls in
    ``BrainData`` resolve to methods that were nowhere in its own ancestry.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        space: BrainSpace,
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

        self._space = space
        # coerce() validates the array against the geometry and fills in the
        # data-dependent parts of the space (which mask a flattened array
        # matches, which hemisphere a half-length array covered).
        self._data = space.coerce(data)
        #: Whether the data array carries a leading time axis.
        self.movie = space.is_movie(self._data)

        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(description=description, state=state, **kwargs)
        # Every scalar view leaves this constructor with numeric bounds. Doing it
        # here rather than in each subclass is what makes that an invariant of the
        # column instead of a step each new space has to remember: `.vmin` used to
        # be a number after `Volume(...)` and `None` after a bare `ScalarView`
        # subclass, so a consumer reading it had to know which class built it.
        self._resolve_percentiles()

    #: Narrowing a *bare annotation* in a subclass is accepted by mypy, which is
    #: what lets Volume and Vertex expose a precisely-typed `space` without an
    #: assert. Narrowing an *assigned* ClassVar is not -- that asymmetry is
    #: exactly what made the old `_cls` untypeable.
    _space: BrainSpace

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    @property
    def space(self) -> BrainSpace:
        return self._space

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one frame's worth of data in this view's space."""
        return self.space.spatial_shape

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

    @staticmethod
    def _sample(shape: tuple[int, ...], value: Optional[float]) -> npt.NDArray:
        """A fresh array of ``shape``: constant ``value``, or standard normal.

        The body shared by ``empty`` and ``random`` in every space. ``np.ones() *
        value`` rather than ``np.full``, which would give an integer array for the
        default ``value=0``.
        """
        if value is None:
            return np.random.randn(*shape)
        return np.ones(shape) * value

    def uniques(self, collapse: bool = False) -> Iterator["Packable"]:
        yield self

    def copy(self, data: npt.NDArray) -> Self:
        """A new view of the same kind, over ``data``, sharing this one's space.

        Cheap: the space carries whatever the geometry lookup produced, so this
        does not re-resolve a mask or reload surfaces from the database.

        Goes through :meth:`BrainSpace.wrap`, which is what makes one
        implementation serve every space -- the per-space constructor arguments
        are the space's business, not this method's.
        """
        return cast(
            Self,
            self.space.wrap(
                data,
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
                description=self.description,
                state=self.state,
                attrs=self.attrs,
            ),
        )

    def _build_raw(self) -> DataviewRGB:
        """Colormap this view and wrap the result as an RGB view of its space.

        Shared by :attr:`Volume.raw` and :attr:`Vertex.raw`, which differ only in
        which concrete RGB class they name -- something the space already knows.
        """
        (r, g, b, a), nan_mask = self._colormap_to_rgba()
        result: DataviewRGB = self.space.wrap_rgb(
            r,
            g,
            b,
            a,
            description=self.description,
            state=self.state,
            priority=self.priority,
        )
        result._nan_mask = nan_mask
        return result

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
        """One implementation for every space; the space supplies its own keys.

        ``Volume`` and ``Vertex`` used to override this to bolt on ``shape`` and
        ``split``/``frames`` respectively, which is the same question -- how does
        the browser unpack this array -- answered per space. It is now
        :meth:`BrainSpace.describe_layout`, so a new space contributes its keys
        without touching this method.
        """
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
            sdict.update(self.space.describe_layout(self.data))
            return sdict

        # No percentile fallback here: `__init__` resolved these, so a `None`
        # would mean something set them back to None after construction, and
        # silently substituting a percentile would hide that.
        sdict.update(
            DataviewJSON(cmap=[self.cmap], vmin=[self.vmin], vmax=[self.vmax])
        )
        sdict.update(DataviewJSON(data=[self.name]))
        sdict.update(self.space.to_json())
        return sdict

    def _write_cmap_slots(self, view: h5py.Dataset) -> None:
        view[2] = _dumps([self.cmap])
        view[3] = _dumps([self.vmin])
        view[4] = _dumps([self.vmax])

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
        self.space.write_hdf_attrs(h5, node)
        return node

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        self._write_data_hdf(h5)
        return self._write_view_node(
            h5, name=name, data=[self.name], xfmname=self.space.view_xfmname
        )

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
    def space(self) -> BrainSpace:
        raise NotImplementedError

    @property
    def raw(self) -> DataviewRGB:
        raise NotImplementedError

    def uniques(self, collapse: bool = False) -> Iterator["Packable"]:
        raise NotImplementedError

    def _write_hdf(
        self, h5: Union[h5py.File, h5py.Group], name: str = "data"
    ) -> h5py.Dataset:
        raise NotImplementedError


class Volume(ScalarView, VolumetricView):
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
            VolumeSpace(subject, xfmname, mask=mask),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )
        self.masked: _masker[Volume] = _masker(self)

    _space: VolumeSpace

    @property
    def space(self) -> VolumeSpace:
        return self._space

    # ------------------------------------------------------------------
    # geometry, delegated to the space
    # ------------------------------------------------------------------
    @property
    def linear(self) -> bool:
        """Whether the data is flattened into mask space rather than a full volume."""
        return self.space.linear

    @property
    def mask(self) -> Optional[npt.NDArray[np.bool_]]:
        """The boolean mask for flattened data, or None for a full volume.

        Previously this attribute simply did not exist on unmasked volumes, so
        every reader needed a ``linear`` guard or a ``hasattr``.
        """
        return self.space.mask

    @property
    def mask_name(self) -> Optional[str]:
        """The database mask name, when the mask came from or was found in the db."""
        return self.space.mask_name

    @property
    def _mask(self) -> MaskSpec:
        """Deprecated. Use :attr:`mask` for the array or :attr:`mask_name` for the name."""
        return self.space.mask_spec

    @property
    def spatial_data(self) -> npt.NDArray:
        """A 3D or 4D volume, automatically unmasking masked data.

        Also reachable as :attr:`~cortex.dataset.views.VolumetricView.volume`,
        which is the name this has always had in user code.
        """
        return self.space.unmask(self.data, self.movie)

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
        """A Volume filled with ``value``, shaped by ``xfmname``'s reference volume.

        ``subject`` and ``xfmname`` must both exist in the pycortex database.
        Other keyword arguments go to the constructor. Useful for tests, and as a
        starting point for arithmetic -- ``Volume.empty(s, x) + 1``.
        """
        shape = VolumeSpace(subject, xfmname).template_shape
        return cls(cls._sample(shape, value), subject, xfmname, **kwargs)

    @classmethod
    def random(cls, subject: str, xfmname: str, **kwargs: Any) -> Self:
        """A Volume of standard normal noise, shaped by ``xfmname``'s reference volume.

        Gaussian, mean 0, s.d. 1. ``subject`` and ``xfmname`` must both exist in
        the pycortex database; other keyword arguments go to the constructor.
        """
        shape = VolumeSpace(subject, xfmname).template_shape
        return cls(cls._sample(shape, None), subject, xfmname, **kwargs)

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
            name: Any = self.space.mask_spec
            if isinstance(self.space.mask_spec, np.ndarray):
                name = "custom"
            maskstr = "%s masked" % name
        if self.movie:
            maskstr += " movie"
        maskstr = maskstr[0].upper() + maskstr[1:]
        return "<%s data for (%s, %s)>" % (maskstr, self.subject, self.xfmname)

    @property
    def raw(self) -> VolumeRGB:
        return cast(VolumeRGB, self._build_raw())


class Vertex(ScalarView, SurfaceView):
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
            SurfaceSpace(subject),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            **kwargs,
        )

    _space: SurfaceSpace

    @property
    def space(self) -> SurfaceSpace:
        return self._space

    # ------------------------------------------------------------------
    # geometry, delegated to the space
    # ------------------------------------------------------------------
    @property
    def llen(self) -> int:
        """Number of vertices in the left hemisphere."""
        return self.space.llen

    @property
    def rlen(self) -> int:
        """Number of vertices in the right hemisphere."""
        return self.space.rlen

    @property
    def nverts(self) -> int:
        """Total number of vertices across both hemispheres."""
        return self.space.nverts

    @property
    def hem(self) -> str:
        """Which hemispheres the data covered: "left", "right" or "both".

        Single-hemisphere data is padded with zeros for the other hemisphere, so
        this records what was originally supplied.
        """
        return self.space.hem

    @property
    def spatial_data(self) -> npt.NDArray:
        """The per-vertex data with a leading frame axis, added if absent.

        Also reachable as :attr:`~cortex.dataset.views.SurfaceView.vertices`,
        which is the name this has always had in user code.
        """
        verts = self.data
        if not self.movie:
            verts = verts[np.newaxis]
        return verts

    @property
    def left(self) -> npt.NDArray:
        """Data for only the left hemisphere vertices."""
        return self.space.split_hemispheres(self.data)[0]

    @property
    def right(self) -> npt.NDArray:
        """Data for only the right hemisphere vertices."""
        return self.space.split_hemispheres(self.data)[1]

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
        """A Vertex filled with ``value``, one entry per vertex of both hemispheres.

        ``subject`` must exist in the pycortex database. Other keyword arguments
        go to the constructor. Useful for tests, and as a starting point for
        arithmetic -- ``Vertex.empty(s) + 1``.
        """
        shape = SurfaceSpace(subject).template_shape
        return cls(cls._sample(shape, value), subject, **kwargs)

    @classmethod
    def random(cls, subject: str, **kwargs: Any) -> Self:
        """A Vertex of standard normal noise, one entry per vertex of both hemispheres.

        Gaussian, mean 0, s.d. 1. ``subject`` must exist in the pycortex
        database; other keyword arguments go to the constructor.
        """
        shape = SurfaceSpace(subject).template_shape
        return cls(cls._sample(shape, None), subject, **kwargs)

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

    @property
    def raw(self) -> VertexRGB:
        return cast(VertexRGB, self._build_raw())


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
def _detect_space(
    attrs: dict[str, Any],
    *,
    subject: str,
    xfmname: Optional[str],
    mask: MaskSpec,
) -> BrainSpace:
    """Pick the space an HDF data node belongs to.

    Consults the registry in order and takes the first space that claims the
    node. Legacy files carry no space discriminator, so the built-in spaces key
    off whether a transform name is present, with ``SurfaceSpace`` last as the
    catch-all; a new space registers ahead of them and tests for something it
    writes itself.
    """
    for space_cls in registered_spaces():
        space = space_cls.from_hdf(
            attrs, subject=subject, xfmname=xfmname, mask=mask
        )
        if space is not None:
            return space
    raise ValueError(
        "No registered brain space claims this data node (subject=%r, xfmname=%r)"
        % (subject, xfmname)
    )


def _from_hdf_data(
    h5: h5py.File,
    name: str,
    xfmname: Optional[str] = None,
    subject: Optional[str] = None,
    **kwargs: Any,
) -> Dataview:
    """Decode a ``__hash``-named node from an HDF file into its view.

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
    mask: MaskSpec = None
    if "mask" in attrs:
        if attrs["mask"].startswith("__"):
            mask = h5[
                "/subjects/%s/transforms/%s/masks/%s"
                % (attrs["subject"], xfmname, attrs["mask"])
            ][()]
        else:
            mask = attrs["mask"]

    space = _detect_space(attrs, subject=subject, xfmname=xfmname, mask=mask)

    # support old style RGB volumes: uint8 with a trailing channel axis
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        alpha = None
        if dnode.shape[-1] == 4:
            alpha = space.wrap(dnode[..., 3])

        # The only surviving kwarg drop, and it is not a workaround for the
        # construction path: the *view* record said "scalar", so it carried a
        # colormap and bounds, and only opening the data node revealed packed
        # RGB. Those three arguments describe a colormap this view does not have,
        # so they are meaningless rather than merely unaccepted. Everything else
        # is forwarded.
        rgb_cls = type(space).views().rgb
        return cast(
            Dataview,
            rgb_cls(
                space.wrap(dnode[..., 0]),
                space.wrap(dnode[..., 1]),
                space.wrap(dnode[..., 2]),
                alpha=alpha,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("cmap", "vmin", "vmax")
                },
            ),
        )

    return cast(Dataview, space.wrap(dnode, **kwargs))


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

    # Surface views have no transform, so slot 7 of the view record is null and
    # `xfmname` arrives as None rather than as a per-channel list. Indexing it
    # unconditionally used to raise TypeError here, and Dataset.from_file
    # swallows exceptions -- so a saved Vertex2D was silently dropped on reload.
    xfmnames = xfmname if isinstance(xfmname, (list, tuple)) else [xfmname] * len(data)

    channels = [
        _from_hdf_data(h5, node, xfmname=xfmnames[i if i < len(xfmnames) else 0],
                       subject=subject)
        if node is not None
        else None
        for i, node in enumerate(data)
    ]
    first = channels[0]
    assert first is not None
    space = first.space

    if len(data) == 2:
        twod_cls = type(space).views().twod
        return cast(
            Dataview,
            twod_cls(
                channels[0],
                channels[1],
                vmin=vmin[0],
                vmin2=vmin[1],
                vmax=vmax[0],
                vmax2=vmax[1],
                subject=subject,
                **kwargs,
            ),
        )
    elif len(data) == 4:
        rgb_cls = type(space).views().rgb
        return cast(
            Dataview,
            rgb_cls(
                channels[0],
                channels[1],
                channels[2],
                alpha=channels[3],
                subject=subject,
                **kwargs,
            ),
        )
    else:
        raise ValueError("Invalid Dataview specification")


from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB  # noqa: E402
from .view2D import Dataview2D, Vertex2D, Volume2D  # noqa: E402
