import os
import json
from typing import Optional, Union
import warnings

import numpy as np
import numpy.typing as npt

from .. import options
from .views import Dataview, Volume, Vertex, VolumeRGB, VertexRGB
from .viewRGB import _warn_alpha_range
from .braindata import BrainData, VolumeData, VertexData

default_cmap2D = options.config.get("basic", "default_cmap2D")

class Dataview2D(Dataview):
    """Abstract base class for 2-dimensional data views.
    """
    dim1: Dataview
    dim2: Dataview

    def __init__(self, description: str="", cmap: Optional[str]=None,
                 vmin: Optional[float]=None, vmax: Optional[float]=None,
                 vmin2: Optional[float]=None, vmax2: Optional[float]=None, state=None,
                 alpha=None, **kwargs):
        self.cmap = cmap or default_cmap2D
        self.vmin = vmin
        self.vmax = vmax
        self.vmin2 = vmin if vmin2 is None else vmin2
        self.vmax2 = vmax if vmax2 is None else vmax2

        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1
        self.description = description
        # Optional per-voxel/vertex alpha map. Kept as a Volume/Vertex (never
        # inside ``attrs``, which is JSON-serialized for the WebGL viewer) so
        # that both renderers can ship it like any other brain.
        self.alpha = alpha

    @property
    def alpha(self):
        """Optional alpha map (Volume/Vertex in [vmin, vmax]) multiplied into
        the colormap alpha. NaN anywhere (dim1, dim2 or alpha) renders as
        alpha 0 regardless of this map."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha is not None and not isinstance(alpha, self._cls):
            alpha = np.asarray(alpha)
            _warn_alpha_range(alpha)
            alpha = self._wrap_alpha(alpha)
        if alpha is not None and alpha.subject != self.dim1.subject:
            raise ValueError("alpha must belong to the same subject as dim1")
        if alpha is not None and getattr(alpha, "xfmname", None) != getattr(self.dim1, "xfmname", None):
            raise ValueError("alpha must use the same transform as dim1")
        self._alpha = alpha
        self._alpha_brain_cache = None

    @property
    def _alpha_brain(self):
        """The alpha map normalized to [0, 1] (vmin=0, vmax=1), as shipped to
        the WebGL viewer and stored in HDF files. NaN is preserved."""
        if self.alpha is None:
            return None
        if self._alpha_brain_cache is None:
            self._alpha_brain_cache = self._wrap_alpha(self._normalized_alpha())
        return self._alpha_brain_cache

    def _wrap_alpha(self, alpha):
        raise NotImplementedError

    def _normalized_alpha(self, full_volume=False):
        """User alpha as float array in [0, 1] (NaN preserved), in the space of
        ``.data`` or, for volumes with ``full_volume=True``, of ``.volume``."""
        alpha = self.alpha
        if alpha is None:
            return None
        raw = alpha.volume if full_volume else alpha.data
        arr = np.asarray(raw, dtype=float)
        if np.asarray(raw).dtype == np.uint8:
            return arr / 255.
        vmin = 0. if alpha.vmin is None else float(alpha.vmin)
        vmax = 1. if alpha.vmax is None else float(alpha.vmax)
        if vmax == vmin:
            return np.where(arr >= vmax, 1., 0.)
        return (arr - vmin) / (vmax - vmin)

    def uniques(self, collapse=False):
        yield self.dim1
        yield self.dim2
        if self.alpha is not None:
            yield self._alpha_brain

    def _write_hdf(self, h5, name="data"):
        self._cls._write_hdf(self.dim1, h5)
        self._cls._write_hdf(self.dim2, h5)
        names = [self.dim1.name, self.dim2.name]
        if self.alpha is not None:
            # Stored normalized to [0, 1] so that it can be restored with
            # vmin=0, vmax=1 (BrainData nodes carry no range).
            self._cls._write_hdf(self._alpha_brain, h5)
            names.append(self._alpha_brain.name)

        viewnode = Dataview._write_hdf(self, h5, name=name)
        viewnode[0] = json.dumps([names])
        viewnode[3] = json.dumps([[self.vmin, self.vmin2]])
        viewnode[4] = json.dumps([[self.vmax, self.vmax2]])
        return viewnode

    def to_json(self, simple=False):
        sdict = dict(data=[[self.dim1.name, self.dim2.name]],
            state=self.state, 
            attrs=self.attrs, 
            desc=self.description,
            cmap=[self.cmap] )

        d1js = self.dim1.to_json()
        d2js = self.dim2.to_json()
        # ``is None`` checks: a legitimate vmin/vmax of 0 must not fall back
        # to the auto range (same class of bug as 5482c8bf for Volume).
        sdict.update(dict(
            vmin = [[d1js['vmin'][0] if self.vmin is None else self.vmin,
                     d2js['vmin'][0] if self.vmin2 is None else self.vmin2]],
            vmax = [[d1js['vmax'][0] if self.vmax is None else self.vmax,
                     d2js['vmax'][0] if self.vmax2 is None else self.vmax2]],
            ))

        if "xfm" in d1js:
            sdict['xfm'] = [[d1js['xfm'][0], d2js['xfm'][0]]]

        if self.alpha is not None:
            sdict['alpha'] = [self._alpha_brain.name]

        return sdict

    def _to_raw(self, data1, data2, alpha=None):
        """Colormap (data1, data2) through the 2D colormap.

        Returns ``(r, g, b, a, nan_mask)`` as uint8 channels. ``a`` is the
        colormap alpha multiplied by the (normalized) user ``alpha`` map when
        given; it is forced to 0 wherever data1, data2 or alpha is NaN.
        """
        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize
        cmapdir = options.config.get("webgl", "colormaps")
        cmap = plt.imread(os.path.join(cmapdir, "%s.png"%self.cmap))
        _warn_non_perceptually_uniform_colormap(self.cmap)

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)
        
        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1]-1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32) 
        dim2 = np.round(d2 * (cmap.shape[0]-1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # map r, g, b, a values between 0 and 255 to avoid problems with
        # VolumeRGB when plotting flatmaps with quickflat
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = colored.T
        r.shape = dim1.shape
        g.shape = dim1.shape
        b.shape = dim1.shape
        a.shape = dim1.shape
        # NaN in either dimension (or in the alpha map) -> alpha = 0
        aidx = np.logical_or(np.isnan(data1), np.isnan(data2))
        if alpha is not None:
            alpha = np.asarray(alpha, dtype=float)
            aidx = np.logical_or(aidx, np.isnan(alpha))
            user = np.clip(np.nan_to_num(alpha, nan=0.0), 0, 1)
            a = np.round(a.astype(float) * user).astype(np.uint8)
        aidx = np.broadcast_to(aidx, a.shape)
        a[aidx] = 0
        return r, g, b, a, aidx

    @property
    def subject(self):
        return self.dim1.subject

class Volume2D(Dataview2D):
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
        dim1 must be a Volume from which the subject can be extracted.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        dim1 must be a Volume from which the subject can be extracted.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the 
        `default_cmap2d` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. If not given, the ``vmin`` of dim1
        is used (its own ``vmin`` for a Volume/Vertex object, the 1st percentile
        of the data for an array).
    vmax : float, optional
        Maximum value in colormap for dim1. If not given, the ``vmax`` of dim1
        is used (its own ``vmax`` for a Volume/Vertex object, the 99th
        percentile of the data for an array).
    vmin2 : float, optional
        Minimum value in colormap for dim2. If not given, the ``vmin`` of dim2
        is used (same rule as ``vmin``).
    vmax2 : float, optional
        Maximum value in colormap for dim2. If not given, the ``vmax`` of dim2
        is used (same rule as ``vmax``).
    alpha : ndarray or Volume/Vertex, optional
        Per-voxel (per-vertex) opacity multiplied into the colormap alpha.
        Arrays are taken in [0, 1]; Volume/Vertex objects are normalized by
        their own ``vmin``/``vmax``. Honored identically by quickflat and the
        WebGL viewer. Wherever dim1, dim2 or alpha is NaN the data is
        rendered fully transparent, regardless of this map.
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and Dataview

    """
    _cls = VolumeData
    dim1: Volume
    dim2: Volume

    def __init__(self, dim1: Union[npt.NDArray, Volume], dim2: Union[npt.NDArray, Volume], subject: Optional[str]=None, xfmname: Optional[str]=None, description: str="", cmap: Optional[str]=None,
                 vmin: Optional[float]=None, vmax: Optional[float]=None, vmin2: Optional[float]=None, vmax2: Optional[float]=None, **kwargs):
        if isinstance(dim1, self._cls):
            if subject is not None or xfmname is not None:
                raise TypeError("Subject and xfmname cannot be specified with Volumes")
            if not isinstance(dim2, self._cls) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
        else:
            if isinstance(dim2, self._cls):
                raise TypeError("If dim2 is a Volume, dim1 must be a Volume as well")
            if subject is None or xfmname is None:
                raise TypeError("Subject and xfmname must be specified with raw data")
            self.dim1 = Volume(dim1, subject, xfmname, vmin=vmin, vmax=vmax)
            self.dim2 = Volume(dim2, subject, xfmname, vmin=vmin2, vmax=vmax2)

        vmin = self.dim1.vmin if vmin is None else vmin
        vmin2 = self.dim2.vmin if vmin2 is None else vmin2
        vmax = self.dim1.vmax if vmax is None else vmax
        vmax2 = self.dim2.vmax if vmax2 is None else vmax2

        super().__init__(description=description, cmap=cmap, vmin=vmin,
                                       vmax=vmax, vmin2=vmin2, vmax2=vmax2, **kwargs)

    def __repr__(self):
        return "<2D volumetric data for (%s, %s)>"%(self.dim1.subject, self.dim1.xfmname)

    def _write_hdf(self, h5, name="data"):
        viewnode = super()._write_hdf(h5, name)
        viewnode[7] = json.dumps([[self.dim1.xfmname, self.dim2.xfmname]])
        return viewnode

    @property
    def raw(self):
        """VolumeRGB object containing the colormapped data from this object.
        """
        if self.dim1.xfmname != self.dim2.xfmname:
            raise ValueError("Both Volumes must have same xfmname to generate single raw volume")

        def _same_mask(a, b):
            return (a.linear and b.linear and a.mask.shape == b.mask.shape
                    and np.all(a.mask == b.mask))

        linear = _same_mask(self.dim1, self.dim2)
        if linear and self.alpha is not None:
            linear = _same_mask(self.dim1, self.alpha)
        if linear:
            r, g, b, a, nan_mask = self._to_raw(
                self.dim1.data, self.dim2.data, self._normalized_alpha())
        else:
            r, g, b, a, nan_mask = self._to_raw(
                self.dim1.volume, self.dim2.volume,
                self._normalized_alpha(full_volume=True))
        result = VolumeRGB(r, g, b, subject=self.dim1.subject,
                           xfmname=self.dim1.xfmname, alpha=a,
                           state=self.state, description=self.description,
                           priority=self.priority)
        result._nan_mask = nan_mask
        return result

    def _wrap_alpha(self, alpha):
        return Volume(alpha, self.dim1.subject, self.dim1.xfmname, vmin=0, vmax=1)


    @property
    def xfmname(self):
        return self.dim1.xfmname

class Vertex2D(Dataview2D):
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
        dim1 must be a Vertex from which the subject can be extracted.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    cmap : str, optional
        Colormap (or colormap name) to use. If not given defaults to the 
        `default_cmap2d` in your pycortex options.cfg file.
    vmin : float, optional
        Minimum value in colormap for dim1. If not given, the ``vmin`` of dim1
        is used (its own ``vmin`` for a Volume/Vertex object, the 1st percentile
        of the data for an array).
    vmax : float, optional
        Maximum value in colormap for dim1. If not given, the ``vmax`` of dim1
        is used (its own ``vmax`` for a Volume/Vertex object, the 99th
        percentile of the data for an array).
    vmin2 : float, optional
        Minimum value in colormap for dim2. If not given, the ``vmin`` of dim2
        is used (same rule as ``vmin``).
    vmax2 : float, optional
        Maximum value in colormap for dim2. If not given, the ``vmax`` of dim2
        is used (same rule as ``vmax``).
    alpha : ndarray or Volume/Vertex, optional
        Per-voxel (per-vertex) opacity multiplied into the colormap alpha.
        Arrays are taken in [0, 1]; Volume/Vertex objects are normalized by
        their own ``vmin``/``vmax``. Honored identically by quickflat and the
        WebGL viewer. Wherever dim1, dim2 or alpha is NaN the data is
        rendered fully transparent, regardless of this map.
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and Dataview

    """
    _cls = VertexData
    blend_curvature = _cls.blend_curvature  # hacky inheritance
    dim1: Vertex
    dim2: Vertex

    def __init__(self, dim1: Union[npt.NDArray, Vertex], dim2: Union[npt.NDArray, Vertex], subject: Optional[str]=None, description: str="", cmap: Optional[str]=None,
                 vmin: Optional[float]=None, vmax: Optional[float]=None, vmin2: Optional[float]=None, vmax2: Optional[float]=None, **kwargs):
        if isinstance(dim1, VertexData):
            if subject is not None:
                raise TypeError("Subject cannot be specified with Vertex")
            if not isinstance(dim2, VertexData) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
        else:
            if isinstance(dim2, self._cls):
                raise TypeError("If dim2 is a Vertex, dim1 must be a Vertex as well")
            if subject is None:
                raise TypeError("Subject must be specified with raw data")
            self.dim1 = Vertex(dim1, subject, vmin=vmin, vmax=vmax)
            self.dim2 = Vertex(dim2, subject, vmin=vmin2, vmax=vmax2)

        vmin = self.dim1.vmin if vmin is None else vmin
        vmin2 = self.dim2.vmin if vmin2 is None else vmin2
        vmax = self.dim1.vmax if vmax is None else vmax
        vmax2 = self.dim2.vmax if vmax2 is None else vmax2

        super().__init__(description=description, cmap=cmap,
                                       vmin=vmin, vmax=vmax, vmin2=vmin2,
                                       vmax2=vmax2, **kwargs)

    def __repr__(self):
        return "<2D vertex data for (%s)>"%self.dim1.subject

    @property
    def raw(self):
        """VertexRGB object containing the colormapped data from this object.
        """
        r, g, b, a, nan_mask = self._to_raw(
            self.dim1.data, self.dim2.data, self._normalized_alpha())
        result = VertexRGB(r, g, b, subject=self.dim1.subject, alpha=a,
                           state=self.state, description=self.description,
                           priority=self.priority)
        result._nan_mask = nan_mask
        return result

    def _wrap_alpha(self, alpha):
        return Vertex(alpha, self.dim1.subject, vmin=0, vmax=1)

    @property
    def vertices(self):
        return self.raw.vertices


def _warn_non_perceptually_uniform_colormap(cmap):
    mapping = {
        "BuOr_2D": "PU_BuOr_covar",
        "RdBu_covar": "PU_RdBu_covar",
        "RdBu_covar2": "PU_BuOr_covar",
        "RdBu_covar_alpha": "PU_RdBu_covar_alpha",
        "RdGn_covar": "PU_RdGn_covar",
        "hot_alpha": "fire_alpha",
    }
    if cmap in mapping:
        warnings.warn("Colormap %r is not perceptually uniform. Consider using"
                      " %r instead." % (cmap, mapping[cmap]), UserWarning)
