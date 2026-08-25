"""Bumpy flatmaps: giving a flatmap the relief of the cortical slab.

Cortex is 2 to 5 mm thick, so if you peeled the slab off the white matter and
laid it down the white side would end up flat and the pial side would sit some
distance above it. That relief is a folding cue which survives even when the
flatmap is covered in data.

The height is ``V_frustum / A_wm``, the folded volume of each column over the
*folded* white matter area beneath it, which with ``r = sqrt(A_pia / A_wm)`` is
``thickness * (1 + r + r**2) / 3``. `r` is what carries the folding: the pia has
more area than the white matter under a gyral crown and less in a fundus.

Dividing by the *flattened* area instead is the more obvious model of a slab
laid flat, and it does not work. A flatmap's area distortion measures
essentially uncorrelated with both mean and Gaussian curvature, so as a
denominator it contributes no folding and injects the flattening algorithm's
artifacts in its place; what is left is close to a map of cortical thickness,
which is blobby and reads as round knobs. `naive_prism_height` computes that
version, for comparison.
"""


import numpy as np
from scipy import sparse

try:
    from scipy.sparse.linalg import factorized as _factorized
except ImportError:
    from scipy.sparse.linalg.dsolve import factorized as _factorized

from .misc import _memo, face_area, face_volume
from .surface import Surface

__all__ = ["FlatSlab", "folding_height", "naive_prism_height"]


def _flat_plane(flat):
    """Flatmap coordinates as a 3D point set lying in the z = 0 plane.

    Only the first two columns of a pycortex flat surface are the flatmap
    coordinates; the third is left over from the flattening and is ignored
    everywhere else too (see `cortex.brainctm`, which stores `pts[:, :2]`).
    """
    plane = np.zeros((len(flat), 3))
    plane[:, :2] = np.asarray(flat)[:, :2]
    return plane


def _lumped(values, polys, nverts):
    """A per-face quantity gathered onto vertices, a third to each corner."""
    return np.bincount(np.asarray(polys).ravel(),
                       weights=np.repeat(values, 3), minlength=nverts) / 3.0


def _regularise_log_height(flat, polys, values, correlation_length):
    """Smooth a positive field in the log, on the flat mesh.

    `Surface.smooth` solves ``(M + t L) y = M x`` with `L` the cotangent
    stiffness and `M` the lumped mass, which is the screened-Poisson system
    wanted here with ``t = lc**2``. Working in the log keeps a twenty-fold
    compression as an offset of three rather than a twenty-fold spike, and
    makes this a geometric rather than an arithmetic mean -- the right
    averaging for a ratio.
    """
    good = values > 0
    target = np.zeros(len(flat))
    target[good] = np.log(values[good])
    if good.any():
        target[~good] = np.median(target[good])

    surf = Surface(flat, polys)
    smoothed = surf.smooth(target, correlation_length ** 2)
    # `Surface.smooth` returns zero for vertices in no triangle, which would
    # come back as a height of one; leave those at the median instead.
    isolated = np.asarray(surf.connected.sum(1)).ravel() == 0
    smoothed[isolated] = target[isolated]
    return np.exp(smoothed)


def folding_height(flat, wm, pia, polys, correlation_length=0.5):
    """Slab height driven by the pial flare, not by the flattening.

    ``V_frustum / A_wm``, which with ``r = sqrt(A_pia / A_wm)`` is
    ``thickness * (1 + r + r**2) / 3``. Over a gyral crown the pia has more area
    than the white matter beneath it, so `r` rises; in a fundus it falls. See
    the module docstring for why the denominator is the folded area.
    """
    nverts = len(wm)
    polys = np.asarray(polys)
    awm = _lumped(face_area(wm[polys]), polys, nverts)
    apia = _lumped(face_area(pia[polys]), polys, nverts)
    thickness = np.sqrt(((pia - wm) ** 2).sum(1))

    good = awm > 0
    r = np.zeros(nverts)
    r[good] = np.sqrt(np.maximum(apia[good], 0.0) / awm[good])
    height = thickness * (1.0 + r + r ** 2) / 3.0

    # Same log-space regularisation as the prism height -- this is a ratio too,
    # and a vertex whose white matter area nearly vanishes would otherwise
    # spike. Smoothed on the flat mesh, not the folded one, because that is
    # where the relief is going to be looked at and where `correlation_length`
    # is measured for every other field here.
    return _regularise_log_height(flat, polys, height, correlation_length)


def naive_prism_height(flat, wm, pia, polys):
    """Height of a volume-preserving vertical prism over the flatmap.

    The obvious reading of the bumpy flatmap idea: each column keeps its folded
    volume over a base of the flattened triangle. Kept for comparison -- it
    spikes wherever flattening compressed a triangle hard, and smoothing a field
    of ratios afterwards does not remove spikes that dominate the mean. See the
    module docstring for the deeper problem with the denominator.

    Parameters
    ----------
    flat : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in flatmap space.
    wm : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the white matter surface.
    pia : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the pial surface.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices, shared by all three surfaces.

    Returns
    -------
    height : 1D ndarray, shape (total_verts,)
        Height of the pial surface above the flatmap at each vertex, in the
        units of the input surfaces.
    """
    nverts = len(wm)
    polys = np.asarray(polys)
    vol = face_volume(wm, pia, polys)
    area = face_area(_flat_plane(flat)[polys])

    # One height per triangle, then averaged onto the vertices. Note this is a
    # mean of ratios, and that is the whole problem: a triangle the flattening
    # crushed has a tiny area in the denominator, so it contributes an enormous
    # height that dominates every vertex it touches.
    faceheight = np.zeros(len(polys))
    good = area > 0
    faceheight[good] = vol[good] / area[good]

    counts = np.bincount(polys.ravel(), minlength=nverts)
    height = np.bincount(polys.ravel(), weights=np.repeat(faceheight, 3),
                         minlength=nverts)
    return np.where(counts > 0, height / np.maximum(counts, 1), 0.0)


class FlatSlab(object):
    """The cortical slab's relief, as an offset from the flat white surface.

    `folding_height`, smoothed and with the whole-map swell taken out.

    Parameters
    ----------
    flat : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in flatmap space. Only the first two columns
        are used; the flatmap is taken to lie in the z = 0 plane.
    wm, pia : 2D ndarray, shape (total_verts, 3)
        The white matter and pial surfaces.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices of the *flat* surface. Vertices in no triangle
        -- the medial wall, which is cut away from the flatmap -- get a zero
        offset.
    correlation_length : float, optional
        Length scale over which the height ratio is regularised -- enough to
        stop a vertex whose white matter area nearly vanishes from spiking.
    polish : float, optional
        Smoothing of the finished relief, as a diffusion time. Beware the
        scale: `t` halves a wavelength of ``2 * pi * sqrt(t)``, so the default
        cuts at 11 mm and not, as it looks, at 3. Much more takes the gyri too.
    detrend : float, optional
        Wavelength above which the relief is flattened. Cortex is regionally
        thicker in some lobes than others, and on S1 that swell is a third of
        the relief's variance in one map-spanning mode -- real, but not folding,
        and the viewer's scale setting multiplies it along with everything else.
        Pass 0 to keep it.

    Attributes
    ----------
    info : dict
        Height statistics from the last computed relief.
    """
    def __init__(self, flat, wm, pia, polys, correlation_length=0.5,
                 polish=3.0, detrend=64.0):
        self.flat = np.asarray(flat, dtype=np.double)
        self.wm = np.asarray(wm, dtype=np.double)
        self.pia = np.asarray(pia, dtype=np.double)
        self.polys = np.asarray(polys)
        self.correlation_length = correlation_length
        self.polish = polish
        self.detrend = detrend
        self.info = {}
        self._cache = {}

        if not (len(self.flat) == len(self.wm) == len(self.pia)):
            raise ValueError("flat, wm and pia must have the same number of "
                             "vertices, got %d, %d and %d"
                             % (len(self.flat), len(self.wm), len(self.pia)))

    @property
    @_memo
    def _submesh(self):
        """Restrict to the vertices that are actually on the flatmap.

        Returns ``(mask, subflat, subwm, subpia, subpolys)``, with `subflat` in
        the z = 0 plane and the triangles reindexed to match.
        """
        mask = np.zeros(len(self.wm), dtype=bool)
        mask[self.polys.ravel()] = True

        vmap = np.zeros(len(self.wm), dtype=np.int64)
        vmap[mask] = np.arange(mask.sum())

        return (mask, _flat_plane(self.flat)[mask], self.wm[mask],
                self.pia[mask], vmap[self.polys])

    @property
    @_memo
    def relaxed(self):
        """The pial surface's height above the flat white matter surface.

        Returns
        -------
        offsets : 2D ndarray, shape (total_verts, 3)
            The first two columns are zero -- the relief is purely vertical --
            and the third is the height. Vertices off the flatmap are zero.
        """
        mask, flat, wm, pia, polys = self._submesh
        surf = Surface(flat, polys)

        relief = folding_height(flat, wm, pia, polys, self.correlation_length)
        if self.polish:
            relief = surf.smooth(relief, self.polish)
        if self.detrend:
            swell = surf.smooth(relief, (self.detrend / (2 * np.pi)) ** 2)
            # about its own mean, so the sheet keeps its average thickness
            relief = relief - (swell - swell.mean())

        self.info = dict(n_verts=int(mask.sum()),
                         height_mean=float(relief.mean()),
                         height_std=float(relief.std()),
                         height_min=float(relief.min()),
                         height_max=float(relief.max()))

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask, 2] = relief
        return offsets
