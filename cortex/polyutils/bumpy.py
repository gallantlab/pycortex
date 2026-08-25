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
version for comparison, and `legacy_js_height` the one the webgl viewer used to
compute in javascript -- which is this module's quantity by another route.
"""


import numpy as np
from scipy import sparse

try:
    from scipy.sparse.linalg import factorized as _factorized
except ImportError:
    from scipy.sparse.linalg.dsolve import factorized as _factorized

from .misc import _memo
from .surface import Surface

__all__ = ["FlatSlab", "face_prism_volumes", "folding_height",
           "legacy_js_height", "naive_prism_height"]


def _flat_plane(flat):
    """Flatmap coordinates as a 3D point set lying in the z = 0 plane.

    Only the first two columns of a pycortex flat surface are the flatmap
    coordinates; the third is left over from the flattening and is ignored
    everywhere else too (see `cortex.brainctm`, which stores `pts[:, :2]`).
    """
    plane = np.zeros((len(flat), 3))
    plane[:, :2] = np.asarray(flat)[:, :2]
    return plane


def _face_areas(pts, polys):
    """Area of each triangle."""
    ppts = pts[polys]
    cross = np.cross(ppts[:, 1] - ppts[:, 0], ppts[:, 2] - ppts[:, 0])
    return 0.5 * np.sqrt((cross ** 2).sum(-1))


def _vertex_areas(pts, polys):
    """One third of the incident face area, summed at each vertex."""
    areas = _face_areas(pts, polys)
    return np.bincount(np.asarray(polys).ravel(), weights=np.repeat(areas, 3),
                       minlength=len(pts)) / 3.0


def _lumped(values, polys, nverts):
    """A per-face quantity gathered onto vertices, a third to each corner."""
    return np.bincount(np.asarray(polys).ravel(),
                       weights=np.repeat(values, 3), minlength=nverts) / 3.0


def _smooth_vectors(surf, vectors, factor):
    """`Surface.smooth`, applied to every column of `vectors` at once.

    The operator depends only on the mesh, so the obvious loop pays for three
    factorisations of the same matrix. One factorisation, three solves.
    """
    vectors = np.asarray(vectors, dtype=np.double)
    if not factor:
        return vectors.copy()

    _, D, W, V = surf.laplace_operator
    npt = len(D)
    lfac = sparse.dia_matrix((D, [0]), (npt, npt)) - factor * (W - V)
    # Vertices in no triangle have an empty row and column; `Surface.smooth`
    # drops them from the solve and leaves them at zero, so do the same.
    good = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]
    solve = _factorized(lfac[good][:, good].tocsc())

    out = np.zeros(vectors.shape)
    for k in range(vectors.shape[1]):
        out[good, k] = solve((D * vectors[:, k])[good])
    return out


def _regularise_log_height(flat, polys, numerator, denominator,
                           correlation_length):
    """Smooth a ratio of two positive fields, in the log, on the flat mesh.

    Solves ``(M + lc^2 L) l = M l*`` once, with `L` the cotangent stiffness and
    `M` the lumped mass. Working in the log keeps a twenty-fold compression as
    an offset of three rather than a twenty-fold spike, and makes this a
    geometric rather than an arithmetic mean -- the right averaging for a ratio.
    """
    nverts = len(flat)
    good = (numerator > 0) & (denominator > 0)
    target = np.zeros(nverts)
    target[good] = np.log(numerator[good] / denominator[good])
    if good.any():
        target[~good] = np.median(target[good])

    _, mass, weights, degree = Surface(flat, polys).laplace_operator
    lhs = (sparse.dia_matrix((mass, [0]), (nverts, nverts))
           + correlation_length ** 2 * (degree - weights)).tocsc()

    # Vertices with no area anchor nothing and leave a singular row.
    goodrows = np.nonzero(mass > 0)[0]
    solve = _factorized(lhs[goodrows][:, goodrows].tocsc())
    logheight = target.copy()
    logheight[goodrows] = solve((mass * target)[goodrows])
    return np.exp(logheight)


def face_prism_volumes(wm, pia, polys):
    """Volume of the cortical slab over each triangle.

    Each triangular prism between the two surfaces is cut into three
    tetrahedra, the same decomposition `polyutils.misc.brick_vol` uses. Nodes
    0-2 are the white matter triangle and 3-5 the pial one.

    Parameters
    ----------
    wm, pia : 2D ndarray, shape (total_verts, 3)
        The two surfaces bounding the slab.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.

    Returns
    -------
    volumes : 1D ndarray, shape (total_polys,)
    """
    polys = np.asarray(polys)
    corners = np.concatenate([np.asarray(wm)[polys], np.asarray(pia)[polys]],
                             axis=1)                       # (nfaces, 6, 3)
    total = np.zeros(len(polys))
    for a, b, c, d in ((0, 1, 2, 4), (0, 2, 3, 4), (2, 3, 4, 5)):
        e = np.stack([corners[:, b] - corners[:, a],
                      corners[:, c] - corners[:, a],
                      corners[:, d] - corners[:, a]], axis=1)
        total += np.abs(np.linalg.det(e)) / 6.0
    return total


def folding_height(flat, wm, pia, polys, correlation_length=0.5):
    """Slab height driven by the pial flare, not by the flattening.

    ``V_frustum / A_wm``, which with ``r = sqrt(A_pia / A_wm)`` is
    ``thickness * (1 + r + r**2) / 3``. Over a gyral crown the pia has more area
    than the white matter beneath it, so `r` rises; in a fundus it falls. See
    the module docstring for why the denominator is the folded area.
    """
    nverts = len(wm)
    awm = _lumped(_face_areas(wm, polys), polys, nverts)
    apia = _lumped(_face_areas(pia, polys), polys, nverts)
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
    return _regularise_log_height(flat, polys, height, np.ones(nverts),
                                  correlation_length)


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
    vol = face_prism_volumes(wm, pia, polys)
    area = _face_areas(_flat_plane(flat), polys)

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


def _umbrella_smooth(data, polys, nverts, factor, iterations):
    """The uniform smoothing the viewer's javascript used.

    Each vertex accumulates its neighbours once per incident face, so shared
    neighbours are weighted twice; there is no area or cotangent weighting.
    Reproduced only so that `legacy_js_height` matches what shipped.
    """
    polys = np.asarray(polys)
    a, b, c = polys[:, 0], polys[:, 1], polys[:, 2]
    idx = np.concatenate([a, a, b, b, c, c])
    counts = np.bincount(idx, minlength=nverts).astype(float)
    counts[counts == 0] = 1.0

    out = np.asarray(data, dtype=float).copy()
    for _ in range(iterations):
        nb = np.concatenate([out[b], out[c], out[a], out[c], out[a], out[b]])
        means = np.bincount(idx, weights=nb, minlength=nverts) / counts
        out = means * factor + out * (1.0 - factor)
    return out


def legacy_js_height(wm, pia, polys, smooth_areas=5, smooth_dists=20,
                     factor=0.1):
    """The bumpy flatmap height the webgl viewer used to compute in javascript.

    Despite its name this never looked at the flatmap: the denominator was the
    *folded* white matter vertex area, so with ``A_pia = r**2 * A_wm`` it
    collapses to ``thickness * (1 + r + r**2) / 3`` -- `folding_height` by
    another route, with weaker smoothing. Kept as a reference. The three
    smoothing arguments default to what the viewer shipped.

    Parameters
    ----------
    wm, pia : 2D ndarray, shape (total_verts, 3)
        The white matter and pial surfaces.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.
    smooth_areas, smooth_dists : int, optional
        Smoothing iterations for the vertex areas and for the thickness.
    factor : float, optional
        Smoothing step size.

    Returns
    -------
    height : 1D ndarray, shape (total_verts,)
    """
    wmareas = _umbrella_smooth(_vertex_areas(wm, polys), polys, len(wm),
                               factor, smooth_areas)
    piaareas = _umbrella_smooth(_vertex_areas(pia, polys), polys, len(wm),
                                factor, smooth_areas)
    dists = _umbrella_smooth(np.sqrt(((pia - wm) ** 2).sum(1)), polys, len(wm),
                             factor, smooth_dists)

    # Volume of a conical frustum with the two vertex areas as its parallel
    # caps and the thickness as its height.
    vol = dists / 3.0 * (wmareas + piaareas + np.sqrt(wmareas * piaareas))
    height = np.zeros(len(wm))
    good = wmareas > 0
    height[good] = vol[good] / wmareas[good]
    return height


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

        relief = folding_height(flat, wm, pia, polys,
                                self.correlation_length)[:, None]
        if self.polish:
            relief = _smooth_vectors(surf, relief, self.polish)
        if self.detrend:
            swell = _smooth_vectors(surf, relief,
                                    (self.detrend / (2 * np.pi)) ** 2)
            # about its own mean, so the sheet keeps its average thickness
            relief = relief - (swell - swell.mean(0))

        self.info = dict(n_verts=int(mask.sum()),
                         height_mean=float(relief.mean()),
                         height_std=float(relief.std()),
                         height_min=float(relief.min()),
                         height_max=float(relief.max()))

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask, 2] = relief[:, 0]
        return offsets
