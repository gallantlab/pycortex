"""Bumpy flatmaps: relaxing the cortical slab onto a flatmap.

A flatmap is made by cutting the white matter surface and flattening it. Cortex
is not infinitesimally thin, though, so if you imagine peeling the cortical slab
off the white matter and laying it down, the white matter side would end up flat
while the pial side would sit some distance above it -- bunched up thicker over
gyri, which flattening compresses, and stretched thinner over sulci, which
flattening expands. That relief is a curvature cue that survives even when the
flatmap is completely covered with data.

The naive way to compute it is to insist on preserving volume in a vertical
column: the height over a triangle is the folded prism volume divided by the
flattened triangle area. That denominator goes to zero wherever the flattening
compressed a triangle hard, so the naive height spikes, and smoothing a field of
ratios afterwards is dominated by exactly those outliers.

What actually stops a real slab of tissue from spiking is not gravity -- for gray
matter the ratio of gravitational to elastic stress over a 2.5mm slab is about
``rho*g*t/mu = 1040*9.81*0.0025/1000 ~ 0.026``, a couple of percent -- but shear.
A tall narrow column is expensive because it shears against its neighbours, and
it relieves that by flowing sideways, which is only possible if the pial surface
is free to move in-plane rather than being pinned directly above the white
matter. So :class:`FlatSlab` gives every pial vertex three degrees of freedom and
minimises an elastic energy.

Note what that leaves as a parameter. With no body force, a uniform material, a
pinned bottom and a free top, the shear modulus factors straight out of the
minimiser: scaling the energy does not move its minimum. The relaxed geometry
depends on Poisson's ratio and nothing else. Young's modulus, the shear modulus
and the tissue density have no effect and are deliberately not accepted here.
Poisson's ratio sets how strictly volume is preserved: at 0.5 the material is
incompressible and reproduces exactly the spiky answer, while around 0.45 an
extremely compressed column can shed a little volume instead of spiking. That
latitude is defensible because the extreme compressions in a flatmap are largely
artifacts of the flattening objective rather than properties of tissue.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

try:
    from scipy.sparse.linalg import factorized as _factorized
except ImportError:
    from scipy.sparse.linalg.dsolve import factorized as _factorized

from .misc import _memo
from .surface import Surface

__all__ = ["FlatSlab", "naive_prism_height", "legacy_js_height"]


# Decomposition of a triangular prism into three tetrahedra. Nodes 0, 1, 2 are
# the bottom (white matter) triangle and 3, 4, 5 the corresponding top (pial)
# triangle. This is the same decomposition `polyutils.misc.brick_vol` uses, and
# reference and deformed configurations are cut identically so that the
# deformation gradient of each tetrahedron is well defined.
_PRISM_TETS = np.array([[0, 1, 2, 4],
                        [0, 2, 3, 4],
                        [2, 3, 4, 5]])


def _prism_tets(polys, nverts):
    """Tetrahedra of the slab, as indices into a ``2 * nverts`` node array.

    Bottom vertex ``i`` is node ``i`` and top vertex ``i`` is node
    ``nverts + i``.

    Parameters
    ----------
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.
    nverts : int
        Number of vertices in one surface.

    Returns
    -------
    tets : 2D ndarray, shape (3 * total_polys, 4)
        Node indices of each tetrahedron.
    """
    prism = np.empty((len(polys), 6), dtype=np.int64)
    prism[:, :3] = polys
    prism[:, 3:] = np.asarray(polys) + nverts
    return prism[:, _PRISM_TETS].reshape(-1, 4)


def _edge_matrix(pts, tets):
    """Tetrahedron edge vectors as the columns of a 3x3 matrix per element."""
    x0 = pts[tets[:, 0]]
    return np.stack([pts[tets[:, 1]] - x0,
                     pts[tets[:, 2]] - x0,
                     pts[tets[:, 3]] - x0], axis=-1)


def _cofactor(F):
    """Cofactor matrix of a stack of 3x3 matrices, i.e. ``dJ/dF``.

    The columns are the cross products of the other two columns, which is both
    cheaper than an inverse and defined even where ``F`` is singular.
    """
    f0, f1, f2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
    return np.stack([np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)],
                    axis=-1)


def lame_parameters(poisson_ratio):
    """Lame parameters of the stable Neo-Hookean energy for a Poisson ratio.

    The shear modulus cancels out of this problem entirely (see the module
    docstring), so it is fixed at 1 in the small-strain limit and only the ratio
    carries meaning. The returned values include the rescaling from Smith, De
    Goes & Kim (2018) that makes the stable Neo-Hookean energy match linear
    elasticity for small deformations, so that `poisson_ratio` means what it
    says.

    Parameters
    ----------
    poisson_ratio : float
        Poisson's ratio, in [0, 0.5). Values approaching 0.5 are incompressible.

    Returns
    -------
    mu : float
        Shear modulus of the energy.
    lam : float
        First Lame parameter of the energy.
    alpha : float
        Rest stability constant, chosen so that an undeformed element is a
        stationary point of the energy.
    """
    if not 0.0 <= poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5), not %r"
                         % (poisson_ratio,))
    mu_lin = 1.0
    lam_lin = 2.0 * poisson_ratio / (1.0 - 2.0 * poisson_ratio) * mu_lin

    # Smith, De Goes & Kim (2018), section 3.4: the log(I_C + 1) term perturbs
    # the small-strain behaviour, and these shifted parameters undo it.
    mu = 4.0 / 3.0 * mu_lin
    lam = lam_lin + 5.0 / 6.0 * mu_lin

    # Setting dPsi/dF = 0 at F = I gives (3/4) mu + lam (1 - alpha) = 0.
    alpha = 1.0 + 3.0 * mu / (4.0 * lam)
    return mu, lam, alpha


def _energy_and_stress(F, mu, lam, alpha):
    """Stable Neo-Hookean energy density and first Piola-Kirchhoff stress.

    ``Psi = mu/2 (I_C - 3) + lam/2 (J - alpha)^2 - mu/2 log(I_C + 1)``

    from Smith, De Goes & Kim (2018). Unlike the textbook ``log J`` Neo-Hookean
    form this has no logarithm of the determinant and no matrix inverse, so both
    the energy and its gradient stay finite and correct even if an element
    inverts partway through the solve.
    """
    cof = _cofactor(F)
    J = np.einsum('ij,ij->i', F[:, :, 0], cof[:, :, 0])
    I1 = np.einsum('ijk,ijk->i', F, F)

    psi = (0.5 * mu * (I1 - 3.0) + 0.5 * lam * (J - alpha) ** 2
           - 0.5 * mu * np.log(I1 + 1.0))
    P = ((mu * (1.0 - 1.0 / (I1 + 1.0)))[:, None, None] * F
         + (lam * (J - alpha))[:, None, None] * cof)
    return psi, P


def naive_prism_height(flat, wm, pia, polys):
    """Height of a volume-preserving vertical prism over the flatmap.

    This is the straightforward reading of the bumpy flatmap idea: each column
    of tissue keeps its folded volume, and its base is the flattened triangle,
    so its height is one over the other. It is included because it is the thing
    the relaxation is meant to improve on -- it spikes wherever flattening
    compressed a triangle hard, and no amount of smoothing a field of ratios
    afterwards removes the spikes, because they dominate the mean.

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


def legacy_js_height(wm, pia, polys, smooth_areas=5, smooth_dists=20,
                     factor=0.1):
    """The bumpy flatmap height the webgl viewer used to compute in javascript.

    Kept so that the new relaxation can be compared against what actually
    shipped rather than against a description of it. Note that despite its name
    this never looked at the flatmap at all: the denominator was the *folded*
    white matter vertex area, so with ``A_pial = r**2 * A_wm`` the whole
    expression collapses to ``thickness * (1 + r + r**2) / 3``.

    Parameters
    ----------
    wm : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the white matter surface.
    pia : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the pial surface.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.
    smooth_areas : int, optional
        Number of smoothing iterations applied to the vertex areas. Default 5,
        matching the shipped viewer.
    smooth_dists : int, optional
        Number of smoothing iterations applied to the thickness. Default 20,
        matching the shipped viewer.
    factor : float, optional
        Smoothing step size. Default 0.1, matching the shipped viewer.

    Returns
    -------
    height : 1D ndarray, shape (total_verts,)
        Height of the pial surface above the flatmap at each vertex.
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


def _vertex_areas(pts, polys):
    """One third of the incident face area, summed at each vertex."""
    areas = _face_areas(pts, polys)
    return np.bincount(np.asarray(polys).ravel(), weights=np.repeat(areas, 3),
                       minlength=len(pts)) / 3.0


def _face_areas(pts, polys):
    """Area of each triangle."""
    ppts = pts[polys]
    cross = np.cross(ppts[:, 1] - ppts[:, 0], ppts[:, 2] - ppts[:, 0])
    return 0.5 * np.sqrt((cross ** 2).sum(-1))


def _flat_plane(flat):
    """Flatmap coordinates as a 3D point set lying in the z = 0 plane.

    Only the first two columns of a pycortex flat surface are the flatmap
    coordinates; the third is left over from the flattening and is ignored
    everywhere else too (see `cortex.brainctm`, which stores `pts[:, :2]`).
    """
    plane = np.zeros((len(flat), 3))
    plane[:, :2] = np.asarray(flat)[:, :2]
    return plane


def face_prism_volumes(wm, pia, polys):
    """Volume of the cortical slab over each triangle.

    Vectorised equivalent of ``[brick_vol(...) for face in polys]``: the prism
    between the white matter and pial triangles is cut into three tetrahedra and
    their volumes summed. Exact for a prism with planar-cut faces, unlike the
    conical frustum approximation the javascript used.

    Parameters
    ----------
    wm : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the white matter surface.
    pia : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the pial surface.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.

    Returns
    -------
    volumes : 1D ndarray, shape (total_polys,)
        Volume of the slab over each triangle.
    """
    nverts = len(wm)
    pts = np.vstack([wm, pia])
    tets = _prism_tets(polys, nverts)
    dm = _edge_matrix(pts, tets)
    vols = np.abs(np.linalg.det(dm)) / 6.0
    return vols.reshape(-1, 3).sum(1)


class FlatSlab(object):
    """The cortical slab, relaxed onto a flatmap.

    Treats the tissue between the white matter and pial surfaces as a soft
    elastic solid, pins its white matter side to the flatmap, and lets its pial
    side settle. Unlike a vertical-prism model the pial surface is free to move
    in-plane, so a column that flattening compressed hard spreads sideways
    against its neighbours instead of spiking upwards -- gyral crowns end up
    wider at the top than at the bottom and sulcal fundi narrower, and no
    post-hoc smoothing is needed.

    Parameters
    ----------
    flat : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in flatmap space. Only the first two columns are
        used; the flatmap is taken to lie in the z = 0 plane.
    wm : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the white matter surface.
    pia : 2D ndarray, shape (total_verts, 3)
        Location of each vertex on the pial surface.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices of the *flat* surface. Vertices that appear in
        no triangle -- the medial wall, which is cut away from the flatmap --
        get a zero offset and take no part in the relaxation.
    poisson_ratio : float, optional
        How strictly volume is preserved, in [0, 0.5). Approaching 0.5 gives an
        incompressible material and reproduces the spiky vertical-prism answer;
        the default 0.45 lets extreme compressions shed a little volume. This is
        the only material parameter that affects the result -- see the module
        docstring.
    correlation_length : float, optional
        Length scale, in the units of the surfaces, over which the initial guess
        is smoothed. Defaults to the median cortical thickness, which is the
        scale over which shear actually couples the slab. Only affects the
        starting point of the relaxation, not its solution.
    max_iter : int, optional
        Maximum number of L-BFGS iterations. Default 400.

    Attributes
    ----------
    info : dict
        Diagnostics from the last computed relaxation: element counts, energies,
        optimiser status and slab volume before and after.
    """
    def __init__(self, flat, wm, pia, polys, poisson_ratio=0.45,
                 correlation_length=None, max_iter=400):
        self.flat = np.asarray(flat, dtype=np.double)
        self.wm = np.asarray(wm, dtype=np.double)
        self.pia = np.asarray(pia, dtype=np.double)
        self.polys = np.asarray(polys)
        self.poisson_ratio = poisson_ratio
        self.correlation_length = correlation_length
        self.max_iter = max_iter
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

        Returns ``(mask, subflat, subwm, subpia, subpolys)`` with the three
        surfaces indexed down to the flatmap vertices and the triangles
        reindexed to match. `subflat` lies in the z = 0 plane.
        """
        mask = np.zeros(len(self.wm), dtype=bool)
        mask[self.polys.ravel()] = True

        vmap = np.zeros(len(self.wm), dtype=np.int64)
        vmap[mask] = np.arange(mask.sum())

        return (mask, _flat_plane(self.flat)[mask], self.wm[mask],
                self.pia[mask], vmap[self.polys])

    @property
    @_memo
    def thickness(self):
        """Distance between the white matter and pial surfaces at each vertex."""
        return np.sqrt(((self.pia - self.wm) ** 2).sum(1))

    @property
    @_memo
    def prism_offsets(self):
        """Starting guess: a smoothed, volume-preserving vertical prism.

        Solves for the height that preserves each column's folded volume, but
        does it in log-height and regularises it with a single screened-Poisson
        solve on the flatmap rather than by iterated smoothing. Working in the
        log keeps a twenty-fold compression as an offset of three rather than a
        twenty-fold spike, and makes the regulariser a geometric rather than an
        arithmetic mean, which is the right averaging for a ratio.

        Useful on its own as a cheap approximation -- it costs one sparse solve
        rather than a nonlinear optimisation -- but it still assumes vertical
        columns, so it has no in-plane component.

        Returns
        -------
        offsets : 2D ndarray, shape (total_verts, 3)
            Position of the pial surface relative to the flat white matter
            surface. The first two columns are zero.
        """
        mask, flat, wm, pia, polys = self._submesh

        vol = face_prism_volumes(wm, pia, polys)
        area = _face_areas(flat, polys)
        nv = len(flat)
        idx = polys.ravel()

        # Ratio of sums rather than sum of ratios: a triangle that flattening
        # crushed to nothing contributes almost no volume and almost no area, so
        # it barely moves the estimate, instead of contributing one enormous
        # ratio that dominates its neighbourhood.
        vvol = np.bincount(idx, weights=np.repeat(vol, 3), minlength=nv) / 3.0
        varea = np.bincount(idx, weights=np.repeat(area, 3), minlength=nv) / 3.0

        good = (vvol > 0) & (varea > 0)
        target = np.zeros(nv)
        target[good] = np.log(vvol[good] / varea[good])
        if good.any():
            target[~good] = np.median(target[good])

        lc = self.correlation_length
        if lc is None:
            lc = np.median(self.thickness[mask])

        surf = Surface(flat, polys)
        _, D, W, V = surf.laplace_operator
        lhs = (sparse.dia_matrix((D, [0]), (nv, nv)) + lc ** 2 * (V - W)).tocsc()

        # Vertices with no area anchor nothing and leave a singular row.
        goodrows = np.nonzero(D > 0)[0]
        solve = _factorized(lhs[goodrows][:, goodrows].tocsc())
        logheight = target.copy()
        logheight[goodrows] = solve((D * target)[goodrows])

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask, 2] = np.exp(logheight)
        return offsets

    def _elements(self):
        """Assemble the tetrahedra, dropping degenerate ones.

        Returns ``(tets, dm_inv, vol0, n_dropped)``.
        """
        mask, flat, wm, pia, polys = self._submesh
        tets = _prism_tets(polys, len(flat))
        dm = _edge_matrix(np.vstack([wm, pia]), tets)
        vol0 = np.abs(np.linalg.det(dm)) / 6.0

        # A prism with a collapsed reference volume has no shape to preserve and
        # an ill-conditioned deformation gradient. Its vertices are still held by
        # the neighbouring prisms.
        scale = np.median(vol0)
        keep = vol0 > 1e-9 * scale if scale > 0 else np.ones(len(vol0), bool)
        n_dropped = int((~keep).sum())

        return tets[keep], np.linalg.inv(dm[keep]), vol0[keep], n_dropped

    @property
    @_memo
    def relaxed(self):
        """The relaxed pial surface, as an offset from the flat white surface.

        Returns
        -------
        offsets : 2D ndarray, shape (total_verts, 3)
            Position of the pial surface relative to the flat white matter
            surface, in the units of the input surfaces. The first two columns
            are the in-plane slip and the third the height. Vertices off the
            flatmap are zero.
        """
        mask, flat, wm, pia, polys = self._submesh
        nv = len(flat)
        nnodes = 2 * nv

        tets, dm_inv, vol0, n_dropped = self._elements()
        mu, lam, alpha = lame_parameters(self.poisson_ratio)

        bottom = flat
        top0 = flat + self.prism_offsets[mask]

        dm_invT = np.swapaxes(dm_inv, 1, 2)
        flat_idx = tets.ravel()
        energies = []

        def objective(x):
            pts = np.vstack([bottom, x.reshape(nv, 3)])
            F = _edge_matrix(pts, tets) @ dm_inv
            psi, P = _energy_and_stress(F, mu, lam, alpha)

            # dE/dDs = vol0 * P * Dm^-T, whose columns are the gradients with
            # respect to the three tetrahedron nodes that define the edges; the
            # fourth node takes minus their sum.
            g = vol0[:, None, None] * (P @ dm_invT)
            node_g = np.empty((len(tets), 4, 3))
            node_g[:, 1:, :] = np.swapaxes(g, 1, 2)
            node_g[:, 0, :] = -node_g[:, 1:, :].sum(1)

            grad = np.empty((nnodes, 3))
            for c in range(3):
                grad[:, c] = np.bincount(flat_idx,
                                         weights=node_g[:, :, c].ravel(),
                                         minlength=nnodes)

            energy = float(np.dot(vol0, psi))
            energies.append(energy)
            return energy, grad[nv:].ravel()

        result = minimize(objective, top0.ravel(), jac=True,
                          method='L-BFGS-B',
                          options=dict(maxiter=self.max_iter,
                                       # a hard iteration can spend several
                                       # evaluations in its line search; do not
                                       # let that stop the solve early
                                       maxfun=4 * self.max_iter))

        top = result.x.reshape(nv, 3)
        self.info = dict(
            n_tets=len(tets),
            n_dropped=n_dropped,
            n_free_verts=nv,
            energy_initial=energies[0] if energies else None,
            energy_final=float(result.fun),
            energy_history=np.asarray(energies),
            gradient_norm=float(np.abs(result.jac).max()),
            iterations=int(result.nit),
            converged=bool(result.success),
            message=str(result.message),
            volume_folded=float(vol0.sum()),
            volume_relaxed=self._slab_volume(np.vstack([bottom, top]), tets),
            poisson_ratio=self.poisson_ratio,
        )

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask] = top - bottom
        return offsets

    @staticmethod
    def _slab_volume(pts, tets):
        """Total volume of a tetrahedralised slab."""
        return float(np.abs(np.linalg.det(_edge_matrix(pts, tets))).sum() / 6.0)
