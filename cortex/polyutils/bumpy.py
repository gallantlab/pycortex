"""Bumpy flatmaps: relaxing the cortical slab onto a flatmap.

A flatmap is made by cutting the white matter surface and flattening it. Cortex
is not infinitesimally thin, though, so if you imagine peeling the cortical slab
off the white matter and laying it down, the white matter side would end up flat
while the pial side would sit some distance above it -- bunched up thicker over
gyri, which flattening compresses, and stretched thinner over sulci, which
flattening expands. That relief is a curvature cue that survives even when the
flatmap is completely covered with data.

Preserving the volume of each column independently gives a height of folded
prism volume over flattened triangle area, which diverges wherever flattening
compressed a triangle to nothing.

This module computes the relief instead by treating the slab as a compressible
hyperelastic solid: its white matter face is pinned to the flatmap as a
Dirichlet boundary, its pial face is left free in all three dimensions, and the
elastic energy is minimised. Shear between neighbouring columns supplies the
regularisation, so a compressed column spreads laterally rather than extruding,
gyral crowns end up wider than their bases and sulcal fundi narrower, and the
result needs no smoothing.

Each triangular prism between the two surfaces is split into three tetrahedra
carrying the stable Neo-Hookean energy of Smith et al. (2018), which stays
finite and correctly signed under element inversion. The energy is minimised
with L-BFGS-B (Byrd et al., 1995) over the pial vertex positions.

The minimisation runs coarse to fine. A quasi-Newton method takes far longer to
resolve the long wavelength part of the answer -- how the sheet as a whole
slides as it settles -- than the local detail, and on a mesh of 150,000 vertices
that dominates the cost. Because a flatmap is planar, coarser meshes can be had
cheaply: take a maximal independent set of the vertices and retriangulate it in
two dimensions, rejecting the triangles Delaunay throws across the medial wall
and the relaxation cuts. Each level is about three to four times smaller than
the one above, so solving it costs a fraction as much, and its solution is
prolonged onto the next level by barycentric interpolation. The coarsest level
starts from a vertical-prism solution regularised by one screened-Poisson solve
in log-height.

Body forces are omitted: for gray matter the ratio of gravitational to elastic
stress over a 2.5 mm slab, ``rho * g * t / mu``, is of order 0.03. Without a
body force the shear modulus factors out of the minimiser, which leaves
Poisson's ratio as the only material parameter affecting the result. It controls
how strictly volume is preserved, and values below 0.5 let the extreme
compressions -- which are largely artifacts of the flattening objective rather
than properties of tissue -- shed volume rather than spike.

References
----------
Smith, B., De Goes, F. and Kim, T. (2018). Stable Neo-Hookean flesh simulation.
ACM Transactions on Graphics 37(2), 1-15.

Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm
for bound constrained optimization. SIAM Journal on Scientific Computing 16(5),
1190-1208.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial import QhullError

try:
    from scipy.sparse.linalg import factorized as _factorized
except ImportError:
    from scipy.sparse.linalg.dsolve import factorized as _factorized

from .misc import _memo
from .surface import Surface

__all__ = ["FlatSlab", "coarsen_flat_mesh", "face_prism_volumes",
           "lame_parameters", "legacy_js_height", "naive_prism_height",
           "prolongation_matrix"]


# Decomposition of a triangular prism into three tetrahedra. Nodes 0, 1, 2 are
# the bottom (white matter) triangle and 3, 4, 5 the corresponding top (pial)
# triangle. This is the same decomposition `polyutils.misc.brick_vol` uses, and
# reference and deformed configurations are cut identically so that the
# deformation gradient of each tetrahedron is well defined.
_PRISM_TETS = np.array([[0, 1, 2, 4],
                        [0, 2, 3, 4],
                        [2, 3, 4, 5]])


def _prism_tets(polys, nverts, layers=1):
    """Tetrahedra of the slab, as indices into a ``(layers + 1) * nverts`` array.

    Vertex ``i`` of layer ``L`` is node ``L * nverts + i``, with layer 0 the
    white matter side and layer `layers` the pial side. Each triangular prism
    between consecutive layers is cut into three tetrahedra.

    More than one layer matters. A single linear element cannot represent a
    shear profile across the slab, and shear across the slab is the whole
    mechanism holding the relief up: with one layer, neighbouring columns of
    tissue exchange material far too freely and the relief flattens out towards
    a slab of uniform thickness. See `FlatSlab.thickness_layers`.

    Parameters
    ----------
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.
    nverts : int
        Number of vertices in one surface.
    layers : int, optional
        Number of element layers through the thickness. Default 1.

    Returns
    -------
    tets : 2D ndarray, shape (3 * layers * total_polys, 4)
        Node indices of each tetrahedron.
    """
    polys = np.asarray(polys)
    prisms = []
    for layer in range(layers):
        prism = np.empty((len(polys), 6), dtype=np.int64)
        prism[:, :3] = polys + layer * nverts
        prism[:, 3:] = polys + (layer + 1) * nverts
        prisms.append(prism)
    return np.vstack(prisms)[:, _PRISM_TETS].reshape(-1, 4)


def _edge_matrix(pts, tets):
    """Tetrahedron edge vectors as the columns of a 3x3 matrix per element."""
    return np.swapaxes(_edge_matrix_T(pts, tets), 1, 2)


def _edge_matrix_T(pts, tets):
    """Tetrahedron edge vectors as the *rows* of a 3x3 matrix per element.

    The relaxation works in this transposed convention throughout because it
    keeps the last axis contiguous: a row slice of a C-ordered ``(n, 3, 3)``
    array is contiguous while a column slice is not, and cross products and
    matrix multiplies over strided views are several times slower.
    """
    x0 = pts[tets[:, 0]]
    return np.stack([pts[tets[:, 1]] - x0,
                     pts[tets[:, 2]] - x0,
                     pts[tets[:, 3]] - x0], axis=1)


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


def _planar_area(pts, polys):
    """Total area of a triangulation, using only the first two coordinates."""
    p = np.asarray(pts)[np.asarray(polys)][:, :, :2]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() / 2.0)


def coarsen_flat_mesh(pts, polys, max_hops=8, area_tol=0.005):
    """Build a coarser triangulation of the same flat region.

    Keeps a maximal independent set of the vertices -- so no two survivors were
    neighbours, which thins the mesh by a factor of three to four -- and
    retriangulates them. Retriangulating is only this easy because a flatmap is
    planar: the vertices can go straight into a two dimensional Delaunay
    triangulation instead of needing a surface-aware decimation.

    Delaunay triangulates the convex hull, though, so it bridges the medial wall
    and every relaxation cut. Those bridges join vertices that are close
    together in the plane but far apart across the surface, so a triangle is
    kept only if its vertices are within a few steps of each other in the *fine
    mesh's* own graph. How many steps that should be depends on how regular the
    mesh is, so it is chosen by measuring: the number picked is the one whose
    coarse mesh covers the same total area as the fine one.

    Parameters
    ----------
    pts : 2D ndarray, shape (total_verts, 2) or (total_verts, 3)
        Vertex positions. Only the first two columns are used.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices.
    max_hops : int, optional
        Largest neighbourhood, in steps through the fine mesh, that a coarse
        triangle may span. Default 8.
    area_tol : float, optional
        Stop widening the neighbourhood once the coarse mesh covers more than
        this fraction of extra area, which means it has started bridging.
        Default 0.005.

    Returns
    -------
    index : 1D ndarray
        Indices into `pts` of the vertices that were kept.
    coarse_polys : 2D ndarray, shape (n_coarse_polys, 3)
        Triangles of the coarse mesh, indexing into `index`.
    """
    pts = np.asarray(pts, dtype=np.double)
    polys = np.asarray(polys)
    nverts = len(pts)
    xy = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(nverts)])

    surf = Surface(xy, polys)
    adj = surf.adj.astype(bool)

    # Boundary vertices go first so that coarsening keeps the outline of the
    # flatmap and of its cuts rather than eating into them.
    boundary = surf.boundary_vertices
    order = np.concatenate([np.nonzero(boundary)[0], np.nonzero(~boundary)[0]])

    status = np.zeros(nverts, np.int8)
    indptr, indices = adj.indptr, adj.indices
    for vert in order:
        if status[vert] == 0:
            status[vert] = 1
            status[indices[indptr[vert]:indptr[vert + 1]]] = 2
    index = np.nonzero(status == 1)[0]

    if len(index) < 4:
        raise ValueError("mesh is too small to coarsen: %d vertices survive"
                         % len(index))

    simplices = Delaunay(pts[index, :2]).simplices
    fine_area = _planar_area(pts, polys)

    # Which survivors are within `hops` steps of each other in the fine graph.
    # Only the rows belonging to survivors are ever read, and they are a third
    # of the mesh, so restrict to them before multiplying rather than after:
    # `reach` is symmetric, so (reach**k)[index] is reach[index] @ reach**(k-1).
    reach = (adj + sparse.eye(nverts, dtype=bool, format='csr')).tocsr()
    step = (reach[index] @ reach).astype(bool)
    best = None
    for hops in range(3, max_hops + 1):
        step = (step @ reach).astype(bool)
        near = step[:, index].tocsr()
        keep = np.ones(len(simplices), bool)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            keep &= np.asarray(near[simplices[:, a], simplices[:, b]]).ravel()

        error = _planar_area(pts[index], simplices[keep]) / fine_area - 1.0
        if best is None or abs(error) < abs(best[0]):
            best = (error, keep)
        if error > area_tol:
            break

    return index, simplices[best[1]]


def prolongation_matrix(pts, index, coarse_polys):
    """Interpolate from a coarse flat mesh back onto the full vertex set.

    Returns the sparse matrix ``P`` for which ``P @ coarse`` is the piecewise
    linear interpolation of a coarse quantity at every vertex of `pts`. Rows for
    the few vertices that fall outside the coarse triangulation -- they sit just
    beyond its boundary -- take the value of the nearest coarse vertex instead.

    Parameters
    ----------
    pts : 2D ndarray, shape (total_verts, 2) or (total_verts, 3)
        Vertex positions to interpolate onto. Only the first two columns are used.
    index : 1D ndarray
        Indices into `pts` of the coarse vertices, as returned by
        `coarsen_flat_mesh`.
    coarse_polys : 2D ndarray, shape (n_coarse_polys, 3)
        Triangles of the coarse mesh, indexing into `index`.

    Returns
    -------
    prolong : sparse matrix, shape (total_verts, len(index))
    """
    from matplotlib.tri import Triangulation

    pts = np.asarray(pts, dtype=np.double)
    coarse = pts[index, :2]
    target = pts[:, :2]

    finder = Triangulation(coarse[:, 0], coarse[:, 1],
                           np.asarray(coarse_polys)).get_trifinder()
    located = finder(target[:, 0], target[:, 1])
    inside = located >= 0

    tris = np.asarray(coarse_polys)[located[inside]]
    corner = coarse[tris]
    v0 = corner[:, 1] - corner[:, 0]
    v1 = corner[:, 2] - corner[:, 0]
    v2 = target[inside] - corner[:, 0]
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / det
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / det

    rows = np.nonzero(inside)[0]
    row = np.concatenate([rows, rows, rows])
    col = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    weight = np.concatenate([1.0 - w1 - w2, w1, w2])

    if (~inside).any():
        outside = np.nonzero(~inside)[0]
        _, nearest = cKDTree(coarse).query(target[outside])
        row = np.concatenate([row, outside])
        col = np.concatenate([col, nearest])
        weight = np.concatenate([weight, np.ones(len(outside))])

    return sparse.coo_matrix((weight, (row, col)),
                             shape=(len(pts), len(index))).tocsr()


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


def _smooth_vectors(surf, vectors, factor):
    """`Surface.smooth`, applied to every column of `vectors` at once.

    `Surface.smooth` assembles and factorises the smoothing operator on each
    call. The operator depends only on the mesh, so smoothing a three-component
    field on a whole hemisphere the obvious way pays for three sparse
    factorisations of the same matrix -- which on a flatmap is most of the cost.
    Same operator, same answer, one factorisation and three back-substitutions.
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


def _prism_height(flat, wm, pia, polys, correlation_length):
    """Volume-preserving vertical prism height, regularised in log-height.

    Solves ``(M + lc^2 L) l = M l*`` once on the flat mesh, where ``l*`` is the
    log of the height that would preserve each column's folded volume, ``L`` is
    the cotangent stiffness matrix and ``M`` the lumped mass. Working in the log
    keeps a twenty-fold compression as an offset of three rather than a
    twenty-fold spike, and makes the regulariser a geometric rather than an
    arithmetic mean, which is the right averaging for a ratio.
    """
    vol = face_prism_volumes(wm, pia, polys)
    area = _face_areas(flat, polys)
    nverts = len(flat)
    idx = np.asarray(polys).ravel()

    # Ratio of sums rather than sum of ratios: a triangle that flattening
    # crushed to nothing contributes almost no volume and almost no area, so it
    # barely moves the estimate, instead of contributing one enormous ratio that
    # dominates its neighbourhood.
    vvol = np.bincount(idx, weights=np.repeat(vol, 3), minlength=nverts) / 3.0
    varea = np.bincount(idx, weights=np.repeat(area, 3), minlength=nverts) / 3.0

    good = (vvol > 0) & (varea > 0)
    target = np.zeros(nverts)
    target[good] = np.log(vvol[good] / varea[good])
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


def _slab_volume(pts, tets):
    """Total volume of a tetrahedralised slab."""
    return float(np.abs(np.linalg.det(_edge_matrix(pts, tets))).sum() / 6.0)


def _slab_nodes(inner, outer, layers):
    """Stack of ``layers + 1`` surfaces evenly spaced between two surfaces."""
    return np.vstack([inner + (outer - inner) * f
                      for f in np.linspace(0.0, 1.0, layers + 1)])


def _assemble_elements(flat, wm, pia, polys, layers=1):
    """Tetrahedra of the slab, dropping degenerate ones.

    Returns ``(tets, dm_inv, vol0, n_dropped)``.
    """
    tets = _prism_tets(polys, len(flat), layers)
    reference = _edge_matrix(_slab_nodes(wm, pia, layers), tets)
    vol0 = np.abs(np.linalg.det(reference)) / 6.0

    # A prism with a collapsed reference volume has no shape to preserve and an
    # ill-conditioned deformation gradient. Its vertices are still held by the
    # neighbouring prisms.
    scale = np.median(vol0)
    keep = vol0 > 1e-9 * scale if scale > 0 else np.ones(len(vol0), bool)

    return (tets[keep], np.linalg.inv(reference[keep]), vol0[keep],
            int((~keep).sum()))


class FlatSlab(object):
    """The cortical slab, relaxed onto a flatmap.

    Minimises the elastic energy of the tissue between the white matter and pial
    surfaces with the white matter side pinned to the flatmap and the pial side
    free in three dimensions. Allowing in-plane motion is what distinguishes
    this from a vertical-prism model: a compressed column spreads laterally
    rather than extruding, so the relief needs no smoothing. See the module
    docstring for the energy and the solver.

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
        incompressible material and recovers the vertical-prism answer including
        its spikes; the default 0.45 lets extreme compressions shed some volume.
        The only material parameter that affects the result.
    correlation_length : float, optional
        Length scale, in the units of the surfaces, over which the initial guess
        is smoothed. Default 0.5 mm. Pass None for the median cortical
        thickness, the scale over which shear couples the slab, which is the
        physically motivated choice and was the old default.

        In principle this affects only the starting point and not the solution.
        In practice it affects the solution, because the solve stops at
        `max_iter` rather than at a tolerance (see `max_iter`), so whatever the
        starting point lacks the relaxation has no chance to put back. The
        smoothing here is a screened-Poisson solve with transfer function
        1/(1 + lc^2 k^2), so its half-power point is at a wavelength of
        2*pi*lc -- for a 2.5 mm median thickness that is **15.7 mm**, which
        removes most of the gyral relief before the solve even starts. On S1,
        dropping it to 0.5 mm raises the amplitude in the 8-16 mm band by 27%
        with the correlation to mean curvature unchanged, i.e. it restores
        signal rather than noise. Converging the solve would be the principled
        fix and would make this parameter irrelevant again.
    max_iter : int, optional
        Maximum number of L-BFGS iterations, applied at every level. Default 60,
        which is calibrated against the hierarchy and against `polish`: what the
        iterations past this point buy is short-wavelength detail, and that is
        smoothed off again afterwards. On S1, raising it to 100 costs a third
        more time and moves the correlation between the polished relief and mean
        curvature by 0.001. Solving with ``levels=1`` needs two to three times
        more.

        Be aware that this is a *cap*, not a tolerance, and at the default the
        solve stops here rather than converging -- on S1 ``info['converged']``
        is False with a final gradient sup-norm around 3e3 against scipy's
        1e-5. That is a deliberate trade for import time, but it has a
        consequence worth knowing: an unconverged solve stays near its starting
        point, so `correlation_length`, which nominally sets only that starting
        point, ends up shaping the answer.
    levels : int, optional
        How many meshes to use, counting the full one. The relaxation is solved
        coarse to fine; each extra level is roughly three to four times smaller
        than the one above it. Default 3. Pass 1 to solve the full mesh directly.
    thickness_layers : int, optional
        Number of element layers through the thickness of the slab. Default 3.
        This is not a refinement knob to be traded away: a single layer cannot
        represent shear across the slab, which is the mechanism holding the
        relief up, and with one layer the relief flattens towards a slab of
        uniform thickness. On a patch of S1 the correlation between the relief
        and mean curvature is 0.52 with one layer against 0.79 with four. Almost
        all of that is recovered by three, which is why three is the default:
        over a whole hemisphere a fourth layer costs 60% more time and moves the
        correlation by 0.002, in the wrong direction.
    smooth : float, optional
        How much to smooth the white-to-pial displacement before using it as the
        elastic reference, as a diffusion time in the units of the surfaces
        squared. **Default 0, i.e. off**, and it should usually stay off.

        This existed to keep segmentation noise out of the relief, and it does,
        but it is the wrong tool: a diffusion time t has its half-power point at
        a wavelength of 2*pi*sqrt(t), so the old default of 1.0 was cutting at
        **6.3 mm** -- squarely inside the gyral band, smoothing away the very
        thickness variation that makes gyri thicker than sulci. On S1 it roughly
        halved the correlation between the relief and mean curvature, from 0.47
        to 0.25 in the 8-16 mm band and 0.62 to 0.42 in 16-32 mm, and even a
        value of 0.05 measured slightly worse than zero. `polish` removes the
        same noise afterwards for a fraction of the signal, so there is no
        regime where this helps. Raise it only for a segmentation whose
        thickness map is visibly noisier than the folding it sits on.
    resolution : float, optional
        Coarsest mesh spacing, in the units of the surfaces, that is worth
        solving at. Levels finer than this are reached by interpolation rather
        than solved. Default 3.2 mm, which on S1 means solving a 3.1 mm mesh
        rather than the 0.9 mm one. This is the setting that buys the speed and
        it is not free: solving down to 1.6 mm takes five times as long and does
        carry more gyral-scale signal. The default is chosen so that a subject
        import costs about a minute rather than about ten; raise it, or lower it
        towards 1.5, according to which of those matters.

        Note that at this default the ``floor`` clamp leaves **only the coarsest
        level actually solved**, so the coarse-to-fine cascade -- which exists
        because long wavelengths are what a quasi-Newton method converges last
        -- is not doing anything. Adding levels *coarser* than the floor would
        cost very little and is the obvious next thing to try. Pass 0 to solve
        every level
        including the full mesh -- which on a whole hemisphere takes upwards of
        forty minutes and is not recommended.
    polish : float, optional
        How much to smooth the finished offsets, as a diffusion time in the
        units of the surfaces squared. Default 2.0.

        This is not cosmetic: levels finer than `resolution` are reached by
        barycentric interpolation, which is only continuous and not smooth, so
        the height field has creases along the coarse triangle edges. A shading
        normal is the derivative of that field, which makes every crease a
        visible discontinuity in the lighting even though the heights themselves
        look fine. On S1 this takes the RMS angle between the normals of
        neighbouring triangles from 4.5 degrees to 2.8.

        It is also the parameter most easily overdone, because the obvious
        metrics for "is it smooth" all improve monotonically with it while the
        anatomy quietly leaves. Its half-power wavelength is 2*pi*sqrt(polish),
        so the old default of 4.0 was cutting at **12.6 mm** and removing 35% of
        the amplitude in the 8-16 mm gyral band. 2.0 cuts at 8.9 mm; 0.5, which
        cuts at 4.4 mm, keeps 21% more gyral relief at the cost of a visibly
        rougher surface (silk 3.8 degrees against 2.8). Pass 0 to get the
        prolonged solution as it comes.

        The creases are the real culprit and this only hides them. Making the
        prolongation C1, or solving the level the creases live on, would let
        this drop by an order of magnitude.
    detrend : float, optional
        Remove relief at wavelengths longer than this, in the units of the
        surfaces. Default 64 mm; pass 0 to keep it.

        The relief has a whole-map swell in it. On S1 the band above 64 mm holds
        a standard deviation of 0.42 mm against a total of 0.70 -- **36% of the
        variance in one mode broad enough to span the flatmap**. It is real:
        cortex is regionally thicker in some lobes than others, and that is what
        this mostly is. But it is not folding, it correlates with mean curvature
        at only 0.30 where the gyral bands manage 0.62-0.66, and the viewer's
        ``bumpy_flatmap_scale`` multiplies it along with everything else. So it
        spends most of the available relief on the least informative thing in
        the field, and the gyri ride on top of it, small.

        Removing it drops the total relief from 0.70 mm to 0.46 mm while leaving
        the gyral bands almost untouched (they lose 10-15%), so turning the
        scale slider up by the same 1.5x gets back to the same overall height
        with the folding half again as strong. It also raises the correlation
        with mean curvature in every band, most in the longest: 0.30 to 0.56
        above 64 mm, 0.66 to 0.71 in 32-64 mm.

        Note this is the *opposite* of exaggerating the low frequencies, which
        is the intuitive move and the wrong one. Boosting the long wavelengths
        boosts this swell hardest, which reads as the whole map getting taller
        rather than as hills and valleys, and measurably makes the gyri weaker
        per unit of height -- a factor-3 long-wavelength boost leaves the 8-32 mm
        bands at 0.78 of their original share. Shaded relief is a high-pass
        filter, so the way to make folding read more strongly is to stop
        spending amplitude on things larger than folding.

    Attributes
    ----------
    info : dict
        Diagnostics from the last computed relaxation: element counts, energies,
        optimiser status and slab volume before and after.
    """
    def __init__(self, flat, wm, pia, polys, poisson_ratio=0.45,
                 correlation_length=0.5, max_iter=60, levels=3,
                 thickness_layers=3, smooth=0.0, resolution=3.2, polish=2.0,
                 detrend=64.0):
        self.flat = np.asarray(flat, dtype=np.double)
        self.wm = np.asarray(wm, dtype=np.double)
        self.pia = np.asarray(pia, dtype=np.double)
        self.polys = np.asarray(polys)
        self.poisson_ratio = poisson_ratio
        self.correlation_length = correlation_length
        self.max_iter = max_iter
        self.levels = levels
        self.thickness_layers = thickness_layers
        self.smooth = smooth
        self.resolution = resolution
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
    def smoothed_pia(self):
        """The pial surface with the millimetre-scale wobble taken out.

        Smooths the displacement from white matter to pia rather than the pial
        coordinates themselves, so the folding is left exactly as it is and only
        the local thickness is regularised. What this removes is segmentation
        noise: cortical thickness does not genuinely vary from one vertex to the
        next, and the relaxation would otherwise reproduce every wobble of it
        faithfully as relief.
        """
        if not self.smooth:
            return self.pia

        surf = Surface(self.wm, self.polys)
        return self.wm + _smooth_vectors(surf, self.pia - self.wm, self.smooth)

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
        mask, flat, wm, _, polys = self._submesh
        lc = self.correlation_length
        if lc is None:
            lc = np.median(self.thickness[mask])

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask, 2] = _prism_height(flat, wm, self.smoothed_pia[mask],
                                         polys, lc)
        return offsets

    @property
    @_memo
    def _hierarchy(self):
        """Progressively coarser versions of the flatmap, finest first.

        Each entry is ``(index into the flatmap submesh, triangles in this
        level's own numbering, index into the level above)``. The top level is
        the submesh itself and has no parent.
        """
        _, flat, _, _, polys = self._submesh
        levels = [(np.arange(len(flat)), polys, None)]

        for _ in range(self.levels - 1):
            parent, parent_polys, _ = levels[-1]
            try:
                index, coarse_polys = coarsen_flat_mesh(flat[parent],
                                                        parent_polys)
            except (ValueError, QhullError):
                break
            # Below a few thousand triangles the solve is quick anyway and the
            # coarse mesh stops resembling the surface.
            if len(coarse_polys) < 2000:
                break
            levels.append((parent[index], coarse_polys, index))

        return levels

    def _solve_level(self, flat, wm, pia, polys, top0, max_iter):
        """Minimise the elastic energy of one mesh, starting from `top0`.

        Returns ``(moved, info)``, where `moved` stacks the relaxed positions
        of every layer above the pinned white matter side, pial layer last.
        """
        layers = self.thickness_layers
        nverts = len(flat)
        nnodes = (layers + 1) * nverts
        tets, dm_inv, vol0, n_dropped = _assemble_elements(flat, wm, pia, polys,
                                                           layers)
        mu, lam, alpha = lame_parameters(self.poisson_ratio)

        # Everything below is transposed relative to the textbook formulas, so
        # that every slice taken per element has a contiguous last axis. Written
        # the other way round this loop spends most of its time walking strided
        # views, and there are hundreds of thousands of elements.
        dm_invT = np.ascontiguousarray(np.swapaxes(dm_inv, 1, 2))
        dm_inv_vol = np.ascontiguousarray(vol0[:, None, None] * dm_inv)
        # One flat scatter beats three strided ones: the gradient is treated as
        # a flat (nnodes * 3,) array so that the weights can be handed to
        # bincount without copying them out of the per-element array first.
        scatter = (tets[:, :, None] * 3 + np.arange(3)).ravel()
        energies = []

        def objective(x):
            # only the white matter side is pinned; every layer above it moves
            pts = np.vstack([flat, x.reshape(layers * nverts, 3)])
            deformed = _edge_matrix_T(pts, tets)
            defgrad = dm_invT @ deformed                     # F transposed

            rows = defgrad[:, 0, :], defgrad[:, 1, :], defgrad[:, 2, :]
            cof = np.stack([np.cross(rows[1], rows[2]),
                            np.cross(rows[2], rows[0]),
                            np.cross(rows[0], rows[1])], axis=1)
            det = np.einsum('ij,ij->i', rows[0], cof[:, 0, :])
            trace = np.einsum('ijk,ijk->i', defgrad, defgrad)

            psi = (0.5 * mu * (trace - 3.0)
                   + 0.5 * lam * (det - alpha) ** 2
                   - 0.5 * mu * np.log(trace + 1.0))
            stress = ((mu * (1.0 - 1.0 / (trace + 1.0)))[:, None, None] * defgrad
                      + (lam * (det - alpha))[:, None, None] * cof)

            # dE/dDs = vol0 * P * Dm^-T; transposed, that is vol0 * Dm^-1 * P^T,
            # whose rows are the gradients with respect to the three nodes that
            # define the edges. The fourth node takes minus their sum.
            node_g = np.empty((len(tets), 4, 3))
            np.matmul(dm_inv_vol, stress, out=node_g[:, 1:, :])
            node_g[:, 0, :] = -node_g[:, 1:, :].sum(1)

            grad = np.bincount(scatter, weights=node_g.ravel(),
                               minlength=nnodes * 3).reshape(nnodes, 3)

            energy = float(np.dot(vol0, psi))
            energies.append(energy)
            return energy, grad[nverts:].ravel()

        result = minimize(objective, top0.ravel(), jac=True,
                          method='L-BFGS-B',
                          options=dict(maxiter=max_iter,
                                       # a hard iteration can spend several
                                       # evaluations in its line search; do not
                                       # let that stop the solve early
                                       maxfun=4 * max_iter,
                                       # L-BFGS keeps this many correction pairs
                                       # to model the curvature with. The default
                                       # of 10 is far too few here: on S1 raising
                                       # it to 60 reaches a 29% lower energy in
                                       # the same number of iterations, for about
                                       # 10% more time per iteration.
                                       maxcor=60))

        moved = result.x.reshape(layers * nverts, 3)
        info = dict(
            n_tets=len(tets),
            thickness_layers=layers,
            n_dropped=n_dropped,
            n_free_verts=nverts,
            energy_initial=energies[0] if energies else None,
            energy_final=float(result.fun),
            energy_history=np.asarray(energies),
            gradient_norm=float(np.abs(result.jac).max()),
            iterations=int(result.nit),
            converged=bool(result.success),
            message=str(result.message),
            volume_folded=float(vol0.sum()),
            volume_relaxed=_slab_volume(np.vstack([flat, moved]), tets),
        )
        return moved, info

    @property
    @_memo
    def relaxed(self):
        """The relaxed pial surface, as an offset from the flat white surface.

        Solved coarse to fine. A cortical flatmap is around 150,000 vertices and
        the long wavelength part of the answer -- how the whole sheet slides
        around as it settles -- is what a quasi-Newton method takes longest to
        find. Solving a mesh three or four times smaller first is cheap, finds
        exactly that part, and leaves the full mesh only the short wavelength
        detail to fill in.

        Returns
        -------
        offsets : 2D ndarray, shape (total_verts, 3)
            Position of the pial surface relative to the flat white matter
            surface, in the units of the input surfaces. The first two columns
            are the in-plane slip and the third the height. Vertices off the
            flatmap are zero.
        """
        mask, flat, wm, pia, polys = self._submesh
        pia = self.smoothed_pia[mask]
        hierarchy = self._hierarchy
        layers = self.thickness_layers

        lc = self.correlation_length
        if lc is None:
            lc = np.median(self.thickness[mask])

        # Levels finer than `resolution` are interpolated rather than solved.
        # The relief varies over millimetres, so resolving it at the scale of a
        # single triangle costs a great deal and adds nothing but the
        # opportunity to wrinkle.
        floor = 0
        if self.resolution:
            for depth, (index, level_polys, _) in enumerate(hierarchy):
                spacing = Surface(flat[index], level_polys).avg_edge_length
                if spacing >= self.resolution:
                    break
                floor = depth + 1
            floor = min(floor, len(hierarchy) - 1)

        moved = None
        levelinfo = []
        for depth in range(len(hierarchy) - 1, -1, -1):
            index, level_polys, _ = hierarchy[depth]
            level_flat, level_wm, level_pia = flat[index], wm[index], pia[index]
            stack = np.tile(level_flat, (layers, 1))

            if moved is None:
                # The coarsest mesh starts from the vertical prism solution,
                # with the intermediate layers spread evenly up to it.
                height = _prism_height(level_flat, level_wm, level_pia,
                                       level_polys, lc)
                rise = np.concatenate(
                    [np.zeros((len(index), 2)), height[:, None]], axis=1)
                start = np.vstack([rise * f for f in
                                   np.linspace(0, 1, layers + 1)[1:]])
            else:
                _, child_polys, index_in_parent = hierarchy[depth + 1]
                prolong = prolongation_matrix(level_flat, index_in_parent,
                                              child_polys)
                start = np.vstack([prolong @ coarse_offsets[i]
                                   for i in range(layers)])
            top0 = stack + start

            if depth < floor:
                # Fine enough that solving it would only add wrinkles; take the
                # interpolated coarse answer as it stands.
                moved = top0
            else:
                # Every solved level gets the same iteration count, which means
                # the coarse ones cost a fraction of the finest solved mesh:
                # about a third per coarsening step. Giving them more than this
                # was measurably a waste -- they were already converged well
                # past the point where the finer mesh could tell the difference.
                moved, info = self._solve_level(level_flat, level_wm, level_pia,
                                                level_polys, top0, self.max_iter)
                info['level'] = depth
                levelinfo.append(info)

            # (layers, nverts, 3); the node stack is block-major by layer, so
            # this reshape is a view rather than an interleave. Needed on every
            # level, solved or not, since the next one down prolongs from it.
            coarse_offsets = (moved - stack).reshape(layers, len(index), 3)

        top = moved[-len(flat):]          # outermost layer is the pial side
        relief = top - flat

        if self.polish:
            # Two things to take out. The obvious one is what is left of the
            # millimetre-scale noise. The less obvious one is that every level
            # below `floor` was reached by barycentric interpolation, which is
            # only continuous and not smooth: the height field has creases along
            # the coarse triangle edges. A shading normal is the derivative of
            # that field, so a crease is a visible discontinuity in the
            # lighting, and this is what turns it back into a surface.
            relief = _smooth_vectors(Surface(flat, polys), relief, self.polish)

        if self.detrend:
            # Take out the whole-map swell. On S1 the band above 64 mm holds a
            # standard deviation of 0.42 mm against a total of 0.70 -- 36% of
            # the relief's variance in a single mode broad enough to span the
            # flatmap. It is real (cortex is regionally thicker in some lobes
            # than others) but it is not folding, it correlates with mean
            # curvature at only 0.30 against 0.62-0.66 for the gyral bands, and
            # the viewer's scale slider multiplies it along with everything
            # else. So it spends most of the available relief on the least
            # informative thing in the field, and the gyri ride on top of it
            # small. Removing it lets the slider be turned up further before the
            # map looks warped, which is worth about 1.5x on the gyral bands.
            surf = Surface(flat, polys)
            swell = _smooth_vectors(surf, relief,
                                    (self.detrend / (2 * np.pi)) ** 2)
            # about its own mean, so the sheet keeps its average thickness
            relief = relief - (swell - swell.mean(0))

        self.info = dict(levelinfo[-1])
        self.info['solved_from_level'] = floor
        self.info['levels'] = levelinfo[::-1]
        self.info['poisson_ratio'] = self.poisson_ratio

        offsets = np.zeros((len(self.wm), 3))
        offsets[mask] = relief
        return offsets
