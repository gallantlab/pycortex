"""Tests for the bumpy flatmap relaxation in `cortex.polyutils.bumpy`.

Most of these check the relaxation against a case where the answer is known
exactly rather than against a stored result, so they say something about whether
the physics is right and not just whether it changed.
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize_scalar

from cortex import polyutils
from cortex.polyutils import bumpy


def slab_grid(n=13, spacing=1.0, thickness=2.5, stretch=1.0):
    """A flat rectangular slab, optionally flattened with a uniform stretch.

    Returns ``(flat, wm, pia, polys, index)`` where `index` maps a grid position
    to its vertex number, so a test can pick out the middle of the patch and
    stay clear of the free edges.
    """
    xs = np.arange(n) * spacing
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    wm = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    pia = wm + np.array([0., 0., thickness])

    flat = wm.copy()
    flat[:, :2] *= stretch

    index = np.arange(n * n).reshape(n, n)
    polys = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = index[i, j], index[i + 1, j]
            c, d = index[i + 1, j + 1], index[i, j + 1]
            polys += [[a, b, c], [a, c, d]]
    return flat, wm, pia, np.array(polys), index


def homogeneous_stretch(stretch, poisson_ratio):
    """Vertical stretch of a uniformly, biaxially stretched incompressible-ish block.

    The exact finite-strain answer for the same energy, found by minimising over
    the one remaining degree of freedom. In the interior of a thin slab the
    relaxation has to reproduce this.
    """
    mu, lam, alpha = bumpy.lame_parameters(poisson_ratio)

    def energy(s):
        F = np.diag([stretch, stretch, s])[None]
        return bumpy._energy_and_stress(F, mu, lam, alpha)[0][0]

    return minimize_scalar(energy, bracket=(0.5, 1.0, 1.5), tol=1e-14).x


def test_prism_volume_matches_brick_vol():
    """The vectorised prism volume agrees with the reference implementation."""
    rng = np.random.default_rng(0)
    wm = rng.normal(size=(30, 3))
    pia = wm + rng.normal(size=(30, 3)) * 0.3 + np.array([0., 0., 2.])
    polys = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [7, 8, 9]])

    expected = np.array([polyutils.brick_vol(np.append(wm[f], pia[f], axis=0))
                         for f in polys])
    got = bumpy.face_prism_volumes(wm, pia, polys)
    assert np.allclose(got, expected)


@pytest.mark.parametrize("poisson_ratio", [0.0, 0.2, 0.35, 0.45, 0.49])
def test_energy_is_stable_at_rest(poisson_ratio):
    """An undeformed element sits at a stationary point of the energy."""
    mu, lam, alpha = bumpy.lame_parameters(poisson_ratio)
    _, stress = bumpy._energy_and_stress(np.eye(3)[None], mu, lam, alpha)
    assert np.abs(stress).max() < 1e-12


@pytest.mark.parametrize("poisson_ratio", [0.0, 0.2, 0.35, 0.45])
def test_poisson_ratio_means_poisson_ratio(poisson_ratio):
    """The small-strain moduli are the ones that were asked for.

    The stable Neo-Hookean energy's ``log(I_C + 1)`` term perturbs its
    behaviour away from linear elasticity, so `lame_parameters` shifts the Lame
    parameters to compensate. If that shift were wrong, `poisson_ratio` would
    quietly mean something else -- and since it is the only material parameter
    that affects the result, nothing else would catch it.
    """
    mu, lam, alpha = bumpy.lame_parameters(poisson_ratio)
    h = 1e-6

    def dstress(i, j, k, l):
        F = np.repeat(np.eye(3)[None], 2, axis=0)
        F[0, k, l] += h
        F[1, k, l] -= h
        stress = bumpy._energy_and_stress(F, mu, lam, alpha)[1]
        return (stress[0, i, j] - stress[1, i, j]) / (2 * h)

    lam_eff = dstress(0, 0, 1, 1)                       # C_1122
    mu_eff = 0.5 * (dstress(0, 0, 0, 0) - lam_eff)      # C_1111 = lam + 2 mu
    assert mu_eff == pytest.approx(1.0, abs=1e-6)
    assert lam_eff / (2 * (lam_eff + mu_eff)) == pytest.approx(poisson_ratio,
                                                               abs=1e-6)


def test_unflattened_slab_is_left_alone():
    """With an identity flat map the reference slab is already the minimiser.

    Every height has to come out as the cortical thickness and nothing may slide
    sideways. This exercises the whole assembly -- tetrahedra, deformation
    gradients, gradient scatter and boundary conditions -- against an answer
    that is known exactly.
    """
    flat, wm, pia, polys, _ = slab_grid(thickness=2.5)
    slab = bumpy.FlatSlab(flat, wm, pia, polys, poisson_ratio=0.45)
    offsets = slab.relaxed

    assert np.abs(offsets[:, :2]).max() < 1e-9
    assert np.abs(offsets[:, 2] - 2.5).max() < 1e-9
    assert slab.info['n_dropped'] == 0


@pytest.mark.parametrize("poisson_ratio", [0.0, 0.35, 0.45])
def test_uniform_stretch_matches_the_exact_solution(poisson_ratio):
    """A uniformly stretched thin slab reproduces the homogeneous answer.

    The slab is made wide relative to its thickness so that the free edges,
    whose boundary layer is a few thicknesses across, do not reach the middle.
    """
    stretch = 1.02
    thickness = 0.5
    flat, wm, pia, polys, index = slab_grid(n=21, spacing=1.0,
                                            thickness=thickness,
                                            stretch=stretch)
    slab = bumpy.FlatSlab(flat, wm, pia, polys, poisson_ratio=poisson_ratio,
                          max_iter=4000)
    offsets = slab.relaxed

    middle = index[10, 10]
    expected = thickness * homogeneous_stretch(stretch, poisson_ratio)
    assert offsets[middle, 2] == pytest.approx(expected, rel=1e-3)
    # By symmetry the middle of the patch has nowhere to slide.
    assert np.abs(offsets[middle, :2]).max() < 1e-3 * thickness


def test_incompressible_slab_conserves_volume():
    """As Poisson's ratio approaches 0.5 the slab stops losing volume."""
    flat, wm, pia, polys, _ = slab_grid(n=15, thickness=1.0, stretch=1.3)

    losses = []
    for poisson_ratio in (0.3, 0.45, 0.499):
        slab = bumpy.FlatSlab(flat, wm, pia, polys,
                              poisson_ratio=poisson_ratio, max_iter=4000)
        slab.relaxed
        losses.append(abs(slab.info['volume_relaxed']
                          / slab.info['volume_folded'] - 1))

    assert losses[0] > losses[1] > losses[2]
    assert losses[-1] < 0.01


def test_vertices_off_the_flatmap_are_untouched():
    """Vertices in no flat triangle -- the medial wall -- get a zero offset."""
    flat, wm, pia, polys, index = slab_grid(n=9, stretch=1.1)
    # Cut a corner out of the flatmap, the way the medial wall is cut away.
    corner = {index[0, 0], index[0, 1], index[1, 0]}
    keep = np.array([not (set(f) & corner) for f in polys])

    slab = bumpy.FlatSlab(flat, wm, pia, polys[keep], poisson_ratio=0.45)
    offsets = slab.relaxed

    off_map = np.ones(len(wm), bool)
    off_map[polys[keep].ravel()] = False
    assert off_map.sum() > 0
    assert np.abs(offsets[off_map]).max() == 0.0
    assert np.abs(offsets[~off_map, 2]).min() > 0.0


def test_relaxation_tames_the_spikes_the_naive_height_has():
    """The whole point: the relaxed height has a far shorter tail.

    A patch whose flattening compresses one small region hard is exactly the
    situation that makes the naive vertical-prism height blow up.
    """
    n = 21
    flat, wm, pia, polys, index = slab_grid(n=n, spacing=1.0, thickness=2.0)
    # Squeeze a disc in the middle of the flatmap towards its centre, leaving
    # the folded surfaces alone: locally the flattening lost a lot of area.
    centre = flat[:, :2].mean(0)
    radial = flat[:, :2] - centre
    dist = np.linalg.norm(radial, axis=1)
    squeeze = np.clip(1.0 - 0.9 * np.exp(-(dist / 3.0) ** 2), 0.05, 1.0)
    flat[:, :2] = centre + radial * squeeze[:, None]

    naive = bumpy.naive_prism_height(flat, wm, pia, polys)
    # `polish` off: it is a fixed diffusion time calibrated to the scale of a
    # real flatmap, where a millimetre or two of smoothing is small next to a
    # gyrus. This grid is 20 units across and the squeezed disc 3 units wide, so
    # at the default it would smooth away most of the very feature under test --
    # which says nothing about the relaxation, and that is what is under test.
    slab = bumpy.FlatSlab(flat, wm, pia, polys, poisson_ratio=0.45,
                          max_iter=4000, polish=0)
    relaxed = slab.relaxed[:, 2]

    assert naive.max() > 4 * relaxed.max()
    # and the relief still has to be there, not flattened away
    assert relaxed.max() > 1.2 * np.median(relaxed)


def test_matches_a_real_flatmap_patch():
    """Runs on a patch of a real subject, with its real cuts and distortions."""
    from cortex import db
    wm, polys = db.get_surf("S1", "wm", "lh")
    pia, _ = db.get_surf("S1", "pia", "lh")
    flat, flatpolys = db.get_surf("S1", "flat", "lh")

    # extract_chunk only carries one auxiliary point set and this needs three,
    # so take a patch by masking instead.
    surf = polyutils.Surface(flat, flatpolys)
    seed = flatpolys[len(flatpolys) // 2, 0]
    patch = np.zeros(len(flat), bool)
    patch[surf.get_euclidean_patch(seed, 15)['vertex_mask']] = True
    keep = patch[flatpolys].all(1)
    assert keep.sum() > 100

    slab = bumpy.FlatSlab(flat, wm, pia, flatpolys[keep], poisson_ratio=0.45,
                          max_iter=200)
    offsets = slab.relaxed

    inpatch = np.zeros(len(flat), bool)
    inpatch[flatpolys[keep].ravel()] = True
    heights = offsets[inpatch, 2]
    thickness = np.linalg.norm(pia - wm, axis=1)[inpatch]

    assert np.isfinite(offsets).all()
    assert (heights > 0).all()
    # relief should be of the same order as cortical thickness, not orders off
    assert 0.2 < np.median(heights) / np.median(thickness) < 5.0
    assert np.abs(offsets[~inpatch]).max() == 0.0


def flat_grid(n=15, spacing=1.0):
    """A flat regular grid of unit right triangles, two per grid cell.

    Returns ``(pts, polys, index)`` where `pts` has shape ``(n * n, 2)`` and
    `index[i, j]` is the vertex number at grid position (i, j).
    """
    xs = np.arange(n) * spacing
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    index = np.arange(n * n).reshape(n, n)
    polys = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = index[i, j], index[i + 1, j]
            c, d = index[i + 1, j + 1], index[i, j + 1]
            polys += [[a, b, c], [a, c, d]]
    return pts, np.array(polys), index


def flat_grid_with_hole(n=25, spacing=1.0, hole=(9, 16)):
    """The same grid, with a square block of cells cut out of the middle.

    `hole` is a half-open ``(lo, hi)`` range of grid cell indices removed in
    both directions, leaving a mesh with a genuine interior hole -- a boundary
    loop with no triangles inside it -- rather than just an edge.
    """
    pts, _, index = flat_grid(n, spacing)
    lo, hi = hole
    polys = []
    for i in range(n - 1):
        for j in range(n - 1):
            if lo <= i < hi and lo <= j < hi:
                continue
            a, b = index[i, j], index[i + 1, j]
            c, d = index[i + 1, j + 1], index[i, j + 1]
            polys += [[a, b, c], [a, c, d]]
    return pts, np.array(polys), index


def test_coarsen_keeps_an_independent_set():
    """No two surviving vertices were neighbours in the fine mesh.

    That is what a maximal independent set means: retriangulating a pair of
    neighbours would put a coarse edge across a gap that was close to zero in
    the fine mesh, which is not a genuine coarsening.
    """
    pts, polys, _ = flat_grid(n=15)
    index, _ = bumpy.coarsen_flat_mesh(pts, polys)

    adj = polyutils.Surface(pts, polys).adj.astype(bool).tocsr()
    kept = np.zeros(len(pts), bool)
    kept[index] = True
    for vert in index:
        neighbours = adj.indices[adj.indptr[vert]:adj.indptr[vert + 1]]
        assert not kept[neighbours].any()


def test_coarsen_reduces_vertex_count():
    """The coarse mesh keeps a small, nonzero fraction of the fine vertices.

    An interior vertex of this grid has six neighbours, so a maximal
    independent set keeps roughly one vertex in four; anything close to one in
    one would mean the independent-set step was not doing anything.
    """
    pts, polys, _ = flat_grid(n=25)
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)
    assert 0 < len(index) <= len(pts) // 2
    assert len(coarse_polys) > 0


def test_coarsen_indices_are_valid():
    """`index` names real vertices and `coarse_polys` indexes into `index`."""
    pts, polys, _ = flat_grid(n=21)
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)

    assert index.ndim == 1
    assert np.issubdtype(index.dtype, np.integer)
    assert (index >= 0).all() and (index < len(pts)).all()
    assert len(np.unique(index)) == len(index)

    assert coarse_polys.ndim == 2 and coarse_polys.shape[1] == 3
    assert (coarse_polys >= 0).all() and (coarse_polys < len(index)).all()
    assert (coarse_polys[:, 0] != coarse_polys[:, 1]).all()
    assert (coarse_polys[:, 1] != coarse_polys[:, 2]).all()
    assert (coarse_polys[:, 0] != coarse_polys[:, 2]).all()


def test_coarsen_preserves_area_on_a_simple_patch():
    """On a hole-free patch the coarse mesh covers almost the same area.

    `coarsen_flat_mesh` widens its bridging tolerance until the coarse area
    stops tracking the fine one, so on a simply connected patch -- where there
    is nothing to bridge -- it should stop close to the true area rather than
    drifting out to the tolerance's edge.
    """
    pts, polys, _ = flat_grid(n=21)
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)

    fine_area = bumpy._planar_area(pts, polys)
    coarse_area = bumpy._planar_area(pts[index], coarse_polys)
    assert coarse_area == pytest.approx(fine_area, rel=0.02)


def test_coarsen_does_not_bridge_a_hole():
    """A hole in the fine mesh must stay a hole in the coarse mesh.

    Plain 2D Delaunay triangulates the convex hull of the surviving vertices,
    so without the fine-mesh-distance rejection the coarse mesh would bridge
    straight over the missing block and its area would jump close to the
    *filled-in* square's area instead of staying near the true,
    hole-punctured area. This is the property the whole hop-counting scheme
    in `coarsen_flat_mesh` exists to protect.
    """
    n, hole = 25, (9, 16)
    pts, polys, _ = flat_grid_with_hole(n=n, hole=hole)
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)

    fine_area = bumpy._planar_area(pts, polys)
    coarse_area = bumpy._planar_area(pts[index], coarse_polys)
    filled_area = float((n - 1) * (n - 1))
    hole_area = float((hole[1] - hole[0]) ** 2)
    assert filled_area - fine_area == pytest.approx(hole_area, abs=1e-9)

    assert coarse_area == pytest.approx(fine_area, rel=0.02)
    # A bridged coarse mesh would land close to the filled-in area; demand
    # that the coarse area instead stay far closer to the true, holed area
    # than to what bridging the hole would have produced.
    assert abs(coarse_area - fine_area) < 0.1 * hole_area
    assert abs(coarse_area - filled_area) > 0.5 * hole_area


def test_prolongation_reproduces_linear_fields_inside_the_hull():
    """Barycentric interpolation is exact for affine functions.

    `P @ f_coarse` must equal an affine `f` evaluated at every fine vertex that
    falls inside the coarse triangulation. Some vertices here do not -- the
    coarse mesh has a hole in it too, per the previous test -- and those take a
    nearest-neighbour value instead, which is not exact for a sloped field. A
    nearest-neighbour row has exactly one stored entry, equal to 1.0; an
    inside row always has three stored entries (possibly with some exactly
    zero), one per corner of the triangle it was found in, even when the
    point sits exactly on a coarse vertex. So `P.getnnz(axis=1) == 1` reliably
    picks out the excluded rows without needing to know which vertices those
    are ahead of time.
    """
    pts, polys, _ = flat_grid_with_hole(n=25, hole=(9, 16))
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)
    P = bumpy.prolongation_matrix(pts, index, coarse_polys)

    def f(xy):
        return 3.0 + 2.0 * xy[:, 0] - 5.0 * xy[:, 1]

    result = np.asarray(P @ f(pts[index])).ravel()
    expected = f(pts)

    outside = P.getnnz(axis=1) == 1
    assert 0 < outside.sum() < len(pts)  # the hole boundary really produces some
    inside = ~outside
    assert np.abs(result[inside] - expected[inside]).max() < 1e-9


def test_prolongation_is_a_partition_of_unity():
    """Every row of `P` sums to 1, and no weight is negative.

    Both hold for the inside-triangle barycentric rows and for the
    outside-hull nearest-neighbour rows, so this needs no masking, unlike the
    linear-field check above.
    """
    pts, polys, _ = flat_grid_with_hole(n=25, hole=(9, 16))
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)
    P = bumpy.prolongation_matrix(pts, index, coarse_polys)

    rowsums = np.asarray(P.sum(axis=1)).ravel()
    assert np.abs(rowsums - 1.0).max() < 1e-10
    assert P.data.min() >= -1e-12


def test_prolongation_shape_and_output_length():
    """`P` has the documented shape and works for scalar- and vector-valued data."""
    pts, polys, _ = flat_grid(n=15)
    index, coarse_polys = bumpy.coarsen_flat_mesh(pts, polys)
    P = bumpy.prolongation_matrix(pts, index, coarse_polys)

    assert sparse.issparse(P)
    assert P.shape == (len(pts), len(index))

    scalar = np.arange(len(index), dtype=float)
    assert (P @ scalar).shape == (len(pts),)

    vector = np.column_stack([scalar, -scalar, 2.0 * scalar])
    assert (P @ vector).shape == (len(pts), 3)


def test_coarse_to_fine_builds_a_shrinking_hierarchy():
    """Each level of the hierarchy is a strictly smaller mesh than the one above.

    The levels also have to be nested -- every coarse vertex is a vertex of the
    level above it -- because that is what lets a coarse solution be prolonged
    onto the finer mesh at all.
    """
    flat, wm, pia, polys, _ = slab_grid(n=70, spacing=1.0, thickness=1.5,
                                        stretch=1.15)
    slab = bumpy.FlatSlab(flat, wm, pia, polys, levels=3)
    hierarchy = slab._hierarchy

    assert len(hierarchy) > 1, "the mesh was big enough to coarsen"
    for (parent, parent_polys, _), (child, child_polys, in_parent) in zip(
            hierarchy, hierarchy[1:]):
        assert len(child) < len(parent)
        assert len(child_polys) < len(parent_polys)
        # nested: the child's vertices are a subset of the parent's, and
        # `in_parent` says where each one sits in the parent's numbering
        assert np.array_equal(parent[in_parent], child)
        assert child_polys.max() < len(child)


def test_coarse_to_fine_matches_a_single_level_solve():
    """Solving coarse to fine changes how long the answer takes, not the answer.

    Both are minimising the same energy over the same mesh; the hierarchy only
    supplies a better starting point. On a patch small enough for both to
    converge properly they have to agree, and if prolongation were putting the
    fine mesh somewhere the single-level solve does not go, this is where it
    would show.

    Both are given ``resolution=0`` so that the finest mesh is actually solved.
    At the default the hierarchy stops short of it and interpolates instead,
    which is a deliberately different answer and not what is under test here.
    """
    flat, wm, pia, polys, _ = slab_grid(n=70, spacing=1.0, thickness=1.5,
                                        stretch=1.15)
    single = bumpy.FlatSlab(flat, wm, pia, polys, levels=1, max_iter=800,
                            resolution=0)
    multi = bumpy.FlatSlab(flat, wm, pia, polys, levels=3, max_iter=800,
                           resolution=0)
    one, many = single.relaxed, multi.relaxed

    assert single.info['converged'] and multi.info['converged']
    assert multi.info['energy_final'] == pytest.approx(
        single.info['energy_final'], rel=1e-6)

    height = np.abs(one[:, 2] - many[:, 2]) / np.maximum(one[:, 2], 1e-9)
    assert np.median(height) < 1e-3
    assert height.max() < 1e-2
    assert np.abs(one[:, :2] - many[:, :2]).max() < 1e-2


def _s1_patch(radius=32):
    """A patch of S1's flatmap, with its white, pial and curvature data."""
    from cortex import db
    wm, _ = db.get_surf("S1", "wm", "lh")
    pia, _ = db.get_surf("S1", "pia", "lh")
    flat, flatpolys = db.get_surf("S1", "flat", "lh")
    curv = db.get_surfinfo("S1", type="curvature").data[:len(wm)]

    surf = polyutils.Surface(bumpy._flat_plane(flat), flatpolys)
    seed = flatpolys[len(flatpolys) // 2, 0]
    inside = np.zeros(len(flat), bool)
    inside[surf.get_euclidean_patch(seed, radius)["vertex_mask"]] = True
    keep = inside[flatpolys].all(1)

    sub = np.zeros(len(flat), bool)
    sub[flatpolys[keep].ravel()] = True
    remap = np.zeros(len(flat), np.int64)
    remap[sub] = np.arange(sub.sum())
    return (flat[sub], wm[sub], pia[sub], remap[flatpolys[keep]], curv[sub])


def test_smoothing_three_components_at_once_matches_smoothing_them_singly():
    """`_smooth_vectors` is `Surface.smooth` with the factorisation shared.

    It exists only because the operator does not depend on what is being
    smoothed, so it must give back exactly what the obvious loop gives back --
    including leaving vertices in no triangle at zero.
    """
    flat, wm, pia, polys, _ = _s1_patch(radius=16)
    surf = polyutils.Surface(bumpy._flat_plane(flat), polys)
    field = pia - wm

    for factor in (0, 1.0, 4.0):
        expected = np.column_stack(
            [surf.smooth(field[:, k].copy(), factor) for k in range(3)])
        np.testing.assert_allclose(
            bumpy._smooth_vectors(surf, field, factor), expected, atol=1e-10)


def _silk(pts, polys):
    """RMS angle, in degrees, between the normals of neighbouring triangles.

    This is what "smooth" means to a shader: the shading normal is the
    derivative of the surface, so it is the angle between neighbours and not
    the height itself that decides whether the relief reads as silk or as
    crumpled foil.
    """
    tri = pts[polys]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]

    edges = np.sort(np.vstack([polys[:, [0, 1]], polys[:, [1, 2]],
                               polys[:, [2, 0]]]), axis=1)
    face = np.tile(np.arange(len(polys)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, f = edges[order], face[order]
    shared = np.all(e[:-1] == e[1:], axis=1)
    a, b = f[:-1][shared], f[1:][shared]

    dot = np.clip((n[a] * n[b]).sum(1), -1, 1)
    return np.degrees(np.sqrt((np.arccos(dot) ** 2).mean()))


def test_the_relief_is_smooth_enough_to_shade():
    """The bumped flatmap has to be smooth at the scale of a triangle.

    Two things put creases into it. The obvious one is what is left of the
    millimetre-scale noise in the segmentation. The less obvious one is that
    every level finer than `resolution` is reached by barycentric
    interpolation, which is only continuous and not smooth, so the height
    field has creases along the coarse triangle edges -- invisible in the
    heights and glaring in the lighting. `polish` is what takes both out.
    """
    flat, wm, pia, polys, _ = _s1_patch()
    plane = bumpy._flat_plane(flat)

    slab = bumpy.FlatSlab(flat, wm, pia, polys)
    assert _silk(plane + slab.relaxed, polys) < 3.5

    rough = bumpy.FlatSlab(flat, wm, pia, polys, polish=0)
    assert _silk(plane + rough.relaxed, polys) > _silk(plane + slab.relaxed,
                                                       polys)


def test_relief_follows_the_folding_and_is_not_mesh_scale_noise():
    """The relief has to track the folding, and not at the scale of a triangle.

    Both halves of this regressed once and neither was caught, because the
    relaxation was only ever checked for energy and volume. A slab one element
    thick cannot represent shear across itself, which is the mechanism holding
    the relief up, so the relief flattened towards uniform thickness; and with
    no regularisation at all, millimetre-scale segmentation noise went straight
    into it. The result looked dimpled rather than like hills and valleys, and
    every scalar being measured at the time got better while it happened.
    """
    flat, wm, pia, polys, curv = _s1_patch()

    surf = polyutils.Surface(bumpy._flat_plane(flat), polys)
    adj = surf.adj.tocsr()
    degree = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)

    def roughness(x):
        """Mesh-scale variation, as a fraction of the field's own spread."""
        return np.sqrt(((x - np.asarray(adj @ x).ravel() / degree) ** 2).mean()) \
            / x.std()

    height = bumpy.FlatSlab(flat, wm, pia, polys).relaxed[:, 2]
    assert np.corrcoef(height, curv)[0, 1] > 0.7
    assert roughness(height) < 0.15

    # and pin the reason, so that turning either off fails here rather than
    # silently degrading what the viewer shows
    thin = bumpy.FlatSlab(flat, wm, pia, polys, thickness_layers=1,
                          smooth=0).relaxed[:, 2]
    assert np.corrcoef(thin, curv)[0, 1] < np.corrcoef(height, curv)[0, 1]
    assert roughness(thin) > roughness(height)
