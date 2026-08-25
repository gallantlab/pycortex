"""Tests for the bumpy flatmap in `cortex.polyutils.bumpy`.

These check the relief against properties that can be stated in advance --
folding produces relief at constant thickness, the height ignores flatmap
distortion, a ridge stays a ridge -- rather than against a stored result, so
they say whether the answer is right and not just whether it changed.
"""

import numpy as np
from scipy.spatial import Delaunay

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


def test_vertices_off_the_flatmap_are_untouched():
    """Vertices in no flat triangle -- the medial wall -- get a zero offset."""
    flat, wm, pia, polys, index = slab_grid(n=9, stretch=1.1)
    # Cut a corner out of the flatmap, the way the medial wall is cut away.
    corner = {index[0, 0], index[0, 1], index[1, 0]}
    keep = np.array([not (set(f) & corner) for f in polys])

    offsets = bumpy.FlatSlab(flat, wm, pia, polys[keep]).relaxed

    off_map = np.ones(len(wm), bool)
    off_map[polys[keep].ravel()] = False
    assert off_map.sum() > 0
    assert np.abs(offsets[off_map]).max() == 0.0
    assert np.abs(offsets[~off_map, 2]).min() > 0.0


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
    """`_smooth_vectors` must agree with the loop it replaces, including
    leaving vertices in no triangle at zero.
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

    A shading normal is the derivative of the height field, so noise that is
    invisible in the heights is glaring in the lighting.
    """
    flat, wm, pia, polys, _ = _s1_patch()
    plane = bumpy._flat_plane(flat)

    slab = bumpy.FlatSlab(flat, wm, pia, polys)
    relief = slab.relaxed
    # Two thresholds, because a bare angle is not scale-free -- the angle
    # between neighbouring faces grows with height for the same shape. The
    # absolute number is what the eye sees; the ratio is smoothness per unit
    # relief, and it is the one that says the surface itself is not rough.
    assert _silk(plane + relief, polys) < 4.5
    assert _silk(plane + relief, polys) / relief[:, 2].std() < 5.0

    rough = bumpy.FlatSlab(flat, wm, pia, polys, polish=0)
    assert _silk(plane + rough.relaxed, polys) > _silk(plane + slab.relaxed,
                                                       polys)


def _band(surf, x, wavelength):
    """Low-pass at a given wavelength.

    `Surface.smooth`'s `factor` is a diffusion time, and one backward-Euler step
    has transfer function 1/(1 + k^2 t), so `factor = t` is half power at a
    wavelength of 2*pi*sqrt(t) -- not sqrt(t), and not the sqrt(2t) the
    Gaussian-equivalent sigma suggests. It is easy to be wrong here by 4.4x.
    """
    return surf.smooth(x.copy(), (wavelength / (2 * np.pi)) ** 2)


def test_the_relief_carries_gyral_scale_signal():
    """The relief has to have energy at the scale of gyri, not just be smooth.

    Every other measure here is local -- `_silk` compares neighbouring faces,
    and a correlation against raw curvature is dominated by whichever band has
    most variance. None of them can see a relief that is beautifully smooth and
    anatomically empty, which is what over-smoothing produces.
    """
    flat, wm, pia, polys, curv = _s1_patch()
    surf = polyutils.Surface(bumpy._flat_plane(flat), polys)
    height = bumpy.FlatSlab(flat, wm, pia, polys).relaxed[:, 2]

    def gyral(x):
        """The 8-16 mm band, where the folding lives."""
        return _band(surf, x, 16.0) - _band(surf, x, 8.0)

    band, cband = gyral(height), gyral(curv)
    assert np.corrcoef(band, cband)[0, 1] > 0.35, (
        "the relief has no folding signal left at gyral scale")

    # and it must not have been smoothed into a featureless sheet: the gyral
    # band has to hold real amplitude, not a rounding error on the mean
    assert band.std() / height.std() > 0.1, (
        "the gyral band has been smoothed away relative to the whole relief")


def _elongation(surf, field, scale=8.0):
    """Structure-tensor coherence: 1 for a ridge, 0 for a round bump.

    Measured from the field itself, which is what keeps it an independent check
    even when the smoothing has been oriented by curvature.
    """
    g = surf.surface_gradient(np.asarray(field, float), at_verts=False)
    idx = np.array([(0, 0), (0, 1), (1, 1)])
    packed = (g[:, :, None] * g[:, None, :])[:, idx[:, 0], idx[:, 1]]

    area = surf.face_areas
    w = np.maximum(np.asarray(surf.connected.dot(area)).ravel(), 1e-20)
    v = surf.connected.dot(area[:, None] * packed) / w[:, None]
    v = bumpy._smooth_vectors(surf, v, (scale / (2 * np.pi)) ** 2)
    packed = v[surf.polys].mean(1)

    sxx, sxy, syy = packed[:, 0], packed[:, 1], packed[:, 2]
    tr = sxx + syy
    disc = np.sqrt(np.maximum((sxx - syy) ** 2 + 4 * sxy ** 2, 0.0))
    ok = tr > 1e-20
    return float((disc[ok]).sum() / tr[ok].sum())


def test_the_elongation_measure_tells_ridges_from_knobs():
    """Pin the instrument before trusting what it says about cortex."""
    g = np.linspace(0, 48, 120)
    X, Y = np.meshgrid(g, g)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    surf = polyutils.Surface(np.column_stack([xy, np.zeros(len(xy))]),
                             Delaunay(xy).simplices)
    x, y = xy[:, 0], xy[:, 1]

    ridges = np.sin(2 * np.pi * x / 8.0)
    knobs = ridges * np.sin(2 * np.pi * y / 8.0)
    assert _elongation(surf, ridges) > 0.9
    assert _elongation(surf, knobs) < 0.4


def test_a_ridge_stays_a_ridge():
    """The whole point: an elongated thickness ridge must not come out beaded.

    Every other synthetic case in this file is a square grid stretched equally
    in both directions, so none of them can tell an elongated relief from a
    chain of round bumps -- which is exactly the defect this guards. The slab is
    given a thickness ridge running along y, and the relief has to come back at
    least as elongated as an honest measurement of that ridge.
    """
    n, spacing = 41, 1.0
    flat, wm, pia, polys, index = slab_grid(n=n, spacing=spacing, thickness=2.0)

    # a ridge along y: thicker in a 6 mm-wide band, uniform down its length
    x = wm[:, 0] - wm[:, 0].mean()
    bump = 1.2 * np.exp(-(x / 3.0) ** 2)
    pia = pia + np.column_stack([np.zeros(len(x)), np.zeros(len(x)), bump])

    surf = polyutils.Surface(bumpy._flat_plane(flat), polys)
    relief = bumpy.FlatSlab(flat, wm, pia, polys).relaxed[:, 2]

    assert _elongation(surf, relief) > 0.85, (
        "the relief lost the ridge's direction; it is beaded rather than "
        "elongated")
    # and it has to still be a ridge in the right place, not just elongated
    assert np.corrcoef(relief, bump)[0, 1] > 0.8


def _curved_ridge_slab(n=41, thickness=2.0, height=2.0, width=5.0):
    """A slab of *exactly* constant thickness, folded into a ridge along y.

    The pia is offset along the surface normal, so on the convex crown it has
    more area than the white matter beneath it and in the flanks less. Thickness
    is uniform to machine precision, so any relief this produces comes from the
    folding and from nothing else.
    """
    g = np.arange(n) * 1.0
    X, Y = np.meshgrid(g, g, indexing='ij')
    z = height * np.exp(-((X - g[-1] / 2) / width) ** 2)
    wm = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)

    gx, gy = np.gradient(z, 1.0, 1.0, axis=(0, 1))
    normal = np.stack([-gx.ravel(), -gy.ravel(), np.ones(n * n)], axis=1)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    pia = wm + thickness * normal

    index = np.arange(n * n).reshape(n, n)
    polys = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = index[i, j], index[i + 1, j]
            c, d = index[i + 1, j + 1], index[i, j + 1]
            polys += [[a, b, c], [a, c, d]]
    flat = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
    return flat, wm, pia, np.array(polys), z.ravel()


def test_folding_alone_produces_relief():
    """Constant thickness, folded: there must still be a bump on the crown.

    This is the property the relief was missing. Cortical thickness is a fairly
    blobby field and on its own it gives a relief of round knobs; what makes a
    flatmap look like gyri is the pial flare, the pia carrying more area than
    the white matter beneath a crown. Here thickness is uniform to machine
    precision, so the flare is the *only* signal available and a quantity that
    ignores it would return a flat sheet.
    """
    flat, wm, pia, polys, crown = _curved_ridge_slab()

    thickness = np.linalg.norm(pia - wm, axis=1)
    assert thickness.std() < 1e-12, "the fixture is supposed to be uniform"

    height = bumpy.folding_height(flat, wm, pia, polys)
    assert height.std() > 0.02 * thickness.mean()
    assert np.corrcoef(height, crown)[0, 1] > 0.6


def test_folding_height_ignores_the_flatmap_distortion():
    """The point of the new denominator, stated as a property.

    `folding_height` divides by the *folded* white matter area, so distorting
    the flatmap must not change it -- the flatmap enters only as the mesh the
    regularisation is solved on. `naive_prism_height` divides by the
    *flattened* area, so the same distortion moves it a great deal. That
    difference is the whole reason for the change: a flatmap's area distortion
    measures essentially uncorrelated with curvature, so as a denominator it
    contributes no folding and injects the flattening algorithm's artifacts
    instead.
    """
    flat, wm, pia, polys, _ = _curved_ridge_slab(n=31)

    # a smooth, folding-unrelated area distortion of the flatmap
    warped = flat.copy()
    warped[:, 0] *= 1.0 + 0.3 * np.sin(2 * np.pi * flat[:, 1] / 30.0)

    def shift(fn):
        a, b = fn(flat), fn(warped)
        return np.abs(b - a).mean() / a.mean()

    folding = shift(lambda f: bumpy.folding_height(f, wm, pia, polys))
    prism = shift(lambda f: bumpy.naive_prism_height(f, wm, pia, polys))
    assert folding < 0.02
    assert prism > 5 * folding


def test_folding_height_is_the_frustum_over_the_white_area():
    """Pin the algebra: V_frustum / A_wm, which is what legacy computed."""
    flat, wm, pia, polys, _ = _curved_ridge_slab(n=21)

    awm = bumpy._vertex_areas(wm, polys)
    apia = bumpy._vertex_areas(pia, polys)
    r = np.sqrt(apia / awm)
    thickness = np.linalg.norm(pia - wm, axis=1)
    expected = thickness * (1 + r + r ** 2) / 3.0

    # unregularised comparison, so use a correlation length short enough that
    # the smoothing is not what is being tested
    got = bumpy.folding_height(flat, wm, pia, polys, correlation_length=1e-4)
    np.testing.assert_allclose(got, expected, rtol=2e-3)
