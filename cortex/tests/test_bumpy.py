"""Tests for the bumpy flatmap relaxation in `cortex.polyutils.bumpy`.

Most of these check the relaxation against a case where the answer is known
exactly rather than against a stored result, so they say something about whether
the physics is right and not just whether it changed.
"""

import numpy as np
import pytest
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
    slab = bumpy.FlatSlab(flat, wm, pia, polys, poisson_ratio=0.45,
                          max_iter=4000)
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
