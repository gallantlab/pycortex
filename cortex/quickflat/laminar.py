import functools

import numpy as np
from matplotlib.tri import Triangulation
import cortex
from cortex.mapper import samplers
from scipy.spatial import Delaunay

def find_triangle(verts, faces, pts):
    """verts (V,2), faces (F,3), pts (N,2) -> (tri_idx, bary); tri_idx = -1 if outside.
        verts: flat
        faces: polys
    """
    idx = Triangulation(verts[:, 0], verts[:, 1], faces).get_trifinder()(pts[:, 0], pts[:, 1])
    A = (verts[faces[idx, :2]] - verts[faces[idx, 2], None]).transpose(0, 2, 1)   # (N,2,2)
    lam = np.einsum('nij,nj->ni', np.linalg.inv(A), pts - verts[faces[idx, 2]])
    bary = np.column_stack([lam, 1 - lam.sum(1)])
    bary[idx < 0] = np.nan
    return idx, bary

def line_interpolation(u_l, v_l, u_r, v_r, gamma):
    u_gamma = gamma * u_l + (1 - gamma) * u_r
    v_gamma = gamma * v_l + (1 - gamma) * v_r
    return u_gamma, v_gamma

def _as_vertex_index(valid):
    """Normalize `valid` (boolean mask or integer index array) to integer indices."""
    valid = np.asarray(valid)
    if valid.dtype == bool:
        valid = np.flatnonzero(valid)
    return valid

def _column_geometry(u_x, v_x, dl, pia, wm, valid):
    """Per-point flatmap -> surface geometry, vectorized over N points.

    Everything here depends only on the flatmap position (u_x, v_x), not on the
    cortical depth alpha, so it is computed once per column rather than once per
    pixel.

    Returns x_p (N,3), x_w (N,3), A_p (N,), A_w (N,), inside (N,) bool mask.
    Rows where `inside` is False (point outside the flatmap hull) are NaN.
    """
    valid = _as_vertex_index(valid)
    pts = np.column_stack([np.ravel(v_x), np.ravel(u_x)]).astype(float)

    simp = dl.find_simplex(pts)
    inside = simp >= 0
    simp_ok = np.where(inside, simp, 0)   # dummy index for outside points

    # Barycentric coordinates, batched over all points at once.
    tf = dl.transform[simp_ok]                                     # (N,3,2)
    lam = np.einsum('nij,nj->ni', tf[:, :2, :2], pts - tf[:, 2, :])  # (N,2)
    bary = np.column_stack([lam, 1 - lam.sum(1)])                   # (N,3)

    # Index only the triangles we actually touched. Going through `valid` here
    # (rather than pia[valid][...]) avoids copying the whole vertex array.
    tri = valid[dl.simplices[simp_ok]]      # (N,3) vertex ids into pia/wm
    P = pia[tri]                            # (N,3,3)
    W_ = wm[tri]                            # (N,3,3)

    x_p = np.einsum('ni,nij->nj', bary, P)
    x_w = np.einsum('ni,nij->nj', bary, W_)

    A_p = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    A_w = 0.5 * np.linalg.norm(np.cross(W_[:, 1] - W_[:, 0], W_[:, 2] - W_[:, 0]), axis=1)

    if not inside.all():
        x_p = np.where(inside[:, None], x_p, np.nan)
        x_w = np.where(inside[:, None], x_w, np.nan)
        A_p = np.where(inside, A_p, np.nan)
        A_w = np.where(inside, A_w, np.nan)

    return x_p, x_w, A_p, A_w, inside

def _depth_blend(alpha, x_p, x_w, A_p, A_w):
    """Equal-volume depth interpolation between pial (alpha=0) and white (alpha=1).

    `alpha` broadcasts against the leading axis of the geometry arrays, so an
    (H,1) alpha and (W,) geometry produce an (H,W,3) result in one shot.
    """
    denom = A_p - A_w
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = 1 - (np.sqrt((1 - alpha) * A_p**2 + alpha * A_w**2) - A_w) / denom
    # As A_w -> A_p the expression above is 0/0; its limit is beta = alpha.
    degenerate = np.abs(denom) <= 1e-9 * np.maximum(A_p, A_w)
    beta = np.where(degenerate, np.broadcast_to(alpha, np.shape(beta)), beta)

    beta = beta[..., None]
    return (1 - beta) * x_p + beta * x_w

def locate_depth_point(u_x, v_x, alpha, dl, pia, wm, valid):
    '''
    u_x, v_x: flatmap coordinates; scalars or arrays that broadcast together
    alpha: depth values in [0, 1], broadcasting against u_x/v_x
    returns: (..., 3) points in the 3D space
    '''
    scalar = all(np.ndim(a) == 0 for a in (u_x, v_x, alpha))
    u_x, v_x, alpha = np.broadcast_arrays(*np.atleast_1d(u_x, v_x, alpha))

    x_p, x_w, A_p, A_w, _ = _column_geometry(u_x, v_x, dl, pia, wm, valid)
    z = _depth_blend(np.ravel(alpha), x_p, x_w, A_p, A_w)
    return z[0] if scalar else z.reshape(u_x.shape + (3,))

@functools.lru_cache(maxsize=4)
def _laminar_geometry(subject, xfmname):
    """Load and triangulate a subject's surfaces. Cached: this is the expensive
    part of make_laminar_profile (~2s), and it is identical for every profile
    drawn on the same subject. Call _laminar_geometry.cache_clear() if the
    surfaces on disk change."""
    xfm = cortex.db.get_xfm(subject, xfmname, xfmtype="coord")

    verts, faces = cortex.db.get_surf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(faces)   # find vertices in flatmap triangles
    dl = Delaunay(verts[valid, :2])
    dl.transform    # scipy builds this lazily; do it once here, not on first query

    pia = xfm(cortex.db.get_surf(subject, "pia", merge=True, nudge=False)[0])
    wm = xfm(cortex.db.get_surf(subject, "wm", merge=True, nudge=False)[0])
    return xfm, dl, pia, wm, valid

def make_laminar_profile(subject, xfmname, u_l, v_l, u_r, v_r, W, H, sampler="nearest"):
    """
    Given a line in flatmap space with endpoints (L,R) and dimensions (W,H), create a cortical profile map that is W pixels wide and H pixels high.
    Along the width, gamma will vary from 0 to 1. Along the height, alpha will vary from 0 to 1.
    For each (gamma,alpha), compute (z), find the enclosing voxel, and then retrieve the value at that voxel. Plot.

    Returns an (H,W) int array of flat voxel indices. Pixels that fall outside
    the flatmap or outside the volume are set to -1.
    """

    sampclass = getattr(samplers, sampler)
    # Create a grid of gamma and alpha values
    gamma_values = np.linspace(0, 1, W)
    alpha_values = np.linspace(0, 1, H)

    xfm, dl, pia, wm, valid = _laminar_geometry(subject, xfmname)

    # The flatmap -> surface geometry depends only on gamma, so it is resolved
    # once per column (W points) instead of once per pixel (W*H points).
    u_gamma, v_gamma = line_interpolation(u_l, v_l, u_r, v_r, gamma_values)
    x_p, x_w, A_p, A_w, _ = _column_geometry(u_gamma, v_gamma, dl, pia, wm, valid)

    # Only the depth blend varies with alpha; broadcast it over the whole grid.
    z = _depth_blend(alpha_values[:, None], x_p, x_w, A_p, A_w)   # (H,W,3)

    profile_map = np.full(H * W, -1, int)
    i, vox, _ = sampclass(z.reshape(-1, 3), xfm.shape)
    # Reversed so the first entry wins for samplers that emit several voxels per
    # point (e.g. trilinear), matching the old `vox[0]`.
    profile_map[i[::-1]] = vox[::-1]

    return profile_map.reshape(H, W)
