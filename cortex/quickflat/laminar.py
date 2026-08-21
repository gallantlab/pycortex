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
    A = (verts[faces[:, :2]] - verts[faces[:, 2], None]).transpose(0, 2, 1)   # (F,2,2)
    lam = np.einsum('nij,nj->ni', np.linalg.inv(A)[idx], pts - verts[faces[idx, 2]])
    bary = np.column_stack([lam, 1 - lam.sum(1)])
    bary[idx < 0] = np.nan
    return idx, bary

def line_interpolation(u_l, v_l, u_r, v_r, gamma):
    u_gamma = gamma * u_l + (1 - gamma) * u_r
    v_gamma = gamma * v_l + (1 - gamma) * v_r
    return u_gamma, v_gamma

def locate_depth_point(u_x, v_x, alphas, dl, pia, wm, valid):
    '''
    u_x, v_x: (N, 2) points in the flat space
    alpha: (N,) depth values in [0, 1]
    returns: (x, y, z) points in the 3D space
    '''
    
    simp = dl.find_simplex([v_x, u_x])
    tf = dl.transform[simp]
    l_a, l_b = tf[:2,:2] @ (np.array([v_x, u_x]) - tf[2])
    l_c = 1 - l_a - l_b
    #idx, bary = find_triangle(verts, faces, np.vstack([u_x, v_x]).T)
    x_p = [l_a, l_b, l_c] @ pia[valid][dl.simplices][simp]
    x_w = [l_a, l_b, l_c] @ wm[valid][dl.simplices][simp]
    #x_p = p_verts[faces[idx, 0]] * bary[:, 0, None] + p_verts[faces[idx, 1]] * bary[:, 1, None] + p_verts[faces[idx, 2]] * bary[:, 2, None]
    #x_w = w_verts[faces[idx, 0]] * bary[:, 0, None] + w_verts[faces[idx, 1]] * bary[:, 1, None] + w_verts[faces[idx, 2]] * bary[:, 2, None]
    #Compute area of pial and white matter triangles
    #A_p = 0.5 * np.linalg.norm(np.cross(p_verts[faces[idx, 1]] - p_verts[faces[idx, 0]], p_verts[faces[idx, 2]] - p_verts[faces[idx, 0]]), axis=1)
    #A_w = 0.5 * np.linalg.norm(np.cross(w_verts[faces[idx, 1]] - w_verts[faces[idx, 0]], w_verts[faces[idx, 2]] - w_verts[faces[idx, 0]]), axis=1)
    A_p = 0.5 * np.linalg.norm(np.cross(pia[valid][dl.simplices][simp][1] - pia[valid][dl.simplices][simp][0],
                                    pia[valid][dl.simplices][simp][2] - pia[valid][dl.simplices][simp][0]))
    A_w = 0.5 * np.linalg.norm(np.cross(wm[valid][dl.simplices][simp][1] - wm[valid][dl.simplices][simp][0],
                                        wm[valid][dl.simplices][simp][2] - wm[valid][dl.simplices][simp][0]))

    beta = 1-(1/(A_p - A_w) *(- A_w + np.sqrt((1 - alphas)*A_p**2 + alphas*A_w**2)))
    z = (1-beta[:, np.newaxis])*x_p + beta[:, np.newaxis]*x_w
    return z

def make_laminar_profile(subject, xfmname, u_l, v_l, u_r, v_r, W, H, sampler="nearest"):
    """
    Given a line in flatmap space with endpoints (L,R) and dimensions (W,H), create a cortical profile map that is W pixels wide and H pixels high.
    Along the width, gamma will vary from 0 to 1. Along the height, alpha will vary from 0 to 1.
    For each (gamma,alpha), compute (z), find the enclosing voxel, and then retrieve the value at that voxel. Plot.
    """

    xfm = cortex.db.get_xfm(subject, xfmname, xfmtype="coord")
    sampclass = getattr(samplers, sampler)
    # Create a grid of gamma and alpha values
    gamma_values = np.linspace(0, 1, W)
    alpha_values = np.linspace(0, 1, H)
    
    # Initialize an empty array to hold the cortical profile values
    profile_map = np.zeros((H, W), int)

    verts, faces = cortex.db.get_surf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(faces) # find vertices in flatmap triangles
    dl = Delaunay(verts[valid,:2])

    pia = xfm(cortex.db.get_surf(subject, "pia", merge=True, nudge=False)[0])
    wm = xfm(cortex.db.get_surf(subject, "wm", merge=True, nudge=False)[0])
    
    for j, gamma in enumerate(gamma_values):
        print(j)
        # Interpolate the flatmap coordinates
        u_gamma, v_gamma = line_interpolation(u_l, v_l, u_r, v_r, gamma)
        # Locate the depth point in 3D space (H x 3)
        # 
        zs = locate_depth_point(u_gamma, v_gamma, alpha_values, dl, pia, wm, valid)
        # Retrieve the value at that voxel (assuming a function get_voxel_value exists)
        _, voxs, _ = sampclass(zs, xfm.shape)
        profile_map[:, j] = voxs
            
    return profile_map
