import numpy as np
from cortex import polyutils

## Fuck this test is annoying
# def test_cube():
# 	from mayavi import mlab
# 	pts, polys = polyutils.make_cube((.5, .5, .5), 1)
# 	mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)
# 	assert True

def test_surfpatch():
    from cortex import db
    wm, polys = db.get_surf("S1", "wm", "lh")
    pia, _ = db.get_surf("S1", "pia", "lh")
    surf = polyutils.Surface(wm, polys)
    subwm, subpia, subpolys = surf.extract_chunk(auxpts=pia)
    subsurf = polyutils.Surface(subwm, subpolys)
    return [patch for patch in subsurf.patches(n=0.5)]
