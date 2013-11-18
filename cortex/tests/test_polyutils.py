import numpy as np
from cortex import polyutils

def test_cube():
	from mayavi import mlab
	pts, polys = polyutils.make_cube((.5, .5, .5), 1)
	mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)
	assert True

def test_surfpatch():
    from cortex import surfs
    wm, polys = surfs.getSurf("S1", "wm", "lh")
    pia, _ = surfs.getSurf("S1", "pia", "lh")
    surf = polyutils.Surface(wm, polys)
    subwm, subpia, subpolys = surf.extract_chunk(auxpts=pia)
    subsurf = polyutils.Surface(subwm, subpolys)
    return [patch for patch in subsurf.patches(n=0.5)]
