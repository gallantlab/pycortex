import numpy as np
from .. import polyutils

def test_cube():
	from mayavi import mlab
	pts, polys = polyutils.make_cube((.5, .5, .5), 1)
	mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)
	assert True

def test_voxelize():
	pts, polys = polyutils.make_cube((2, 2, 2), 2)
	vox = polyutils.voxelize(pts, polys, shape=(4, 4, 4), center=(0,0,0), mp=False)
	target = np.array(
	  [[[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]], dtype=np.uint8)

	assert allclose(vox, target)