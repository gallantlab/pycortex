import numpy as np
from .. import mapper
from .. import polyutils

def test_nearest():
	pts, polys = polyutils.make_cube(size=.5)
	mask = mapper.Nearest._getmask(pts, polys, (2, 2, 2))
	mfunc = mapper.Nearest(mask, mask, (2, 2, 2))
	data = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
	assert np.allclose(mfunc(data)[0], np.arange(1, 9))

def test_trilin():
	pts, polys = polyutils.make_cube(size=.5)
	mask = mapper.Trilinear._getmask(pts, polys, (2, 2, 2))
	mfunc = mapper.Trilinear(mask, mask, (2, 2, 2))
	data = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
	expected = np.array([ 2.75,  3.25,  3.75,  4.25,  4.75,  5.25,  5.75,  6.25])
	assert np.allclose(mfunc(data), expected)

