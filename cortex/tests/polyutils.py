from .. import polyutils

def test_cube():
	from mayavi import mlab
	pts, polys = polyutils.make_cube((.5, .5, .5), 1)
	mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)
	assert True