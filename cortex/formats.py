import numpy as np

def read_off(filename):
	pts, polys = [], []
	with open(filename) as fp:
		assert fp.readline()[:3] == 'OFF', 'Not an OFF file'
		npts, nface, nedge = map(int, fp.readline().split())
		print(npts, nface)
		for i in range(npts):
			pts.append([float(p) for p in fp.readline().split()])

		for i in range(nface):
			polys.append([int(i) for i in fp.readline().split()][1:])

	return np.array(pts), np.array(polys)