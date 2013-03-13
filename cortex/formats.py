import os
import glob
import numpy as np

def read(globname):
	preference = ['npz', 'vtk', 'off']

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

def read_npz(filename):
	npz = np.load(filename)
	return npz['pts'], npz['polys']

def read_vtk(filename):
	with open(filename) as vtk:
		pts, polys = None, None
		line = vtk.readline()
		while len(line) > 0 and (pts is None or polys is None):
			if line.startswith("POINTS"):
				_, n, dtype = line.split()
				data = vtk.readline().split()
				n = int(n)
				nel = n*3
				while len(data) < nel:
					data += vtk.readline().split()
				pts = np.array(data, dtype=float).reshape(n, 3)
			elif line.startswith("POLYGONS"):
				_, n, nel = line.split()
				nel = int(nel)
				data = vtk.readline().split()
				while len(data) < nel:
					data += vtk.readline().split()
				polys = np.array(data, dtype=np.uint32).reshape(int(n), 4)[:,1:]

			line = vtk.readline()
		return pts, polys

def write_vtk(outfile, pts, polys, norms=None):
	with open(outfile, "w") as fp:
		fp.write("# vtk DataFile Version 3.0\nWritten by pycortex\nASCII\nDATASET POLYDATA\n")
		fp.write("POINTS %d float\n"%len(pts))
		np.savetxt(fp, pts, fmt='%0.12g')
		fp.write("\n")

		fp.write("POLYGONS %d %d\n"%(len(polys), 4*len(polys)))
		spolys = np.hstack((3*np.ones((len(polys),1), dtype=polys.dtype), polys))
		np.savetxt(fp, spolys, fmt='%d')
		fp.write("\n")

		if norms is not None and len(norms) == len(pts):
			fp.write("NORMALS Normals float")
			np.savetxt(fp, norms, fmt='%0.12g')
