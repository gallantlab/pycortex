import numpy as np

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
