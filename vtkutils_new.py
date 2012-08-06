import itertools
import numpy as np
#import vtkctm

def read_old(vtk):
	pts, polys = vtkctm.readVTK(vtk)
	return pts, polys, None

def read(vtk):
	with open(vtk) as vtk:
		pts, polys = None, None
		line = vtk.next()
		while len(line) > 0 and (pts is None or polys is None):
			if line.startswith("POINTS"):
				_, n, dtype = line.split()
				data = vtk.next().split()
				n = int(n)
				nel = n*3
				while len(data) < nel:
					data += vtk.next().split()
				pts = np.array(data, dtype=float).reshape(n, 3)
			elif line.startswith("POLYGONS"):
				_, n, nel = line.split()
				nel = int(nel)
				data = vtk.next().split()
				while len(data) < nel:
					data += vtk.next().split()
				polys = np.array(data, dtype=np.uint32).reshape(int(n), 4)[:,1:]
			line = vtk.next()
		return pts, polys, None

