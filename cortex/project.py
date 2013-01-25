import numpy as np
import nibabel

import polyutils
from db import surfs

def nearest(subject, xfmname):
	pass

def polyhedra(subject, xfmname):
	from tvtk.api import tvtk

	pia = surfs.getVTK(subject, "pia")
	wm = surfs.getVTK(subject, "whitematter")
	flat = surfs.getVTK(subject, "flat")

	coord, epifile = surfs.getXfm(subject, xfmname)
	nib = nibabel.load(epifile)
	shape = nib.get_shape()
	
	#All necessary tvtk objects for measuring intersections
	voxel = tvtk.CubeSource()
	trivox = tvtk.TriangleFilter()
	trivox.set_input(voxel.get_output())
	measure = tvtk.MassProperties()
	poly = tvtk.PolyData()
	bop = tvtk.BooleanOperationPolyDataFilter()

	masks = []
	for (wpts, _, _), (ppts, _, _), (_, polys, _) in zip(pia, wm, flat):
		mask = sparse.csr_matrix((len(wpts), np.prod(shape)))

		surf = polyutils.Surface(polyutils.transform(coord, ppts), polys)
		for i, (pts, polys) in enumerate(surf.polyhedra(polyutils.transform(coord, wpts))):
			if len(pt) > 0:
				poly.set(points=pts, polys=polys)
				measure.set_input(poly)
				measure.update()
				totalvol = measure.volume
				bop.set_input(0, poly)

				pvox, pweight = [], []

				bmin = pt.min(0).round()
				bmax = pt.max(0).round() + 1
				vidx = np.mgrid[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
				for vox in vidx.reshape(3, -1).T:
					voxel.center = vox
					voxel.update()
					trivox.update()
					bop.set_input(1, trivox.get_output())
					bop.update()
					measure.set_input(bop.get_output())
					measure.update()
					if measure.volume > 1e-6:
						pvox.append(vox)
						pweight.append(measure.volume / totalvol)

				if i % 100 == 0:
					print i

		masks.append(mask)

