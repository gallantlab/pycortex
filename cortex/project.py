import numpy as np
import nibabel

import polyutils
from db import surfs

def nearest(subject, xfmname):
	pass

def polyhedra(subject, xfmname):
	pia = surfs.getVTK(subject, "pia")
	wm = surfs.getVTK(subject, "whitematter")
	flat = surfs.getVTK(subject, "flat")

	coord, epifile = surfs.getXfm(subject, xfmname)
	nib = nibable.load(epifile)
	xfm = np.dot(nib.get_header().get_base_affine(), coord)

	for (wpts, _, _), (ppts, _, _), (_, polys, _) in zip(pia, wm, flat):
		surf = polyutils.Surface(polyutils.transform(xfm, ppts), polys)
		for pt, poly in surf.polyhedra(polyutils.transform(xfm, wpts)):
			if len(pt) > 0:
				bbox = pt.min(0), pt.max(0)
				