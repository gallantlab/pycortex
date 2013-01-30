import os

import nibabel
import numpy as np
from scipy import sparse

import polyutils
from db import surfs

class Projection(object):
	'''Projects data from epi volume onto surface using various projection methods'''
	def __init__(self, subject, xfmname, **kwargs):
		ptype = self.__class__.__name__.lower()
		fnames = surfs.getFiles(subject)
		xfm, epifile = surfs.getXfm(subject, xfmname)
		nib = nibabel.load(epifile)
		self.shape = nib.get_shape()[:3][::-1]
		self.xfmfile = fnames['xfms'].format(xfmname=xfmname)
		self.cachefile = fnames['projcache'].format(xfmname=xfmname, projection=ptype)

		try:
			npz = np.load(self.cachefile)
			if npz['mtime'] != os.stat(self.xfmfile).st_mtime:
				raise IOError
			self.mask_l = npz['left']
			self.mask_r = npz['right']
		except IOError:
			self._recache(subject, xfmname, **kwargs)

	@property
	def mask(self):
		mask = np.array(self.mask_l.sum(0) + self.mask_r.sum(0))
		return (mask.squeeze() != 0).reshape(*self.shape)

	def __call__(self, data):
		projected = []
		for mask in [self.mask_l, self.mask_r]:
			if data.ndim in (1, 2):
				#pre-masked data
				if data.ndim == 1:
					assert len(data) == self.mask.sum(), 'Invalid mask size'
					shape = (np.prod(self.shape), 1)
				else:
					assert data.shape[1] == self.mask.sum(), 'Invalid mask size'
					shape = (np.prod(self.shape), data.shape[0])
				normalized = sparse.csc_matrix(shape)
				normalized[self.mask.ravel()] = data.T
			elif data.ndim == 3:
				normalized = data.ravel()
			elif data.ndim == 4:
				normalized = data.reshape(len(data), -1).T
			else:
				raise ValueError

			projected.append(np.array(self.mask * normalized).T.squeeze())

		return projected

	def _recache(self, left, right):
		self.mask_l = left
		self.mask_r = right
		np.savez(self.cachefile, mtime=os.stat(self.xfmfile).st_mtime, left=left, right=right)

class Nearest(Projection):
	def _recache(self, subject, xfmname):
		masks = []
		for hemi in surfs.getCoords(subject, xfmname):
			mask = sparse.csr_matrix((len(hemi), np.prod(self.shape)), dtype=bool)
			ravelidx = np.ravel_multi_index(hemi.T[::-1], self.shape)
			for i, idx in enumerate(ravelidx):
				mask[i, idx] = True
			masks.append(mask)
		super(Nearest, self)._recache(mask[0], mask[1])

class Trilinear(Projection):
	def _recache(self, subject, xfmname):
		raise NotImplementedError

class Gaussian(Projection):
	def _recache(self, subject, xfmname, std=2):
		raise NotImplementedError

class GaussianThickness(Projection):
	def _recache(self, subject, xfmname, std=2):
		raise NotImplementedError

class Polyhedral(Projection):
	def _recache(self, subject, xfmname):
		from tvtk.api import tvtk

		pia = surfs.getVTK(subject, "pia")
		wm = surfs.getVTK(subject, "whitematter")
		flat = surfs.getVTK(subject, "flat")

		coord, epifile = surfs.getXfm(subject, xfmname, xfmtype='coord')
				
		#All necessary tvtk objects for measuring intersections
		poly = tvtk.PolyData()
		voxel = tvtk.CubeSource()
		trivox = tvtk.TriangleFilter()
		trivox.set_input(voxel.get_output())
		measure = tvtk.MassProperties()
		bop = tvtk.BooleanOperationPolyDataFilter()
		
		masks = []
		for (wpts, _, _), (ppts, _, _), (_, polys, _) in zip(pia, wm, flat):
			#iterate over hemispheres
			mask = sparse.csr_matrix((len(wpts), np.prod(self.shape)))

			surf = polyutils.Surface(polyutils.transform(coord, ppts), polys)
			for i, (pts, polys) in enumerate(surf.polyhedra(polyutils.transform(coord, wpts))):
				if len(pt) > 0:
					poly.set(points=pts, polys=polys)
					measure.set_input(poly)
					measure.update()
					totalvol = measure.volume
					bop.set_input(0, poly)

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
							idx = np.ravel_multi_index(vox[::-1], self.shape)
							mask[i, idx] = measure.volume / totalvol

				if i % 100 == 0:
					print i

			masks.append(mask)
		super(Polyhedral, self)._recache(masks[0], masks[1])

