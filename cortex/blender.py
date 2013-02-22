import numpy as np
from matplotlib import cm, colors

import bpy.ops
from bpy import context as C
from bpy import data as D

class Hemi(object):
	def __init__(self, pts, polys, curv=None, name='hemi'):
		self.mesh = D.meshes.new(name)
		self.mesh.from_pydata(pts.tolist(), [], polys.tolist())
		self._loopidx = np.zeros((len(self.mesh.loops),), dtype=np.uint32)
		self.mesh.loops.foreach_get('vertex_index', self._loopidx)
		self.obj = D.objects.new(name, self.mesh)
		self.obj.scale = .1, .1, .1
		C.scene.objects.link(self.obj)
		C.scene.objects.active = self.obj
		#Add basis shape
		bpy.ops.object.shape_key_add()
		self.addVColor(curv, name='curvature',vmin=-1, vmax=1)
	
	def addVColor(self, color, name='color', cmap=cm.RdBu_r, vmin=None, vmax=None):
		if color.ndim == 1:
			if vmin is None:
				vmin = color.min()
			if vmax is None:
				vmax = color.max()
			color = cmap((color - vmin) / (vmax - vmin))[:,:3]

		vcolor = self.mesh.vertex_colors.new(name)
		for i, j in enumerate(self._loopidx):
			vcolor.data[i].color = list(color[j])

	def addShape(self, shape, name=None):
		C.scene.objects.active = self.obj
		self.obj.select = True
		bpy.ops.object.shape_key_add()
		key = D.shape_keys[-1].key_blocks[-1]
		if name is not None:
			key.name = name

		for i in range(len(key.data)):
			key.data[i].co = shape[i]
		return key

def show(data, subject, xfmname, types=('inflated',)):
	from .db import surfs
	surfs.getVTK(data, "fiducial")

def flatten(subjfs, retinotopy):
	pass