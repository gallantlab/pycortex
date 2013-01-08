import tempfile
import numpy as np

from db import surfs
from utils import get_cortical_mask
from openctm import CTMfile

class BrainCTM(object):
	def __init__(self, subject, xfm):
		self.name = subject
		self.xfmname = xfm
		self.files = surfs.getFiles(subject)
		self.mask = get_cortical_mask(self.name, self.xfmname)
		self.coords = surfs.getCoords(self.name, self.xfmname)
		self.curvs = np.load(surfs.getAnat(self.name, type='curvature'))

		self.left = Hemi()
		self.right = Hemi()

		left, right = surfs.getVTK(subject, "fiducial")
		merge = np.vstack([left, right])
		self.

	def addSurf(self, typename):
		left, right = surfs.getVTK(self.name, typename, nudge=True)
		self.left.addSurf(left[0])
		self.right.addSurf(right[0])

class Hemi(object):
	def __init__(self):
		self.nsurfs = 0
		self.tf = tempfile.NamedTemporaryFile()
		self.ctm = CTMfile(tf.name, "w")

	def addSurf(self, pts):
		self.ctm.addAttrib(pts, 'morphTarget%d'%self.nsurfs)
		self.nsurfs += 1

	def setMesh(self, pts, polys):
		pass

def _norm_pts(pts, minmax):
