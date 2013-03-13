import numpy as np
import nibabel

class Transform(object):
	def __init__(self, xfm, epifile):
		self.xfm = xfm
		self.epi = nibabel.load(epifile)
		self.shape = self.epi.get_shape()

	def __call__(self, pts):
		return np.dot(xfm, np.hstack([pts, np.ones((len(pts),1))]).T)[:3].T

	@classmethod
	def from_fsl(cls, xfm, epifile, rawfile):
	    ## Adapted from dipy.external.fsl.flirt2aff#############################
	    import numpy.linalg as npl
	    
	    epi = nibabel.load(epifile)
	    raw = nibabel.load(rawfile)
	    in_hdr = epi.get_header()
	    ref_hdr = raw.get_header()
	    
	    # get_zooms gets the positive voxel sizes as returned in the header
	    inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
	    refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
	    
	    if npl.det(in_hdr.get_best_affine())>=0:
	        inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
	    if npl.det(ref_hdr.get_best_affine())>=0:
	        refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

	    epi = nibabel.load(epifile).get_header().get_base_affine()
	    M = nibabel.load(raw).get_affine()
	    inv = np.linalg.inv

	    coord = np.dot(inv(inspace), np.dot(inv(xfm), np.dot(refspace, inv(M))))
	    return cls(coord, epifile)

	def to_fsl(self, rawfile):
		import numpy.linalg as npl

        raw = nibabel.load(rawfile)
        in_hdr = self.epi.get_header()
        ref_hdr = raw.get_header()
        
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        
        if npl.det(in_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        M = raw.get_affine()
        inv = np.linalg.inv

        fslx = inv(np.dot(inspace, np.dot(self.xfm, np.dot(M, inv(refspace)))))
        return fslx

    def save(self, subject, name, type):
    	from . import db


def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr