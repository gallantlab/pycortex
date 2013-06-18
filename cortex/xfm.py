import os
import numpy as np
import nibabel

class Transform(object):
    '''A standard affine transform. Typically holds a transform from anatomical fiducial space to epi magnet space.
    '''
    def __init__(self, xfm, epifile):
        self.xfm = xfm
        self.epi = epifile
        if isinstance(epifile, (str, unicode)):
            self.epi = nibabel.load(epifile)
        self.shape = self.epi.get_shape()[:3][::-1]

    def __call__(self, pts):
        return np.dot(self.xfm, np.hstack([pts, np.ones((len(pts),1))]).T)[:3].T

    def __mul__(self, other):
        assert other.shape == (4,4)
        return Transform(np.dot(self.xfm, other), self.epi)

    def __rmul__(self, other):
        assert other.shape == (4,4)
        return Transform(np.dot(other, self.xfm), self.epi)

    def __repr__(self):
        path, fname = os.path.split(self.epi.get_filename())
        return "<Transform into %s space>"%fname

    def save(self, subject, name, xfmtype="magnet"):
        from .db import surfs
        surfs.loadXfm(subject, name, self.xfm, xfmtype=xfmtype, epifile=self.epi.get_filename())

    @property
    def inv(self):
        return Transform(np.linalg.inv(self.xfm), self.epi)

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
        # Assure that both determinants are negative, i.e. that both spaces are FLIPPED (??)
        if npl.det(in_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        M = raw.get_affine()
        inv = np.linalg.inv
        coord = np.dot(inv(inspace), np.dot(inv(xfm), np.dot(refspace, inv(M))))
        return cls(coord, epi)

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

def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr