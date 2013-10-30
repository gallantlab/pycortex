import os
import numpy as np

class Transform(object):
    '''A standard affine transform. Typically holds a transform from anatomical fiducial space to epi magnet space.
    '''
    def __init__(self, xfm, reference):
        self.xfm = xfm
        self.reference = None
        if isinstance(reference, str):
            try:
                import nibabel
                self.reference = nibabel.load(reference)
                self.shape = self.reference.shape[:3][::-1]
            except:
                self.reference = reference
        elif isinstance(reference, tuple):
            self.shape = reference
        else:
            self.reference = reference
            self.shape = self.reference.shape[:3][::-1]

    def __call__(self, pts):
        return np.dot(self.xfm, np.hstack([pts, np.ones((len(pts),1))]).T)[:3].T

    @property
    def inv(self):
        ref = self.reference
        if ref is None:
            ref = self.shape
        return Transform(np.linalg.inv(self.xfm), ref)

    def __mul__(self, other):
        ref = self.reference
        if ref is None:
            ref = self.shape
        return Transform(np.dot(self.xfm, other), ref)

    def __rmul__(self, other):
        ref = self.reference
        if ref is None:
            ref = self.shape
        return Transform(np.dot(other, self.xfm), ref)

    def __repr__(self):
        try:
            path, fname = os.path.split(self.reference.get_filename())
            return "<Transform into %s space>"%fname
        except AttributeError:
            return "<Reference free affine transform>"

    def save(self, subject, name, xfmtype="magnet"):
        if self.reference is None:
            raise ValueError('Cannot save reference-free transforms into the database')
        from .db import surfs
        surfs.loadXfm(subject, name, self.xfm, xfmtype=xfmtype, reference=self.reference.get_filename())

    @classmethod
    def from_fsl(cls, xfm, basefile, reffile):
        """
        takes transform xfm (estimated FROM basefile TO reffile) and converts to GLab COORDINATE transform
        """
        ## Adapted from dipy.external.fsl.flirt2aff#############################
        import nibabel
        import numpy.linalg as npl
        
        baseIm = nibabel.load(basefile)
        refIm = nibabel.load(reffile)
        base_hdr = baseIm.get_header()
        ref_hdr = refIm.get_header()
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(base_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        # Assure that both determinants are negative, i.e. that both spaces are FLIPPED (??)
        if npl.det(base_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(base_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        M = refIm.get_affine()
        inv = np.linalg.inv
        coord = np.dot(M,np.dot(inv(refspace),np.dot(xfm,inspace)))
        # This works as well (demonstration of different path to same transform)
        #coord = np.dot(inv(inspace), np.dot(inv(xfm), np.dot(refspace, inv(M))))
        #coord = inv(coord)
        return cls(coord, refIm)

    def to_fsl(self, basefile):
        """
        Converts a Glab transform to an FSL transform.
        The resulting FSL transform goes FROM the space of the "basefile" input
        TO the space of the reference nifti stored in the GLab transform.
        """
        import nibabel
        import numpy.linalg as npl

        baseIm = nibabel.load(basefile)
        in_hdr = baseIm.get_header()
        ref_hdr = self.reference.get_header()
        
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))

        if npl.det(in_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        M = self.reference.get_affine()
        inv = np.linalg.inv
        fslx = np.dot(refspace,np.dot(inv(M),np.dot(self.xfm,inv(inspace))))
        #fslx = inv(np.dot(inspace, np.dot(inv(self.xfm), np.dot(M, inv(refspace)))))
        return fslx


def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr