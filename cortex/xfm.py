import os
import numpy as np

class Transform(object):
    '''
    A standard affine transform. Typically holds a transform from anatomical 
    magnet space to epi file space.
    '''
    def __init__(self, xfm, reference):
        self.xfm = xfm
        self.reference = None
        if isinstance(reference, str):
            import nibabel
            try:
                self.reference = nibabel.load(reference)
                self.shape = self.reference.shape[:3][::-1]
            except IOError:
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
        if isinstance(other, Transform):
            other = other.xfm
        return Transform(np.dot(self.xfm, other), ref)

    def __rmul__(self, other):
        ref = self.reference
        if ref is None:
            ref = self.shape
        if isinstance(other, Transform):
            other = other.xfm
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
    def from_fsl(cls, xfm, infile, reffile):
        """
        Converts fsl transform xfm (estimated FROM infile TO reffile) 
        to a pycortex COORD transform. 
        """
        ## Adapted from dipy.external.fsl.flirt2aff#############################
        import nibabel
        import numpy.linalg as npl
        
        try:
            inIm = nibabel.load(infile)
        except AttributeError:
            inIm = infile
        
        refIm = nibabel.load(reffile)
        in_hdr = inIm.get_header()
        ref_hdr = refIm.get_header()
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        # Since FSL does not use the full transform info in the nifti header, 
        # determine whether the transform indicates that the X axis should be 
        # flipped; if so, flip the X axis (for both infile and reffile)
        if npl.det(in_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        inAffine = inIm.get_affine()
        inv = np.linalg.inv
        coord = np.dot(inv(refspace),np.dot(xfm,np.dot(inspace,inv(inAffine))))
        return cls(coord, refIm)

    def to_fsl(self, infile):
        """
        Converts a pycortex transform to an FSL transform.
        The resulting FSL transform goes FROM the space of the "infile" input
        TO the space of the reference nifti stored in the pycortex transform.

        This should ONLY be used for "coord" transforms! Will fail hard for 
        "magnet" transforms!
        """
        import nibabel
        import numpy.linalg as npl

        try:
            inIm = nibabel.load(infile)
        except AttributeError:
            inIm = infile
        in_hdr = inIm.get_header()
        ref_hdr = self.reference.get_header()
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        # Since FSL does not use the full transform info in the nifti header, 
        # determine whether the transform indicates that the X axis should be 
        # flipped; if so, flip the X axis (for both infile and reffile)
        if npl.det(in_hdr.get_best_affine())>=0:
            print("Determinant is > 0: FLIPPING!")
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            print("Determinant is > 0: FLIPPING!")
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

        inAffine = inIm.get_affine()
        inv = np.linalg.inv
        fslx = np.dot(refspace,np.dot(self.xfm,np.dot(inAffine,inv(inspace))))
        return fslx

def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr