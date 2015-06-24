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

        if isstr(reference):
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
        from .database import db
        db.save_xfm(subject, name, self.xfm, xfmtype=xfmtype, reference=self.reference.get_filename())

    @classmethod
    def from_fsl(cls, xfm, func_nii, anat_nii):
        """Converts an fsl transform to a pycortex transform.

        Converts a transform computed using FSL's FLIRT to a transform ("xfm") object in pycortex.
        The transform must have been computed FROM the nifti volume specified in `func_nii` TO the 
        volume specified in `anat_nii` (See Notes below).

        Parameters
        ----------
        xfm : array
            4x4 transformation matrix, loaded from an FSL .mat file, for a transform computed 
            FROM the func_nii volume TO the anat_nii volume. Alternatively, a string file name
            for the FSL .mat file.
        anat_nii : str or nibabel.Nifti1Image
            nibabel image object (or path to nibabel-readable image) for anatomical volume from 
            which cortical surface was created
        func_nii : str or nibabel.Nifti1Image
            nibabel image object (or string path to nibabel-readable image) for (functional) data volume 
            to be projected onto cortical surface

        Returns
        -------
        xfm : cortex.xfm.Transform object
            A pycortex COORD transform. 

        Notes
        -----
        The transform is assumed to be computed FROM the functional data TO the anatomical data.
        In FSL speak, that means that the arguments to flirt should have been:
        flirt -in <func_nii> -ref <anat_nii> ...

        """
        ## -- Adapted from dipy.external.fsl.flirt2aff -- ##
        import nibabel
        import numpy.linalg as npl
        inv = npl.inv

        # Load transform from text file, if string is provided
        if isinstance(xfm,(str,unicode)):
            with open(xfm,'r') as fid:
                L = fid.readlines()
            xfm  = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

        # Internally, pycortex computes the OPPOSITE transform: from anatomical volume to functional volume. 
        # Thus, assign anat to "infile" (starting point for transform)
        infile = anat_nii
        # Assign func to "reffile" (end point for transform)
        reffile = func_nii
        # and invert the usual direction (change from func>anat to anat>func)
        xfm = inv(xfm)

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
        
        coord = np.dot(inv(refspace),np.dot(xfm,np.dot(inspace,inv(inAffine))))
        return cls(coord, refIm)

    def to_fsl(self, anat_nii, direction='func>anat'):
        """Converts a pycortex transform to an FSL transform.

        Uses the stored "reference" file provided when the transform was created (usually 
        a functional data or statistical volume) and the supplied anatomical file to 
        create an FSL transform. By default, returns the transform FROM the refernce volume
        (usually the functional data volume) to the anatomical volume (`anat_nii` input). 

        Parameters
        ----------
        anat_nii : str or nibabel.Nifti1Image
            nibabel image object (or path to nibabel-readable image) for anatomical volume from 
            which cortical surface was created

        direction : str, optional {'func>anat', 'anat>func'}
            Direction of transform to return. Defaults to 'func>anat'

        Notes
        -----
        This function will only work for "coord" transform objects, (those retrieved with 
        cortex.db.get_xfm(xfmtype='coord',...)). It will fail hard for "magnet" transforms!

        """
        import nibabel
        import numpy.linalg as npl
        inv = npl.inv
        ## -- Internal notes -- ## 
        # pycortex transforms are internally stored as anatomical space -> functional data space
        # transforms. Thus the anatomical file is the "infile" in FSL-speak.
        infile = anat_nii

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
        
        fslx = np.dot(refspace,np.dot(self.xfm,np.dot(inAffine,inv(inspace))))
        if direction=='func>anat':
            return inv(fslx)
        elif direction=='anat>func':
            return fslx

def isstr(obj):
    """Check for stringy-ness in python 2.7 or 3"""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr
