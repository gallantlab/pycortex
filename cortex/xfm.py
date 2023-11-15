"""Affine transformation class
"""
import os
import numpy as np
import subprocess

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
        if isinstance(xfm, str):
            with open(xfm, 'r') as fid:
                L = fid.readlines()
            xfm  = np.array([[np.float_(s) for s in ll.split() if s] for ll in L])

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
        in_hdr = inIm.header
        ref_hdr = refIm.header
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

        inAffine = inIm.affine

        coord = np.dot(inv(refspace),np.dot(xfm,np.dot(inspace,inv(inAffine))))
        return cls(coord, refIm)

    def to_fsl(self, anat_nii, direction='func>anat'):
        """Converts a pycortex transform to an FSL transform.

        Uses the stored "reference" file provided when the transform was created (usually
        a functional data or statistical volume) and the supplied anatomical file to
        create an FSL transform. By default, returns the transform FROM the reference volume
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
        in_hdr = inIm.header
        ref_hdr = self.reference.header
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

        inAffine = inIm.affine

        fslx = np.dot(refspace,np.dot(self.xfm,np.dot(inAffine,inv(inspace))))
        if direction=='func>anat':
            return inv(fslx)
        elif direction=='anat>func':
            return fslx

    @classmethod
    def from_freesurfer(cls, fs_register, func_nii, subject, freesurfer_subject_dir=None):
        """Converts a FreeSurfer transform to a pycortex transform.

        Converts a transform computed using FreeSurfer alignment tools (e.g., bbregister) to
        a transform ("xfm") object in pycortex. The transform must have been computed
        FROM the nifti volume specified in `func_nii` TO the anatomical volume of
        the FreeSurfer subject `subject` (See Notes below).

        Parameters
        ----------
        fs_register : array
            4x4 transformation matrix, described in an FreeSurfer .dat or .lta file, for a transform computed
            FROM the func_nii volume TO the anatomical volume of the FreeSurfer subject `subject`.
            Alternatively, a string file name for the FreeSurfer .dat or .lta file.
        func_nii : str or nibabel.Nifti1Image
            nibabel image object (or string path to nibabel-readable image) for (functional) data volume
            to be projected onto cortical surface
        subject : str
            FreeSurfer subject name for which the anatomical volume was registered for.
        freesurfer_subject_dir : str | None
            Directory of FreeSurfer subjects. Defaults to the value for
            the environment variable 'SUBJECTS_DIR' (which should be set
            by freesurfer)

        Returns
        -------
        xfm : cortex.xfm.Transform object
            A pycortex COORD transform.

        Notes
        -----
        The transform is assumed to be computed FROM the functional data TO the anatomical data of
        the specified FreeSurfer subject. In FreeSurfer speak, that means that the arguments to
        FreeSurfer alignment tools should have been:
        bbregister --s <subject> --mov <func_nii> --reg <fs_register> ...

        """
        import subprocess
        import nibabel
        import numpy.linalg as npl
        inv = npl.inv

        # Load anatomical to functional transform from register.dat file, if string is provided
        if isinstance(fs_register, str):
            with open(fs_register, 'r') as fid:
                L = fid.readlines()
            anat2func = np.array([[np.float_(s) for s in ll.split() if s] for ll in L[4:8]])
        else:
            anat2func = fs_register

        # Set FreeSurfer subject directory
        if freesurfer_subject_dir is None:
            freesurfer_subject_dir = os.environ['SUBJECTS_DIR']

        # Set path to the anatomical volume used to compute fs_register
        anat_mgz = os.path.join(freesurfer_subject_dir, subject, 'mri', 'orig.mgz')

        # Read vox2ras transform for the anatomical volume
        try:
            cmd = ('mri_info', '--vox2ras', anat_mgz)
            L = decode(subprocess.check_output(cmd)).splitlines()
            anat_vox2ras = np.array([[np.float_(s) for s in ll.split() if s] for ll in L])
        except OSError:
            print ("Error occurred while executing:\n{}".format(' '.join(cmd)))
            raise

        # Read tkrvox2ras transform for the anatomical volume
        anat_tkrvox2ras = _vox2ras_tkr(anat_mgz)

        # Read tkvox2ras transform for the functional volume
        func_tkrvox2ras = _vox2ras_tkr(func_nii)

        # Calculate pycorex transform (i.e. scanner to functional transform)
        coord = np.dot(inv(func_tkrvox2ras), np.dot(anat2func, np.dot(anat_tkrvox2ras, inv(anat_vox2ras))))

        try:
            refIm = nibabel.load(func_nii)
        except AttributeError:
            refIm = func_nii

        return cls(coord, refIm)


    def to_freesurfer(self, fs_register, subject, freesurfer_subject_dir=None):
        """Converts a pycortex transform to a FreeSurfer transform.

        Converts a transform stored in pycortex xfm object to the FreeSurfer format
        (i.e., register.dat format: https://surfer.nmr.mgh.harvard.edu/fswiki/RegisterDat)

        Parameters
        ----------
        fs_register : str
            Output path for the FreeSurfer formatted transform to be output.
        subject : str
            FreeSurfer subject name from which the pycortex subject was imported

        freesurfer_subject_dir : str | None
            Directory of FreeSurfer subjects. If None, defaults to the value for
            the environment variable 'SUBJECTS_DIR' (which should be set
            by freesurfer)

        """
        import tempfile
        import subprocess
        import nibabel
        import numpy.linalg as npl
        from .database import db
        inv = npl.inv

        # Set path to the anatomical volume for the FreeSurfer subject
        anat = db.get_anat(subject, type='raw')

        # Read vox2ras transform for the anatomical volume
        anat_vox2ras = anat.affine

        # Read tkrvox2ras transform for the  anatomical volume
        anat_tkrvox2ras = _vox2ras_tkr(anat.get_filename())

        # Read tkvox2ras transform for the functional volume
        func_tkrvox2ras = _vox2ras_tkr(self.reference.get_filename())

        # Read voxel resolution of the functional volume
        func_voxres = self.reference.header.get_zooms()

        # Calculate FreeSurfer transform
        fs_anat2func = np.dot(func_tkrvox2ras, np.dot(self.xfm, np.dot(anat_vox2ras, inv(anat_tkrvox2ras))))

        # Write out to `fs_register` in register.dat format
        with open(fs_register, 'w') as fid:
            fid.write('{}\n'.format(subject))
            fid.write('{:.6f}\n'.format(func_voxres[0]))
            fid.write('{:.6f}\n'.format(func_voxres[1]))
            fid.write('0.150000\n')
            for row in fs_anat2func:
                fid.write(' '.join(['{:.15e}'.format(x) for x in row]) + '\n')
        print('Wrote:')
        subprocess.call(('cat', fs_register))

        return fs_anat2func

def isstr(obj):
    """Check for stringy-ness in python 2.7 or 3"""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)
    
def decode(obj):
    if isinstance(obj, bytes):
        obj = obj.decode()
    return obj
        

def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr


def _vox2ras_tkr(image):
    """Run `mri_info --vox2ras-tkr` on `image` and return a numpy array with the
    output affine"""
    try:
        cmd = ('mri_info', '--vox2ras-tkr', image)
        L = decode(subprocess.check_output(cmd)).splitlines()
        # Skip headers/additional information. Example output of
        # mri_info --vox2ras-tkr
        #
        # niiRead(): NIFTI_UNITS_UNKNOWN, assuming mm
        #   -2.61900    0.00000    0.00000   81.18900
        #    0.00000    0.00000    2.60000  -63.70000
        #    0.00000   -2.61900    0.00000   87.73650
        #    0.00000    0.00000    0.00000    1.00000
        #
        # Just take the last 4 lines because the length of the extra info is
        # unpredictable.
        L = L[-4:]
        tkrvox2ras = np.array(
            [[np.float_(s) for s in ll.split() if s] for ll in L])
    except OSError as e:
        print("Error occurred while executing:\n{}".format(' '.join(cmd)))
        raise e
    return tkrvox2ras
