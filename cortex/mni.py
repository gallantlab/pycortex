import os
import nibabel
import tempfile
import subprocess
import numpy as np

from . import options
from . import db

import shlex

fslprefix = options.config.get("basic", "fsl_prefix")
fsldir = os.getenv("FSLDIR")
if fsldir is None:
    import warnings
    warnings.warn("Can't find FSLDIR environment variable, assuming default FSL location..")
    fsldir = "/usr/share/fsl/5.0"

default_template = os.path.join(fsldir, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")

def _save_fsl_xfm(filename, xfm):
    np.savetxt(filename, xfm, "%0.10f")

def _load_fsl_xfm(filename):
    return np.loadtxt(filename)

def compute_mni_transform(subject, xfm,
                          template=default_template):
    """Compute transform from the space specified by `xfm` to MNI standard space.

    Parameters
    ----------
    subject : str
        Subject identifier
    xfm : str
        Name of functional space transform. Can be 'identity' for anat space.
    template : str, optional
        Path to MNI template volume. Defaults to FSL's MNI152_T1_1mm_brain.

    Returns
    -------
    func_to_mni : numpy.ndarray
        Transformation matrix from the space specified by `xfm` to MNI space.
    """
    # Set up some paths
    anat_to_mni_xfm = tempfile.mktemp()

    # Get anatomical image
    anat_filename = db.get_anat(subject, "brainmask").get_filename()
    
    # First use flirt to align masked subject anatomical to MNI template
    cmd = shlex.split(" ".join(["{fslprefix}flirt".format(fslprefix=fslprefix),
                     "-searchrx -180 180",
                     "-searchry -180 180",
                     "-searchrz -180 180",
                     "-ref", template,
                     "-in", anat_filename,
                     "-omat", anat_to_mni_xfm]))
    
    subprocess.call(cmd)

    # Then load that transform and concatenate it with the functional to anatomical transform
    anat_to_mni = np.loadtxt(anat_to_mni_xfm)
    func_to_anat = db.get_xfm(subject, xfm).to_fsl(anat_filename)
    
    func_to_mni = np.dot(anat_to_mni, func_to_anat)

    return func_to_mni

def transform_to_mni(volumedata, func_to_mni, 
                     template=default_template,
                     use_flirt=True):
    """Transform data in `volumedata` to MNI space, resample at 1mm resolution.

    Parameters
    ----------
    volumedata : VolumeData
        Data to be transformed to MNI space.
    func_to_mni : numpy.ndarray
        Transformation matrix from the space of `volumedata` to MNI space. Get this
        from compute_mni_transform.
    template : str, optional
        Path to MNI template volume, used as reference for flirt. Defaults to FSL's 
        MNI152_T1_1mm_brain.

    Returns
    -------
    mni_volumedata : nibabel.nifti1.Nifti1Image
        `volumedata` after transformation to MNI space.
    """
    if use_flirt:
        # Set up paths
        func_nii = tempfile.mktemp(".nii.gz")
        func_to_mni_xfm = tempfile.mktemp(".mat")
        func_in_mni = tempfile.mktemp(".nii.gz")

        # Save out relevant things
        volumedata.save_nii(func_nii)
        _save_fsl_xfm(func_to_mni_xfm, func_to_mni)
        
        # Use flirt to resample functional data
        subprocess.call(["{fslprefix}flirt".format(fslprefix=fslprefix),
                         "-in", func_nii,
                         "-ref", template,
                         "-applyxfm", "-init", func_to_mni_xfm,
                         "-out", func_in_mni])

        return nibabel.load(func_in_mni)
    else:
        from nipy.algorithms.registration import resample
        from . import xfm
        func_xfm = db.get_xfm(volumedata.subject, volumedata.xfmname)
        #xfm = cortex.db.get_mnixfm("AHfs", "AHfs_auto1")
        affine = func_xfm.reference.get_affine()
        volumedata_nii = nibabel.Nifti1Image(volumedata.volume.T.squeeze(), affine)
        nof_xfm = xfm.Transform.from_fsl(func_to_mni, 
                                         func_xfm.reference.get_filename(), 
                                         template)
        resampled = resample(volumedata_nii, 
                             nof_xfm.xfm, 
                             reference=nibabel.load(template), 
                             mov_voxel_coords=True,
                             interp_order=1)
        return resampled


def transform_surface_to_mni(subject, surfname):
    """Transform the surface named `surfname` for subject called `subject` into
    MNI coordinates. Returns [(lpts, lpolys), (rpts, rpolys)].

    Parameters
    ----------
    subject : str
        Subject identifier
    surfname : str
        Surface identifier

    Returns
    -------
    mni_surface : [(ndarray, ndarray), (ndarray, ndarray)]
        MNI-transformed surface in same format returned by db.get_surf.
    """
    # Get MNI affine transform
    mni_affine = nibabel.load(default_template).get_affine()

    # Get subject anatomical-to-MNI transform
    mni_xfm = np.dot(mni_affine, db.get_mnixfm(subject, "identity"))

    # Get transform from surface points to anatomical space
    ident_xfm = db.get_xfm(subject, "identity", xfmtype="coord")

    # Get surfaces
    (lpts, lpolys), (rpts, rpolys) = db.get_surf(subject, surfname)

    # Transform surface points into anatomical space
    anat_lpts, anat_rpts = ident_xfm(lpts), ident_xfm(rpts)

    # Transform anatomical space points to MNI space
    mni_lpts, mni_rpts = [np.dot(mni_xfm, np.hstack([p, np.ones((p.shape[0],1))]).T).T[:,:3]
                          for p in anat_lpts, anat_rpts]

    return [(mni_lpts, lpolys), (mni_rpts, rpolys)]

def transform_mni_to_subject(subject, xfm, volarray, func_to_mni,
                             template=default_template):
    """Transform data in `volarray` from MNI space to functional space specified by `xfm`.

    Parameters
    ----------
    subject : str
        Subject identifier
    xfm : str
        Name of functional space that data will be transformed into.
    volarray : numpy.ndarray
        3D volume in MNI space (should have same size as `template`)
    func_to_mni : numpy.ndarray
        Transformation matrix from `xfm` space to MNI space. Get this
        from compute_mni_transform.
    template : str, optional
        Path to MNI template volume, used as reference. Defaults to FSL's 
        MNI152_T1_1mm_brain.

    Returns
    -------
    xfm_volumedata : nibabel.nifti1.Nifti1Image
        `volarray` after transformation from MNI space to space specified by `xfm`.

    """
    # Set up paths
    mnispace_func_nii = tempfile.mktemp(".nii.gz")
    mni_to_func_xfm = tempfile.mktemp(".mat")
    funcspace_nii = tempfile.mktemp(".nii.gz")

    # Save out relevant things
    affine = nibabel.load(template).get_affine()
    nibabel.save(nibabel.Nifti1Image(volarray, affine), mnispace_func_nii)
    _save_fsl_xfm(mni_to_func_xfm, np.linalg.inv(func_to_mni))

    # Use flirt to resample data to functional space
    ref_filename = db.get_xfm(subject, xfm).reference.get_filename()
    
    subprocess.call(["{fslprefix}flirt".format(fslprefix=fslprefix),
                     "-in", mnispace_func_nii,
                     "-ref", ref_filename,
                     "-applyxfm", "-init", mni_to_func_xfm,
                     "-out", funcspace_nii])

    return nibabel.load(funcspace_nii)
