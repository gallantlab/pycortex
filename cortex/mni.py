import tempfile
import numpy as np
import subprocess
import nibabel

import cortex
from cortex.options import config

fslprefix = config.get("basic", "fsl_prefix")

def _save_fsl_xfm(filename, xfm):
    np.savetxt(filename, xfm, "%0.10f")

def _load_fsl_xfm(filename):
    return np.loadtxt(filename)

def compute_mni_transform(subject, xfm,
                          template="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"):
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
    anat_filename = cortex.db.get_anat(subject, "brainmask").get_filename()
    
    # First use flirt to align masked subject anatomical to MNI template
    subprocess.call(["{fslprefix}flirt".format(fslprefix=fslprefix),
                     "-searchrx -180 180",
                     "-searchry -180 180",
                     "-searchrz -180 180",
                     "-ref", template,
                     "-in", anat_filename,
                     "-omat", anat_to_mni_xfm])

    # Then load that transform and concatenate it with the functional to anatomical transform
    anat_to_mni = np.loadtxt(anat_to_mni_xfm)
    func_to_anat = cortex.db.get_xfm(subject, xfm).to_fsl(anat_filename)
    
    func_to_mni = np.dot(anat_to_mni, func_to_anat)

    return func_to_mni

def transform_to_mni(volumedata, func_to_mni, 
                     template="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"):
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

def transform_mni_to_subject(subject, xfm, volarray, func_to_mni,
                             template="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"):
    """Transform data in `volarray` from MNI space to functional space specified by `xfm`.

    Parameters
    ----------
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
    ref_filename = cortex.db.get_xfm(subject, xfm).reference.get_filename()
    
    subprocess.call(["{fslprefix}flirt".format(fslprefix=fslprefix),
                     "-in", mnispace_func_nii,
                     "-ref", ref_filename,
                     "-applyxfm", "-init", mni_to_func_xfm,
                     "-out", funcspace_nii])

    return nibabel.load(funcspace_nii)
    
