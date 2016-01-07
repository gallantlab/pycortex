import os
import numpy as np

def manual(subject, xfmname, reference=None, **kwargs):
    """Open GUI for manually aligning a functional volume to the cortical surface for `subject`. This
    creates a new transform called `xfm`. The name of a nibabel-readable file (e.g. nii) should be
    supplied as `reference`. This image will be copied into the database.

    To modify an existing functional-anatomical transform, `reference` can be left blank, and the
    previously used reference will be loaded.

    <<ADD DETAILS ABOUT TRANSFORMATION MATRIX FORMAT HERE>>

    When the GUI is closed, the transform will be saved into the pycortex database. The GUI requires
    Mayavi support.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        String identifying the transform to be created or loaded.
    reference : str, optional
        Path to a nibabel-readable image that will be used as the reference for this transform.
        If given the default value of None, this function will attempt to load an existing reference
        image from the database.
    kwargs : dict
        Passed to mayavi_aligner.get_aligner.

    Returns
    -------
    m : 2D ndarray, shape (4, 4)
        Transformation matrix.
    """
    from .database import db
    from .mayavi_aligner import get_aligner
    def save_callback(aligner):
        db.save_xfm(subject, xfmname, aligner.get_xfm("magnet"), xfmtype='magnet', reference=reference)
        print("saved xfm")

    def view_callback(aligner):
        print('view-only mode! ignoring changes')

    # Check whether transform w/ this xfmname already exists
    view_only_mode = False
    try:
        db.get_xfm(subject, xfmname)
        # Transform exists, make sure that reference is None
        if reference is not None:
            raise ValueError('Refusing to overwrite reference for existing transform %s, use reference=None to load stored reference' % xfmname)

        # if masks have been cached, quit! user must remove them by hand
        from glob import glob
        if len(glob(db.get_paths(subject)['masks'].format(xfmname=xfmname, type='*'))):
            print('Refusing to overwrite existing transform %s because there are cached masks. Delete the masks manually if you want to modify the transform.' % xfmname)
            checked = False
            while not checked:
                resp = raw_input("Do you want to continue in view-only mode? (Y/N) ").lower().strip()
                if resp in ["y", "yes", "n", "no"]:
                    checked = True
                    if resp in ["y", "yes"]:
                        view_only_mode = True
                        print("Continuing in view-only mode...")
                    else:
                        raise ValueError("Exiting...")
                else:
                    print("Didn't get that, please try again..")
    except IOError:
        # Transform does not exist, make sure that reference exists
        if reference is None or not os.path.exists(reference):
            raise ValueError('Reference image file (%s) does not exist' % reference)




    m = get_aligner(subject, xfmname, epifile=reference, **kwargs)
    m.save_callback = view_callback if view_only_mode else save_callback
    m.configure_traits()

    return m

def automatic(subject, xfmname, reference, noclean=False, bbrtype="signed"):
    """Create an automatic alignment using the FLIRT boundary-based alignment (BBR) from FSL.

    If `noclean`, intermediate files will not be removed from /tmp. The `reference` image and resulting
    transform called `xfmname` will be automatically stored in the database.

    It's good practice to open up this transform afterward in the manual aligner and check how it worked.
    Do that using the following (with the same `subject` and `xfmname` used here, no need for `reference`):
    > align.manual(subject, xfmname)

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        String identifying the transform to be created.
    reference : str
        Path to a nibabel-readable image that will be used as the reference for this transform.
        Usually, this is a single (3D) functional data volume.
    noclean : bool, optional
        If True intermediate files will not be removed from /tmp (this is useful for debugging things),
        and the returned value will be the name of the temp directory. Default False.

    Returns
    -------
    Nothing unless `noclean` is True.
    """
    import shlex
    import shutil
    import tempfile
    import subprocess as sp

    from .database import db
    from .xfm import Transform
    from .options import config

    fsl_prefix = config.get("basic", "fsl_prefix")
    schfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "bbr.sch")

    retval = None
    try:
        cache = tempfile.mkdtemp()
        absreference = os.path.abspath(reference)
        raw = db.get_anat(subject, type='raw').get_filename()
        bet = db.get_anat(subject, type='brainmask').get_filename()
        wmseg = db.get_anat(subject, type='whitematter').get_filename()
        #Compute anatomical-to-epi transform
        print('FLIRT pre-alignment')
        cmd = '{fslpre}flirt  -in {epi} -ref {bet} -dof 6 -omat {cache}/init.mat'.format(
           fslpre=fsl_prefix, cache=cache, epi=absreference, bet=bet)
        if sp.call(cmd, shell=True) != 0:
           raise IOError('Error calling initial FLIRT')

        print('Running BBR')
        # Run epi-to-anat transform (this is more stable than anat-to-epi in FSL!)
        cmd = '{fslpre}flirt -in {epi} -ref {raw} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule {schfile} -bbrtype {bbrtype}'
        cmd = cmd.format(fslpre=fsl_prefix, cache=cache, raw=bet, wmseg=wmseg, epi=absreference, schfile=schfile, bbrtype=bbrtype)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.loadtxt(os.path.join(cache, "out.mat"))
        # Pass transform as FROM epi TO anat; transform will be inverted
        # back to anat-to-epi, standard direction for pycortex internal
        # storage by from_fsl
        xfm = Transform.from_fsl(x,absreference,raw)
        # Save as pycortex 'coord' transform
        xfm.save(subject,xfmname,'coord')
        print('Success')

    finally:
        if not noclean:
            shutil.rmtree(cache)
        else:
            retval = cache

    return retval

def autotweak(subject, xfmname):
    """Tweak an alignment using the FLIRT boundary-based alignment (BBR) from FSL.
    Ideally this function should actually use a limited search range, but it doesn't.
    It's probably not very useful.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        String identifying the transform to be tweaked.
    """
    import shlex
    import shutil
    import tempfile
    import subprocess as sp

    from .database import db
    from .xfm import Transform
    from .options import config

    fsl_prefix = config.get("basic", "fsl_prefix")
    schfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "bbr.sch")

    magnet = db.get_xfm(subject, xfmname, xfmtype='magnet')
    try:
        cache = tempfile.mkdtemp()
        epifile = magnet.reference.get_filename()
        raw = db.get_anat(subject, type='raw').get_filename()
        bet = db.get_anat(subject, type='brainmask').get_filename()
        wmseg = db.get_anat(subject, type='whitematter').get_filename()
        initmat = magnet.to_fsl(db.get_anat(subject, 'raw').get_filename())
        with open(os.path.join(cache, 'init.mat'), 'w') as fp:
            np.savetxt(fp, initmat, fmt='%f')
        print('Running BBR')
        cmd = '{fslpre}flirt -in {epi} -ref {raw} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule {schfile}'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=epifile)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.loadtxt(os.path.join(cache, "out.mat"))
        # Pass transform as FROM epi TO anat; transform will be inverted
        # back to anat-to-epi, standard direction for pycortex internal
        # storage by from_fsl
        Transform.from_fsl(x, epifile, raw).save(subject, xfmname+"_auto", 'coord')
        print('Saved transform as (%s, %s)'%(subject, xfmname+'_auto'))
    finally:
        shutil.rmtree(cache)
