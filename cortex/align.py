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
    from .db import surfs
    from .mayavi_aligner import get_aligner
    def save_callback(aligner):
        surfs.loadXfm(subject, xfmname, aligner.get_xfm("magnet"), xfmtype='magnet', reference=reference)
        print("saved xfm")

    # Check whether transform w/ this xfmname already exists
    try:
        surfs.getXfm(subject, xfmname)
        # Transform exists, make sure that reference is None
        if reference is not None:
            raise ValueError('Refusing to overwrite reference for existing transform %s, use reference=None to load stored reference' % xfmname)
    except IOError:
        # Transform does not exist, make sure that reference exists
        if reference is None or not os.path.exists(reference):
            raise ValueError('Reference image file (%s) does not exist' % reference)

    m = get_aligner(subject, xfmname, epifile=reference, **kwargs)
    m.save_callback = save_callback
    m.configure_traits()
    
    magnet = m.get_xfm("magnet")
    epi = os.path.abspath(m.epi_file.get_filename())

    checked = False
    while not checked:
        resp = raw_input("Save? (Y/N) ").lower().strip()
        if resp in ["y", "yes", "n", "no"]:
            checked = True
            if resp in ["y", "yes"]:
                print("Saving...")
                try:
                    surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', reference=reference)
                except Exception as e:
                    print("AN ERROR OCCURRED, THE TRANSFORM WAS NOT SAVED: %s"%e)
                print("Complete!")
            else:
                print("Cancelled... %s"%resp)
        else:
            print("Didn't get that, please try again..")
    
    return m

def automatic(subject, xfmname, reference, noclean=False):
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

    from .db import surfs
    from .xfm import Transform

    retval = None
    try:
        cache = tempfile.mkdtemp()
        absreference = os.path.abspath(reference)
        raw = surfs.getAnat(subject, type='raw').get_filename()
        bet = surfs.getAnat(subject, type='brainmask').get_filename()
        wmseg = surfs.getAnat(subject, type='whitematter').get_filename()
        # Compute anatomical-to-epi transform
        print('FLIRT pre-alignment')
        cmd = 'fsl5.0-flirt -ref {bet} -in {epi} -dof 6 -omat {cache}/init.mat'.format(cache=cache, epi=absreference, bet=bet)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling initial FLIRT')
        print('Running BBR')
        # Run epi-to-anat transform (this is more stable than anat-to-epi in FSL!)
        cmd = 'fsl5.0-flirt -in {epi} -ref {raw} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule /usr/share/fsl/5.0/etc/flirtsch/bbr.sch'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=absreference)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.loadtxt(os.path.join(cache, "out.mat"))
        # Invert transform back to anat-to-epi, pycortex standard direction, and save as pycortextransform
        inv = np.linalg.inv
        Transform.from_fsl(inv(x),raw,absreference).save(subject,xfmname,'coord')
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

    from .db import surfs
    from .xfm import Transform

    magnet = surfs.getXfm(subject, xfmname, xfmtype='magnet')
    try:
        cache = tempfile.mkdtemp()
        epifile = magnet.reference.get_filename()
        raw = surfs.getAnat(subject, type='raw').get_filename()
        bet = surfs.getAnat(subject, type='brainmask').get_filename()
        wmseg = surfs.getAnat(subject, type='whitematter').get_filename()
        initmat = magnet.to_fsl(surfs.getAnat(subject, 'raw').get_filename())
        with open(os.path.join(cache, 'init.mat'), 'w') as fp:
            np.savetxt(fp, initmat, fmt='%f')
        print('Running BBR')
        cmd = 'fsl5.0-flirt -ref {raw} -in {epi} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule /usr/share/fsl/5.0/etc/flirtsch/bbr.sch'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=epifile)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.loadtxt(os.path.join(cache, "out.mat"))
        Transform.from_fsl(x, epifile, raw).save(subject, xfmname+"_auto", 'coord')
        print('Saved transform as (%s, %s)'%(subject, xfmname+'_auto'))
    finally:
        shutil.rmtree(cache)
