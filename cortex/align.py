"""Contains functions for making alignments between functional data and the surface, or, finding where the brain is.
"""
import os
import numpy as np
from builtins import input

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
                resp = input("Do you want to continue in view-only mode? (Y/N) ").lower().strip()
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

def fs_manual(subject, xfmname, output_name="register.lta", wm_color="yellow", 
    pial_color="blue", wm_surface='white', noclean=False, reference=None, inspect_only=False):
    """Open Freesurfer FreeView GUI for manually aligning/adjusting a functional
    volume to the cortical surface for `subject`. This creates a new transform
    called `xfmname`. The name of a nibabel-readable file (e.g. NIfTI) should be
    supplied as `reference`. This image will be copied into the database.

    IMPORTANT: This function assumes that the resulting .lta file is saved as:
    "{default folder chosen by FreeView (should be /tmp/fsalign_xxx)}/{output_name}".
    
    NOTE: Half-fixed some potential bugs in here, related to assumptions about how 
    results from mri_info calls would be formatted. IFF .dat files are written
    based on nii files that have been stripped of their headers, then there will 
    be an extra line at the top stating that the coordinates are assumed to be in mm.
    Without this line, the code here fails. Seems brittle, ripe for future bugs.

    ALSO: all the freesurfer environment stuff shouldn't be necessary, except that
    I don't know what vox2ras-tkr is doing.


    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        The name of the transform to be modified.
    output_name : str
        The name of the .lta file generated after FreeView editing.
    wm_color : str | "blue"
        Color of the white matter surface. Default is "blue". This can
        also be adjusted in the FreeView GUI.
    pial_color : str | "red"
        Color of the pial surface. Default is "red". This can also be adjusted
        the FreeView GUI.
    noclean : boolean | False
        If True, intermediate files will not be removed from /tmp/fsalign_xxx
        (this is useful for debugging things), and the returned value will be
        the name of the temp directory. Default False.
    reference : str
        name of reference (generally, functional) volume. Only provide this if
        you are working from scratch (if no transform exists already), else
        it will throw an error.
    inspect_only : boolean | False
        Whether to open transform to view only (if True, nothing is saved
        when freeview is closed)
    wm_surface : string
        name for white matter surface to use. 'white' or 'smoothwm'

    Returns
    -------
    Nothing unless noclean is true.
    """

    import subprocess as sp
    import tempfile
    import shutil
    from .xfm import Transform
    from .database import db

    retval = None

    try:
        try:
            cache = tempfile.mkdtemp(prefix="fsalign_")
            sub_xfm = db.get_xfm(subject, xfmname)

            # if masks have been cached, quit! user must remove them by hand
            from glob import glob
            masks_exist = len(glob(db.get_paths(subject)['masks'].format(xfmname=xfmname, type='*')))
            if masks_exist and not inspect_only:
                print('Refusing to overwrite existing transform %s because there are cached masks. Delete the masks manually if you want to modify the transform.' % xfmname)
                raise ValueError('Exiting...')
            if reference is not None:
                raise ValueError('Refusing to overwrite extant reference for transform')
        except IOError:
            if reference is None: 
                print("Transform does not exist!")

        if reference is None:
            # Load load extant transform-relevant things
            reference = sub_xfm.reference.get_filename()
            _ = sub_xfm.to_freesurfer(os.path.join(cache, "register.dat"), subject) # Transform in freesurfer .dat format
            # Command for FreeView and run
            cmd = ("freeview -v $SUBJECTS_DIR/{sub}/mri/orig.mgz "
                    "{ref}:reg={reg} "
                   "-f $SUBJECTS_DIR/{sub}/surf/lh.{wms}:edgecolor={wmc} $SUBJECTS_DIR/{sub}/surf/rh.{wms}:edgecolor={wmc} "
                   "$SUBJECTS_DIR/{sub}/surf/lh.pial:edgecolor={pialc} $SUBJECTS_DIR/{sub}/surf/rh.pial:edgecolor={pialc}")
            cmd = cmd.format(sub=subject, ref=reference, reg=os.path.join(cache, "register.dat"),
                             wmc=wm_color, pialc=pial_color, wms=wm_surface)
            print('=== Calling (NO REFERENCE PROVIDED): ===')
            print(cmd)
        else:
            # Command for FreeView and run
            cmd = ("freeview -v $SUBJECTS_DIR/{sub}/mri/orig.mgz "
                   "{ref} "
                   "-f $SUBJECTS_DIR/{sub}/surf/lh.{wms}:edgecolor={wmc} $SUBJECTS_DIR/{sub}/surf/rh.{wms}:edgecolor={wmc} "
                   "$SUBJECTS_DIR/{sub}/surf/lh.pial:edgecolor={pialc} $SUBJECTS_DIR/{sub}/surf/rh.pial:edgecolor={pialc}")
            cmd = cmd.format(sub=subject, ref=reference,
                             wmc=wm_color, pialc=pial_color, 
                             wms=wm_surface)
            print('=== Calling: ===')
            print(cmd)
            
        if not inspect_only:
            sfile = os.path.join(cache, output_name)
            print('\nREGISTRATION MUST BE SAVED AS:\n\n{}'.format(sfile))

        # Run and save transform when user is done editing
        if sp.call(cmd, shell=True) != 0:
            raise IOError("Problem with FreeView!")
        else:
            if not inspect_only:
                # Convert transform into .dat format
                # Unclear why we're not just saving in .dat format above...?
                reg_dat = os.path.join(cache, os.path.splitext(output_name)[0] + ".dat")
                cmd = "lta_convert --inlta {inlta} --outreg {regdat}"
                cmd = cmd.format(inlta=os.path.join(cache, output_name), regdat=reg_dat)
                if sp.call(cmd, shell=True) != 0:
                    raise IOError("Error converting lta into dat!")
                # Save transform to pycortex
                xfm = Transform.from_freesurfer(reg_dat, reference, subject)
                db.save_xfm(subject, xfmname, xfm.xfm, xfmtype='coord', reference=reference)
                print("saved xfm")
    except Exception as e:
        raise(e)
    finally:
        if not noclean:
            shutil.rmtree(cache)
        else:
            retval = cache
    return retval


def automatic(subject, xfmname, reference, noclean=False, bbrtype="signed", pre_flirt_args='', use_fs_bbr=False):
    """Create an automatic alignment using the FLIRT boundary-based alignment (BBR) from FSL.

    If `noclean`, intermediate files will not be removed from /tmp. The `reference` image and resulting
    transform called `xfmname` will be automatically stored in the database.

    It's good practice to open up this transform afterward in the manual aligner and check how it worked.
    Do that using the following (with the same `subject` and `xfmname` used here, no need for `reference`):
    > align.manual(subject, xfmname)

    If automatic alignment gives you a very bad answer, you can try giving the pre-BBR FLIRT
    some hints by passing '-usesqform' in as `pre_flirt_args`.

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
    bbrtype : str, optional
        The 'bbrtype' argument that is passed to FLIRT.
    pre_flirt_args : str, optional
        Additional arguments that are passed to the FLIRT pre-alignment step (not BBR).
    use_fs_bbr : bool, optional
        If True will use freesurfer bbregister instead of FSL BBR.
    save_dat : bool, optional
        If True, will save the register.dat file from freesurfer bbregister into
        freesurfer's $SUBJECTS_DIR/subject/tmp.

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

        if use_fs_bbr:
            print('Running freesurfer BBR')
            cmd = 'bbregister --s {sub} --mov {absref} --init-fsl --reg {cache}/register.dat --t1'
            cmd = cmd.format(sub=subject, absref=absreference, cache=cache)

            if sp.call(cmd, shell=True) != 0:
                raise IOError('Error calling freesurfer BBR!')

            xfm = Transform.from_freesurfer(os.path.join(cache, "register.dat"), absreference, subject)
        else:
            raw = db.get_anat(subject, type='raw').get_filename()
            bet = db.get_anat(subject, type='brainmask').get_filename()
            wmseg = db.get_anat(subject, type='whitematter').get_filename()
            #Compute anatomical-to-epi transform
            print('FLIRT pre-alignment')
            cmd = '{fslpre}flirt  -in {epi} -ref {bet} -dof 6 {pre_flirt_args} -omat {cache}/init.mat'.format(
               fslpre=fsl_prefix, cache=cache, epi=absreference, bet=bet, pre_flirt_args=pre_flirt_args)
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
