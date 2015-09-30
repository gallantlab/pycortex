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

    # Check whether transform w/ this xfmname already exists
    try:
        db.get_xfm(subject, xfmname)
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
                    db.save_xfm(subject, xfmname, magnet, xfmtype='magnet', reference=reference)
                except Exception as e:
                    print("AN ERROR OCCURRED, THE TRANSFORM WAS NOT SAVED: %s"%e)
                print("Complete!")
            else:
                print("Cancelled... %s"%resp)
        else:
            print("Didn't get that, please try again..")
    
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

def anat_to_mni(subject, xfmname, noclean=False):
    """Create an automatic alignment of an anatomical image to the MNI standard.

    If `noclean`, intermediate files will not be removed from /tmp. The `reference` image and resulting 
    transform called `xfmname` will be automatically stored in the database.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        String identifying the transform to be created.
    anatimg : str
        Path to a nibabel-readable image that will be used as the reference for this transform.
        This should be a 3D anatomical volume.
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

    print('anat_to_mni, subject: %s, xfmname: %s' % (subject, xfmname))
    
    try:
        raw_anat = db.get_anat(subject, type='raw').get_filename()
        bet_anat = db.get_anat(subject, type='brainmask').get_filename()
        betmask_anat = db.get_anat(subject, type='brainmask_mask').get_filename()
        anat_dir = os.path.dirname(raw_anat)
        odir = anat_dir

        # stem for the reoriented-into-MNI anatomical images (required by FLIRT/FNIRT)
        reorient_anat = 'reorient_anat'
        reorient_cmd = '{fslpre}fslreorient2std {raw_anat} {adir}/{ra_raw}'.format(fslpre=fsl_prefix,raw_anat=raw_anat, adir=odir, ra_raw=reorient_anat)
        print('Reorienting anatomicals using fslreorient2std, cmd like: \n%s' % reorient_cmd)
        if sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on raw anatomical')
        reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_raw}_brain'.format(fslpre=fsl_prefix,bet_anat=bet_anat, adir=odir, ra_raw=reorient_anat)
        if sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on brain-extracted anatomical')

        ra_betmask = reorient_anat + "_brainmask"
        reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_betmask}'.format(fslpre=fsl_prefix,bet_anat=betmask_anat, adir=odir, ra_betmask=ra_betmask)
        if sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on brain-extracted mask')
        
        fsldir = os.environ['FSLDIR']
        standard = '%s/data/standard/MNI152_T1_1mm'%fsldir
        bet_standard = '%s_brain'%standard
        standardmask = '%s_mask_dil'%bet_standard
        cout = 'mni2anat' #stem of the filenames of the transform estimates

        # initial affine anatomical-to-standard registration using FLIRT. required, as the output xfm is used as a start by FNIRT.
        flirt_cmd = '{fslpre}flirt -in {bet_standard} -ref {adir}/{ra_raw}_brain -dof 6 -omat /tmp/{cout}_flirt'
        flirt_cmd = flirt_cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, bet_standard=bet_standard, adir=odir, cout=cout)
        print('Running FLIRT to estimate initial affine transform')
        #if sp.call(flirt_cmd, shell=True) != 0:
        #    raise IOError('Error calling FLIRT with command: %s' % flirt_cmd)

        # FNIRT mni-to-anat transform estimation cmd (does not apply any transform, but generates estimate [cout])
        cmd = '{fslpre}fnirt --in={standard} --ref={ad}/{ra_raw} --refmask={ad}/{refmask} --aff=/tmp/{cout}_flirt --cout={anat_dir}/{cout}_fnirt --fout={anat_dir}/{cout}_field --iout=/tmp/mni2anat_iout --config=T1_2_MNI152_2mm'
        cmd = cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, standard=standard, refmask=ra_betmask, ad=odir, anat_dir=anat_dir, cout=cout)
        print('Running FNIRT to estimate transform... this can take a while')
        #if sp.call(cmd, shell=True) != 0:
        #    raise IOError('Error calling fnirt with cmd: %s'%cmd)

        # we now have, in /tmp/cout_fnirt, the warp estimate that should be passed to img2stdcoord.
        # let's get all vertex coordinates
        cfile = '/tmp/fid_coords'
        cfile_warped = '/tmp/mni_coords'
        [pts, polys] = db.get_surf(subject,"fiducial",merge=True)
        # np.savetxt(cfile, pts, fmt='%g')

        xfm_cmd = 'cat {coordfile} | {fslpre}img2stdcoord -mm -std {ad}/{ra_raw} -img {standard} -warp {anat_dir}/{cout}_fnirt > {cfile_warped}'
        xfm_cmd = xfm_cmd.format(coordfile=cfile, fslpre=fsl_prefix, ra_raw=reorient_anat, standard=standard, ad=odir, anat_dir=anat_dir, cout=cout, cfile_warped=cfile_warped)

        print('raw anatomical: %s\nbet anatomical: %s\nflirt cmd:%s\nfnirt cmd: %s\nxfm cmd: %s' % (raw_anat,bet_anat,flirt_cmd,cmd,xfm_cmd))

    finally:
        pass


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
