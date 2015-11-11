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

def anat_to_mni(subject, do=True):
    """Create an automatic alignment of an anatomical image to the MNI standard.

    This function does the following:
    1) Re-orders orientation labels on anatomical images using fslreorient2std (without modifying the existing files)
    2) Calls FLIRT
    3) Calls FNIRT with the transform estimated by FLIRT as the specified
    4) Gets the resulting warp field, samples it at each vertex location, and calculates MNI coordinates.
    5) Saves these coordinates as a surfinfo file in the db.

    Parameters
    ----------
    subject : str
        Subject identifier.
    do : bool
        Actually execute the commands (True), or just print them (False, useful for debugging).

    Returns
    -------
    pts : the vertices of the fiducial surface
    mnipts : the mni coordinates of those vertices (same shape as pts, corresponding indices)
    """

    import shlex
    import shutil
    import tempfile
    import subprocess as sp
    import nibabel as nib

    from .database import db
    from .xfm import Transform
    from .options import config
    from .dataset import Volume

    fsl_prefix = config.get("basic", "fsl_prefix")
    cache = tempfile.mkdtemp()

    print('anat_to_mni, subject: %s' % subject)
    
    try:
        raw_anat = db.get_anat(subject, type='raw').get_filename()
        bet_anat = db.get_anat(subject, type='brainmask').get_filename()
        betmask_anat = db.get_anat(subject, type='brainmask_mask').get_filename()
        anat_dir = os.path.dirname(raw_anat)
        odir = cache

        # stem for the reoriented-into-MNI anatomical images (required by FLIRT/FNIRT)
        reorient_anat = 'reorient_anat'
        reorient_cmd = '{fslpre}fslreorient2std {raw_anat} {adir}/{ra_raw}'.format(fslpre=fsl_prefix,raw_anat=raw_anat, adir=odir, ra_raw=reorient_anat)
        print('Reorienting anatomicals using fslreorient2std, cmd like: \n%s' % reorient_cmd)
        if do and sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on raw anatomical')
        reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_raw}_brain'.format(fslpre=fsl_prefix,bet_anat=bet_anat, adir=odir, ra_raw=reorient_anat)
        if do and sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on brain-extracted anatomical')

        ra_betmask = reorient_anat + "_brainmask"
        reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_betmask}'.format(fslpre=fsl_prefix,bet_anat=betmask_anat, adir=odir, ra_betmask=ra_betmask)
        if do and sp.call(reorient_cmd, shell=True) != 0:
            raise IOError('Error calling fslreorient2std on brain-extracted mask')
        
        fsldir = os.environ['FSLDIR']
        standard = '%s/data/standard/MNI152_T1_1mm'%fsldir
        bet_standard = '%s_brain'%standard
        standardmask = '%s_mask_dil'%bet_standard
        cout = 'mni2anat' #stem of the filenames of the transform estimates

        # initial affine anatomical-to-standard registration using FLIRT. required, as the output xfm is used as a start by FNIRT.
        flirt_cmd = '{fslpre}flirt -in {bet_standard} -ref {adir}/{ra_raw}_brain -dof 6 -omat {adir}/{cout}_flirt'
        flirt_cmd = flirt_cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, bet_standard=bet_standard, adir=odir, cout=cout)
        print('Running FLIRT to estimate initial affine transform with command:\n%s'%flirt_cmd)
        if do and sp.call(flirt_cmd, shell=True) != 0:
            raise IOError('Error calling FLIRT with command: %s' % flirt_cmd)

        # FNIRT mni-to-anat transform estimation cmd (does not apply any transform, but generates estimate [cout])
        # the MNI152 2mm config is used even though we're referencing 1mm, per this FSL list post:
        # https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;d14e5a9d.1105
        cmd = '{fslpre}fnirt --in={standard} --ref={ad}/{ra_raw} --refmask={ad}/{refmask} --aff={ad}/{cout}_flirt --cout={ad}/{cout}_fnirt --fout={ad}/{cout}_field --iout={ad}/{cout}_iout --config=T1_2_MNI152_2mm'
        cmd = cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, standard=standard, refmask=ra_betmask, ad=odir, anat_dir=anat_dir, cout=cout)
        print('Running FNIRT to estimate transform, using the following command... this can take a while:\n%s'%cmd)
        if do and sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling fnirt with cmd: %s'%cmd)

        [pts, polys] = db.get_surf(subject,"fiducial",merge="True")

        #print('raw anatomical: %s\nbet anatomical: %s\nflirt cmd:%s\nfnirt cmd: %s\npts: %s' % (raw_anat,bet_anat,flirt_cmd,cmd,pts))

        # take the reoriented anatomical, get its affine coord transform, invert this, and save it
        reo_xfmnm = 'reorient_inv'
        # need to change this line, as the reoriented anatomical is not in the db but in /tmp now
        # re_anat = db.get_anat(subject,reorient_anat)
        reo_anat_fn = '{odir}/{reorient_anat}.nii.gz'.format(odir=odir,reorient_anat=reorient_anat)
        # print(reo_anat_fn)
        # since the reoriented anatomicals aren't stored in the db anymore, db.get_anat() will not work (?)
        re_anat = nib.load(reo_anat_fn)
        reo_xfm = Transform(np.linalg.inv(re_anat.get_affine()),re_anat)
        reo_xfm.save(subject,reo_xfmnm,"coord")

        # get the reoriented anatomical's qform and its inverse, they will be needed later
        aqf = re_anat.get_qform()
        aqfinv = np.linalg.inv(aqf)

        # load the warp field data as a volume
        # since it's not in the db anymore but in /tmp instead of:
        # warp = db.get_anat(subject,'%s_field'%cout)
        # it's this:
        warp_fn = '{ad}/{cout}_field.nii.gz'.format(ad=odir,cout=cout)
        # print warp_fn
        warp = nib.load(warp_fn)
        wd = warp.get_data()
        # need in (t,z,y,x) order
        wd = np.swapaxes(wd,0,3) # x <--> t
        wd = np.swapaxes(wd,1,2) # y <--> z
        wv = Volume(wd,subject,reo_xfmnm)

        # now do the mapping! this gets the warp field values at the corresponding points
        # (uses fiducial surface by default)
        warpvd = wv.map(projection="lanczos")

        # reshape into something sensible
        warpverts_L = [vs for vs in np.swapaxes(warpvd.left,0,1)]
        warpverts_R = [vs for vs in np.swapaxes(warpvd.right,0,1)]
        warpverts_ordered = np.concatenate((warpverts_L, warpverts_R))

        # append 1s for matrix multiplication (coordinate transformation)
        o = np.ones((len(pts),1))
        pad_pts = np.append(pts, o, axis=1)

        # print pts, len(pts), len(pts[0]), warpverts_ordered, len(warpverts_ordered), pad_pts, len(pad_pts), pad_pts[0]

        # transform vertex coords from mm to vox using the anat's qform
        voxcoords = [aqfinv.dot(padpt) for padpt in pad_pts]
        # add the offsets specified in the warp at those locations (ignoring the 1s here)
        mnivoxcoords = [voxcoords[n][:-1] + warpverts_ordered[n] for n in range(len(voxcoords))]
        # re-pad for matrix multiplication
        pad_mnivox = np.append(mnivoxcoords, o, axis=1)

        # multiply by the standard's qform to recover mm coords
        std = nib.load('%s.nii.gz'%standard)
        stdqf = std.get_qform()
        mni_coords = np.array([stdqf.dot(padmni)[:-1] for padmni in pad_mnivox])

        # some debug output
        # print pts, mni_coords
        # print pts[0], mni_coords[0]
        # print len(pts), len(mni_coords)
        # print type(pts), type(pts[0][0]), type(mni_coords)

        # now split mni_coords into left and right arrays for saving
        nverts_L = len(warpverts_L)
        #print nverts_L
        left = mni_coords[:nverts_L]
        right = mni_coords[nverts_L:]
        #print len(left), len(right)

        mni_surfinfo_fn = db.get_paths(subject)['surfinfo'].format(type='mnicoords',opts='')
        print('Saving mni coordinates as a surfinfo...')
        np.savez(mni_surfinfo_fn,leftpts=left,rightpts=right)

        return (pts, mni_coords)

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
