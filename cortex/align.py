import os
import nibabel
import numpy as np

import utils

def manual(subject, xfmname, epi=None, xfm=None, xfmtype="magnet"):
    from mayavi_aligner import get_aligner
    def save_callback(aligner):
        import db
        db.surfs.loadXfm(subject, xfmname, aligner.get_xfm("magnet"), xfmtype='magnet', epifile=epi)
        print "saved xfm"

    m = get_aligner(subject, xfmname, epi=epi, xfm=xfm, xfmtype=xfmtype)
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
                print "Saving..."
                try:
                    import db
                    db.surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', epifile=epi)
                except Exception as e:
                    print "AN ERROR OCCURRED, THE TRANSFORM WAS NOT SAVED: %s"%e
                print "Complete!"
            else:
                print "Cancelled... %s"%resp
        else:
            print "Didn't get that, please try again.."
    
    return m

def automatic(subject, name, epifile, noclean=False):
    '''
    Attempts to create an automatic alignment. If [noclean], intermediate files will not be removed from /tmp.
    '''
    import subprocess as sp
    import tempfile
    import shutil
    import shlex

    import db

    try:
        cache = tempfile.mkdtemp()
        epifile = os.path.abspath(epifile)
        raw = db.surfs.getAnat(subject, type='raw')
        bet = db.surfs.getAnat(subject, type='brainmask')
        wmseg = db.surfs.getAnat(subject, type='whitematter')

        print 'FLIRT pre-alignment'
        cmd = 'fsl5.0-flirt -ref {bet} -in {epi} -dof 6 -omat {cache}/init.mat'.format(cache=cache, epi=epifile, bet=bet)
        assert sp.call(cmd, shell=True) == 0, 'Error calling initial FLIRT'

        print 'Running BBR'
        cmd = 'fsl5.0-flirt -ref {raw} -in {epi} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule /usr/share/fsl/5.0/etc/flirtsch/bbr.sch'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=epifile)
        assert sp.call(cmd, shell=True) == 0, 'Error calling BBR flirt'

        xfm = np.loadtxt(os.path.join(cache, "out.mat"))
        
        ## Adapted from dipy.external.fsl.flirt2aff#############################
        import numpy.linalg as npl
        
        in_hdr = nibabel.load(epifile).get_header()
        ref_hdr = nibabel.load(raw).get_header()
        
        # get_zooms gets the positive voxel sizes as returned in the header
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        
        if npl.det(in_hdr.get_best_affine())>=0:
            inspace = np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if npl.det(ref_hdr.get_best_affine())>=0:
            refspace = np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))
        ########################################################################

        epi = nibabel.load(epifile).get_header().get_base_affine()
        M = nibabel.load(raw).get_affine()
        X = xfm
        inv = np.linalg.inv

        coord = np.dot(npl.inv(inspace), np.dot(inv(X), np.dot(refspace, inv(M))))
        db.surfs.loadXfm(subject, name, coord, xfmtype="coord", epifile=epifile)

    finally:
        if not noclean:
            shutil.rmtree(cache)
        else:
            pass

    return locals()


def _x_flipper(N_i):
    #Copied from dipy
    flipr = np.diag([-1, 1, 1, 1])
    flipr[0,3] = N_i - 1
    return flipr
