import os
import numpy as np

from . import utils

def manual(subject, xfmname, reference=None, **kwargs):
    from .db import surfs
    from .mayavi_aligner import get_aligner
    def save_callback(aligner):
        surfs.loadXfm(subject, xfmname, aligner.get_xfm("magnet"), xfmtype='magnet', reference=reference)
        print("saved xfm")

    if reference is not None:
        try:
            surfs.getXfm(subject, xfmname)
            raise ValueError('Refusing to overwrite an existing transform')
        except IOError:
            pass

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

def automatic(subject, name, epifile, noclean=False):
    '''
    Attempts to create an automatic alignment. If [noclean], intermediate files will not be removed from /tmp.
    '''
    import shlex
    import shutil
    import tempfile
    import subprocess as sp

    from .db import surfs
    from .xfm import Transform

    retval = None
    try:
        cache = tempfile.mkdtemp()
        epifile = os.path.abspath(epifile)
        raw = surfs.getAnat(subject, type='raw').get_filename()
        bet = surfs.getAnat(subject, type='brainmask').get_filename()
        wmseg = surfs.getAnat(subject, type='whitematter').get_filename()
        # The following transformations compute EPI-to-ANATOMICAL transformations.
        # These are BACKWARDS from what we eventually want
        print('FLIRT pre-alignment')
        cmd = 'fsl5.0-flirt -ref {bet} -in {epi} -dof 6 -omat {cache}/init.mat'.format(cache=cache, epi=epifile, bet=bet)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling initial FLIRT')

        print('Running BBR')
        cmd = 'fsl5.0-flirt -ref {raw} -in {epi} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule /usr/share/fsl/5.0/etc/flirtsch/bbr.sch'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=epifile)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.loadtxt(os.path.join(cache, "out.mat"))
        # Original code (before ML modification of from_fsl):
        #Transform.from_fsl(x, epifile, raw).save(subject, name, 'coord')
        # Modified by ML 2013.07
        # Take the inverse of the transform
        inv = np.linalg.inv
        Transform.from_fsl(inv(x),raw,epifile).save(subject,name,'coord')
        print('Success')

    finally:
        if not noclean:
            shutil.rmtree(cache)
        else:
            retval = cache

    return retval

def autotweak(subject, name):
    import shlex
    import shutil
    import tempfile
    import subprocess as sp

    from .db import surfs
    from .xfm import Transform

    magnet = surfs.getXfm(subject, name, xfmtype='magnet')
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
        Transform.from_fsl(x, epifile, raw).save(subject, name+"_auto", 'coord')
        print('Saved transform as (%s, %s)'%(subject, name+'_auto'))
    finally:
        shutil.rmtree(cache)
