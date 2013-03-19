import os
import nibabel
import numpy as np

from . import utils

def manual(subject, xfmname, epifile=None, **kwargs):
    from .mayavi_aligner import get_aligner
    def save_callback(aligner):
        from .db import surfs
        surfs.loadXfm(subject, xfmname, aligner.get_xfm("magnet"), xfmtype='magnet', epifile=epifile)
        print("saved xfm")

    m = get_aligner(subject, xfmname, epifile=epifile, **kwargs)
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
                    from . import db
                    db.surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', epifile=epifile)
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
        raw = surfs.getAnat(subject, type='raw')
        bet = surfs.getAnat(subject, type='brainmask')
        wmseg = surfs.getAnat(subject, type='whitematter')

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
        Transform.from_fsl(x, epifile, raw).save(subject, name, 'coord')
        print('Success')

    finally:
        if not noclean:
            shutil.rmtree(cache)
        else:
            retval = cache

    return retval

def autotweak(subject, name):
    import shutil
    import tempfile
    from .db import surfs
    magnet = surfs.getXfm(subject, name, xfmtype='magnet')
    try:
        cache = tempfile.mkdtemp()
        epifile = os.path.abspath(epifile)
        raw = surfs.getAnat(subject, type='raw')
        bet = surfs.getAnat(subject, type='brainmask')
        wmseg = surfs.getAnat(subject, type='whitematter')
        initmat = magnet.to_fsl(surfs.getAnat(subject, 'raw'))
        with open(os.path.join(cache, 'init.mat'), 'w') as fp:
            np.savetxt(fp, initmat, fmt='%f')
        print('Running BBR')
        cmd = 'fsl5.0-flirt -ref {raw} -in {epi} -dof 6 -cost bbr -wmseg {wmseg} -init {cache}/init.mat -omat {cache}/out.mat -schedule /usr/share/fsl/5.0/etc/flirtsch/bbr.sch'
        cmd = cmd.format(cache=cache, raw=raw, wmseg=wmseg, epi=epifile)
        if sp.call(cmd, shell=True) != 0:
            raise IOError('Error calling BBR flirt')

        x = np.load
    finally:
        shutil.rmtree(cache)

    raise NotImplementedError
