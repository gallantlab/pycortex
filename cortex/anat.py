import os
import shlex
import shutil
import tempfile
import subprocess as sp

import numpy as np
import nibabel

from . import db
from . import utils

def brainmask(subject):
    anatform = db.surfs.getFiles(subject)['anats']
    print('Brain masking anatomical...')
    raw = anatform.format(type='raw')
    bet = anatform.format(type='brainmask')
    cmd = 'fsl5.0-bet {raw} {bet} -B -v'.format(raw=raw, bet=bet)
    assert sp.call(cmd, shell=True) == 0, "Error calling fsl-bet"

def whitematter(subject):
    anatform = db.surfs.getFiles(subject)['anats']
    bet = anatform.format(type='brainmask')
    fast = anatform.format(type='whitematter')
    if not os.path.exists(bet):
        brainmask(subject)

    try:
        cache = tempfile.mkdtemp()
        print("Segmenting the brain...")
        cmd = 'fsl5.0-fast -o {cache}/fast {bet}'.format(cache=cache, bet=bet)
        assert sp.call(cmd, shell=True) == 0, "Error calling fsl-fast"
        cmd = 'fsl5.0-fslmaths {cache}/fast_pve_2 -thr 0.5 -bin {out}'.format(cache=cache, out=fast)
        assert sp.call(cmd, shell=True) == 0, 'Error calling fsl-maths'
    finally:
        shutil.rmtree(cache)

def curvature(subject, **kwargs):
    curvs = utils.get_curvature(subject, **kwargs)
    anatform = db.surfs.getFiles(subject)['anats']
    curv = anatform.format(type='curvature')
    curv, ext = os.path.splitext(curv)
    np.savez_compressed('%s.npz'%curv, left=curvs[0], right=curvs[1])
