import os
import shlex
import shutil
import tempfile
import subprocess as sp

import numpy as np

from . import utils
from .db import surfs
from .xfm import Transform

def brainmask(outfile, subject):
    raw = surfs.getAnat(subject, type='raw').get_filename()
    print('Brain masking anatomical...')
    cmd = 'fsl5.0-bet {raw} {bet} -B -v'.format(raw=raw, bet=outfile)
    assert sp.call(cmd, shell=True) == 0, "Error calling fsl-bet"

def whitematter(outfile, subject):
    try:
        voxelize(outfile, subject, surf="wm")
    except IOError:
        bet = surfs.getAnat(subject, type='brainmask').get_filename()
        try:
            cache = tempfile.mkdtemp()
            print("Segmenting the brain...")
            cmd = 'fsl5.0-fast -o {cache}/fast {bet}'.format(cache=cache, bet=bet)
            assert sp.call(cmd, shell=True) == 0, "Error calling fsl-fast"
            cmd = 'fsl5.0-fslmaths {cache}/fast_pve_2 -thr 0.5 -bin {out}'.format(cache=cache, out=outfile)
            assert sp.call(cmd, shell=True) == 0, 'Error calling fsl-maths'
        finally:
            shutil.rmtree(cache)

def voxelize(outfile, subject, surf='wm', mp=True):
    '''Voxelize the whitematter surface to generate the white matter mask'''
    from . import polyutils
    nib = surfs.getAnat(subject, "raw")
    shape = nib.get_shape()
    vox = np.zeros(shape, dtype=bool)
    for pts, polys in surfs.getSurf(subject, surf, nudge=False):
        xfm = Transform(np.linalg.inv(nib.get_affine()), nib)
        vox += polyutils.voxelize(xfm(pts), polys, shape=shape, center=(0,0,0), mp=mp)
        
    import nibabel
    nib = nibabel.Nifti1Image(vox, nib.get_affine(), header=nib.get_header())
    nib.to_filename(outfile)

    return vox.T
