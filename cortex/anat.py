import os
import shlex
import shutil
import tempfile
import subprocess as sp

import nibabel
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

def curvature(outfile, subject, **kwargs):
    left, right = utils.get_curvature(subject, **kwargs)
    np.savez(outfile, left=left, right=right)

def distortion(outfile, subject, type='areal', **kwargs):
    left, right = utils.get_distortion(subject, type=type, **kwargs)
    np.savez(outfile, left=left, right=right)

def thickness(outfile, subject):
    pl, pr = surfs.getSurf(subject, "pia")
    wl, wr = surfs.getSurf(subject, "wm")
    left = np.sqrt(((pl[0] - wl[0])**2).sum(1))
    right = np.sqrt(((pr[0] - wr[0])**2).sum(1))
    np.savez(outfile, left=left, right=right)

def voxelize(outfile, subject, surf='wm', mp=True):
    '''Voxelize the whitematter surface to generate the white matter mask'''
    from . import polyutils
    shape = surfs.getAnat(subject, "raw").get_shape()
    vox = np.zeros(shape, dtype=bool)
    for pts, polys in surfs.getSurf(subject, surf, nudge=False):
        xfm = Transform(np.linalg.inv(nib.get_affine()), nib)
        vox += polyutils.voxelize(xfm(pts), polys, shape=shape, center=(0,0,0), mp=mp)

    if surf == 'wm':
        nib = nibabel.Nifti1Image(vox, nib.get_affine(), header=nib.get_header())
        nib.to_filename(outfile)

    return vox.T

def flatmask(outfile, subject, height=1024):
    from . import polyutils
    import Image
    import ImageDraw
    pts, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    bounds = [p for p in polyutils.trace_poly(polyutils.boundary_edges(polys))]
    left, right = bounds[0], bounds[1]
    aspect = (height / (pts.max(0) - pts.min(0))[1])
    lpts = (pts[left] - pts.min(0)) * aspect
    rpts = (pts[right] - pts.min(0)) * aspect

    im = Image.new('L', (int(aspect * (pts.max(0) - pts.min(0))[0]), height))
    draw = ImageDraw.Draw(im)
    draw.polygon(lpts[:,:2].ravel().tolist(), fill=255)
    draw.polygon(rpts[:,:2].ravel().tolist(), fill=255)
    np.savez(outfile, mask=np.array(im) > 0)