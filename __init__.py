import os
import json
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
options = json.load(open(os.path.join(cwd, "defaults.json")))

from db import surfs
import view

def mosaic(data, xy=(6, 5), trim=10, skip=1, show=True, **kwargs):
    '''mosaic(data, xy=(6, 5), trim=10, skip=1)

    Turns volume data into a mosaic, useful for quickly viewing volumetric data

    Parameters
    ----------
    data : array_like
        3D volumetric data to mosaic
    xy : tuple, optional
        tuple(x, y) for the grid of images. Default (6, 5)
    trim : int, optional
        How many pixels to trim from the edges of each image. Default 10
    skip : int, optional
        How many slices to skip in the beginning. Default 1
    '''
    assert len(data.shape) == 3, "Are you sure this is volumetric?"
    dat = data.copy()
    if trim>0:
        dat = dat[:, trim:-trim, trim:-trim]
    d = dat.shape[1:]
    output = np.zeros(d*np.array(xy))
    
    c = skip
    for i in range(xy[0]):
        for j in range(xy[1]):
            if c < len(dat):
                output[d[0]*i:d[0]*(i+1), d[1]*j:d[1]*(j+1)] = dat[c]
            c+= 1
    
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(output, **kwargs)
        plt.xticks([])
        plt.yticks([])

    return output

def flatten(data, subject=None, xfmname=None, **kwargs):
    import view
    import matplotlib.pyplot as plt
    data = view.quickflat(data, 
        subject or options['default_subject'],
        xfmname or options['default_xfm'])
    plt.imshow(data, aspect='equal', **kwargs)
    return data

def epi_to_anat(data, subject=None, xfmname=None, filename=None):
    '''/usr/share/fsl/4.1/bin/flirt -in /tmp/epidat.nii -applyxfm -init /tmp/coordmat.mat -out /tmp/realign.nii.gz -paddingsize 0.0 -interp trilinear -ref /tmp/anat.nii'''
    import nifti
    import shlex
    import tempfile
    import subprocess as sp
    epifile = tempfile.mktemp(suffix=".nii")
    anatfile = tempfile.mktemp(suffix=".nii")
    xfmfile = tempfile.mktemp()
    if filename is None:
        filename = tempfile.mktemp(suffix=".nii")

    #load up relevant data, get transforms
    anatdat = surfs.subjects[subject].anatomical
    anataff = anatdat.get_affine()
    epixfm = np.linalg.inv(surfs.getXfm(subject, xfmname)[0])
    xfm = np.dot(np.abs(anataff), epixfm)
    np.savetxt(xfmfile, xfm, "%f")

    #save the epi data and the anatomical (probably in hdr/img) into nifti
    nifti.NiftiImage(data).save(epifile)
    nifti.NiftiImage(np.array(anatdat.get_data()).T).save(anatfile)

    cmd = "fsl4.1-flirt -in {epi} -applyxfm -init {xfm} -out {out} -paddingsize 0.0 -interp trilinear -ref {anat}"
    sp.Popen(shlex.split(cmd.format(epi=epifile, xfm=xfmfile, anat=anatfile, out=filename))).wait()
    output = nifti.NiftiImage(filename).data.copy()
    print "Saving to %s"%filename
    os.unlink(epifile)
    os.unlink(anatfile)
    os.unlink(xfmfile)
    return output

def get_cortical_mask(subject, xfmname, shape=(31, 100, 100)):
    data = np.zeros(shape)
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    data[tuple(coords.T)] = 1

    return data


def get_vox_dist(subject, xfmname, shape=(31, 100, 100)):
    '''Get the distance (in mm) from each functional voxel to the closest
    point on the surface.

    Parameters
    ----------
    subject : str
        Name of the subject
    xfmname : str
        Name of the transform
    shape : tuple
        Output shape for the mask

    Returns
    -------
    dist : ndarray
        Distance (in mm) to the closest point on the surface

    argdist : ndarray
        Point index for the closest point
    '''
    from scipy.spatial import cKDTree
    fiducial, polys, norms = surfs.getVTK(subject, "fiducial")
    xfm, epi = surfs.getXfm(subject, xfmname)
    idx = np.mgrid[:shape[0], :shape[1], :shape[2]].reshape(3, -1).T
    widx = np.append(idx[:,::-1], np.ones((len(idx),1)), axis=-1).T
    mm = np.dot(np.linalg.inv(xfm), widx)[:3].T

    tree = cKDTree(fiducial)
    dist, argdist = tree.query(mm)
    dist.shape = shape
    argdist.shape = shape
    return dist, argdist

def get_roi_mask(subject, xfmname, roi=None, shape=(31, 100, 100)):
    '''Return a bitmask for the given ROIs'''
    import svgroi
    fiducial, polys, norms = surfs.getVTK(subject, "fiducial")
    flat, polys, norms = surfs.getVTK(subject, "flat")
    valid = np.unique(polys)
    flat, fiducial = flat[valid], fiducial[valid]
    
    wpts = np.append(fiducial, np.ones((len(fiducial), 1)), axis=-1).T
    xfm, epi = surfs.getXfm(subject, xfmname)
    coords = np.dot(xfm, wpts)[:3].T.round().astype(int)

    svgfile = os.path.join(options['file_store'], "overlays", "{subj}_rois.svg".format(subj=subject))
    rois = svgroi.ROIpack(flat[:,:2], svgfile)

    #return rois, flat, coords

    if roi is None:
        roi = rois.names

    if isinstance(roi, str):
        mask = np.zeros(shape, dtype=bool)
        mask.T[tuple(coords[rois.get_roi(roi)].T)] = True
        return mask
    elif isinstance(roi, list):
        assert len(roi) < 64, "Too many rois for roimask, select a few"
        idx = dict()
        mask = np.zeros(shape, dtype=np.uint64)
        for i, name in enumerate(roi):
            idx[name] = 1<<i
            mask.T[tuple(coords[rois.get_roi(name)].T)] |= 1<<i
        return mask, idx
