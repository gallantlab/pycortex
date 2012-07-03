import os
import json
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
options = json.load(open(os.path.join(cwd, "defaults.json")))

from db import surfs
import view
from utils import unmask

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
    data = np.zeros(shape, dtype=bool)
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    pts, polys, norms = surfs.getVTK(subject, "flat", merge=True)
    coords = coords[np.unique(polys)]
    data.T[tuple(coords.T)] = True
    
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
    fiducial, polys, norms = surfs.getVTK(subject, "fiducial", merge=True)
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
    flat, polys, norms = surfs.getVTK(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    flat = flat[valid]
    
    coords = np.vstack(db.surfs.getCoords(subject, xfmname))
    coords = coords[valid]

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

def get_roi_masks(subject,xfmname,roiList=None,shape=(31,100,100),Dst=2,overlapOpt='cut'):
    '''
    Return a numbered mask + dictionary of roi numbers
    roiList is a list of ROIs (which better be defined in the .svg file)

    '''
    import svgroi
    # merge = True returns LEFT HEM as first (nL) values, then RIGHT
    flat, polys, norms = surfs.getVTK(subject, "flat", merge=True, nudge=True)
    vertIdx = np.unique(polys) # Needed for determining index for ROIs later
    flat = flat[vertIdx]
    # Get 3D coords
    coords = np.vstack(db.surfs.getCoords(subject, xfmname))
    nVerts = np.max(coords.shape)
    coords = coords[vertIdx]
    nValidVerts = np.max(coords.shape)
    # Get voxDst,voxIdx (voxIdx has NOT had invalid 2-D vertices removed by "vertIdx" index)
    voxDst,voxIdx = get_vox_dist(subject,xfmname,shape)
    voxIdxF = voxIdx.flatten()
    # Get L,R hem separately
    L,R = surfs.getVTK(subject, "flat", merge=False, nudge=True)
    nL = len(np.unique(L[1]))
    #nVerts = len(idxL)+len(idxR)
    # mask for left hemisphere
    Lmask = (voxIdx < nL).flatten()
    Rmask = np.logical_not(Lmask)
    CxMask = (voxDst < Dst).flatten()
    # Get ROIs from inkscape SVGs
    svgfile = os.path.join(options['file_store'], "overlays", "{subj}_rois.svg".format(subj=subject))
    rois = svgroi.ROIpack(flat[:,:2], svgfile)
    
    #return rois, flat, coords, voxDst, voxIdx ## rois is a list of class svgROI; flat = flat cortex coords; coords = 3D coords
    if roiList is None:
        roiList = rois.names

    if isinstance(roiList, str):
        roiList = [roiList]
    # First: get all roi voxels into 4D volume
    tmpMask = np.zeros((np.prod(shape),len(roiList),2),np.bool)
    for ir,roi in enumerate(roiList):
        # Irritating index switching:
        roiIdxB1 = np.zeros((nValidVerts,),np.bool) # binary index 1
        roiIdxS1 = rois.get_roi(roi) # substitution index 1 (in valid vertex space)
        roiIdxB1[roiIdxS1] = True
        roiIdxB2 = np.zeros((nVerts,),np.bool) # binary index 2
        roiIdxB2[vertIdx] = roiIdxB1
        roiIdxS2 = np.nonzero(roiIdxB2)[0] # substitution index 2 (in ALL fiducial vertex space)
        roiIdxB3 = np.in1d(voxIdxF,roiIdxS2) # binary index to 3D volume (flattened, though)
        tmpMask[:,ir,0] = np.all(np.array([roiIdxB3,Lmask,CxMask]),axis=0)
        tmpMask[:,ir,1] = np.all(np.array([roiIdxB3,Rmask,CxMask]),axis=0)
    
    # cover 
    mask = np.zeros(np.prod(shape),dtype=np.int64)
    roiIdx = {}
    if overlapOpt=='cut':
        toCut = np.sum(tmpMask,axis=1)>1
        # Note that indexing by voxIdx guarantees that there will be no overlap in ROIs
        # (unless there are overlapping assignments to ROIs on the surface), due to 
        # each voxel being assigned only ONE closest vertex
        print('%d voxels cut'%np.sum(toCut))
        tmpMask[toCut] = False 
        for ir,roi in enumerate(roiList):
            mask[tmpMask[:,ir,0]] = -ir-1
            mask[tmpMask[:,ir,1]] = ir+1
            roiIdx[roi] = ir+1
        mask.shape = shape
    elif overlapOpt=='split':
        pass
    return mask,roiIdx