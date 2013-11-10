import io
import os
import sys
import binascii
import numpy as np

from .db import surfs
from .volume import mosaic, unmask

def get_roipack(*args, **kwargs):
    import warnings
    warnings.warn('Please use surfs.getOverlay instead', DeprecationWarning)
    return surfs.getOverlay(*args, **kwargs)

def get_mapper(*args, **kwargs):
    from .mapper import get_mapper
    return get_mapper(*args, **kwargs)

def get_ctmpack(subject, types=("inflated",), method="raw", level=0, recache=False, decimate=False):
    ctmcache = "%s_[{types}]_{method}_{level}.json"%subject
    ctmform = os.path.join(surfs.getCache(subject), ctmcache)
    
    lvlstr = ("%dd" if decimate else "%d")%level
    ctmfile = ctmform.format(types=','.join(types), method=method, level=lvlstr)
    if os.path.exists(ctmfile) and not recache:
        return ctmfile

    print("Generating new ctm file...")
    from . import brainctm
    ptmap = brainctm.make_pack(ctmfile, subject, types=types, method=method, level=level, decimate=decimate)
    return ctmfile

def get_cortical_mask(subject, xfmname, type='nearest'):
    from .db import surfs
    if type == 'cortical':
        ppts, polys = surfs.getSurf(subject, "pia", merge=True, nudge=False)
        wpts, polys = surfs.getSurf(subject, "wm", merge=True, nudge=False)
        thickness = np.sqrt(((ppts - wpts)**2).sum(1))

        dist, idx = get_vox_dist(subject, xfmname)
        cortex = np.zeros(dist.shape, dtype=bool)
        verts = np.unique(idx)
        for i, vert in enumerate(verts):
            mask = idx == vert
            cortex[mask] = dist[mask] <= thickness[vert]
            if i % 100 == 0:
                print("%0.3f%%"%(i/float(len(verts)) * 100))
        return cortex
    elif type in ('thick', 'thin'):
        dist, idx = get_vox_dist(subject, xfmname)
        return dist < dict(thick=8, thin=2)[type]
    else:
        return get_mapper(subject, xfmname, type=type).mask


def get_vox_dist(subject, xfmname, surface="fiducial"):
    """Get the distance (in mm) from each functional voxel to the closest
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
    """
    import nibabel
    from scipy.spatial import cKDTree

    fiducial, polys = surfs.getSurf(subject, surface, merge=True)
    xfm = surfs.getXfm(subject, xfmname)
    z, y, x = xfm.shape
    idx = np.mgrid[:x, :y, :z].reshape(3, -1).T
    mm = xfm.inv(idx)

    tree = cKDTree(fiducial)
    dist, argdist = tree.query(mm)
    dist.shape = (x,y,z)
    argdist.shape = (x,y,z)
    return dist.T, argdist.T

def get_hemi_masks(subject, xfmname, type='nearest'):
    '''Returns a binary mask of the left and right hemisphere
    surface voxels for the given subject.
    '''
    return get_mapper(subject, xfmname, type=type).hemimasks

def add_roi(data, name="new_roi", open_inkscape=True, add_path=True, **kwargs):
    """Add new overlay data to the ROI file for a subject.

    Parameters
    ----------
    data : BrainData
        The data that will be overlaid on the ROI file.
    name : str, optional
        Name that will be assigned to the new dataset. <<IS THIS NECESSARY ANYMORE?>>
    open_inkscape : bool, optional
        If True, Inkscape will automatically open the ROI file.
    add_path : bool, optional
        If True, a new SVG layer will automatically be created in the ROI group
        with the same `name` as the overlay.
    kwargs : dict
        Passed to cortex.quickflat.make_png
    """
    import subprocess as sp
    from .utils import get_roipack
    from . import quickflat
    # May require more flexible code to deal with other type of datasets (vertex,etc)
    rois = get_roipack(data.data.subject)
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()
    quickflat.make_png(fp, data, height=1024, with_rois=False, with_labels=False, **kwargs)
    fp.seek(0)
    rois.add_roi(name, binascii.b2a_base64(fp.read()), add_path)
    if open_inkscape:
        return sp.call(["inkscape", '-f', rois.svgfile])

def get_roi_verts(subject, roi=None):
    '''Return vertices for the given ROIs'''
    rois = surfs.getOverlay(subject)

    if roi is None:
        roi = rois.names

    roidict = dict()
    if isinstance(roi, str):
        roi = [roi]

    for name in roi:
        roidict[name] = rois.get_roi(name)

    return roidict

def get_roi_mask(subject, xfmname, roi=None, projection='nearest'):
    '''Return a bitmask for the given ROIs'''

    mapper = get_mapper(subject, xfmname, type=projection)
    rois = get_roi_verts(subject, roi=roi)
    output = dict()
    for name, verts in list(rois.items()):
        left, right = mapper.backwards(verts)
        output[name] = left + right
        
    return output

def get_roi_masks(subject,xfmname,roiList=None,Dst=2,overlapOpt='cut'):
    '''
    Return a numbered mask + dictionary of roi numbers
    roiList is a list of ROIs (which should be defined in the .svg file)
    '''
    # Get ROIs from inkscape SVGs
    rois, vertIdx = get_roipack(subject, remove_medial=True)

    # Retrieve shape from the reference
    import nibabel
    shape = surfs.getXfm(subject, xfmname).shape
    
    # Get 3D coords
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    nVerts = np.max(coords.shape)
    coords = coords[vertIdx]
    nValidVerts = np.max(coords.shape)
    # Get voxDst,voxIdx (voxIdx has NOT had invalid 2-D vertices removed by "vertIdx" index)
    voxDst,voxIdx = get_vox_dist(subject,xfmname)
    voxIdxF = voxIdx.flatten()
    # Get L,R hem separately
    L,R = surfs.getSurf(subject, "flat", merge=False, nudge=True)
    nL = len(np.unique(L[1]))
    #nVerts = len(idxL)+len(idxR)
    # mask for left hemisphere
    Lmask = (voxIdx < nL).flatten()
    Rmask = np.logical_not(Lmask)
    if type(Dst) in (str,unicode) and Dst.lower()=='cortical':
        CxMask = surfs.getMask(subject,xfmname,'cortical').flatten()
    else:
        CxMask = (voxDst < Dst).flatten()
    
    #return rois, flat, coords, voxDst, voxIdx ## rois is a list of class svgROI; flat = flat cortex coords; coords = 3D coords
    if roiList is None:
        roiList = rois.names
    else:
        roiList = [r for r in roiList if r in ['Cortex','cortex']+rois.names]

    if isinstance(roiList, str):
        roiList = [roiList]
    # First: get all roi voxels into 4D volume
    tmpMask = np.zeros((np.prod(shape),len(roiList),2),np.bool)
    dropROI = []
    for ir,roi in enumerate(roiList):
        if roi.lower()=='cortex':
            roiIdxB3 = np.ones(Lmask.shape)>0
        else:
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
        if not np.any(tmpMask[:,ir]):
            dropROI += [ir]
    # Cull rois with no voxels
    keepROI = np.array([not ir in dropROI for ir in range(len(roiList))])
    # Cull rois requested, but not avialable in pycortex
    roiListL = [r for ir,r in enumerate(roiList) if not ir in dropROI]
    tmpMask = tmpMask[:,keepROI,:]
    # Kill all overlap btw. "Cortex" and other ROIs
    roiListL_lower = [xx.lower() for xx in roiListL]
    if 'cortex' in roiListL_lower:
        cIdx = roiListL_lower.index('cortex')
        # Left:
        OtherROIs = tmpMask[:,np.arange(len(roiListL))!=cIdx,0] 
        tmpMask[:,cIdx,0] = np.logical_and(np.logical_not(np.any(OtherROIs,axis=1)),tmpMask[:,cIdx,0])
        # Right:
        OtherROIs = tmpMask[:,np.arange(len(roiListL))!=cIdx,1]
        tmpMask[:,cIdx,1] = np.logical_and(np.logical_not(np.any(OtherROIs,axis=1)),tmpMask[:,cIdx,1])

    # Second: 
    mask = np.zeros(np.prod(shape),dtype=np.int64)
    roiIdx = {}
    if overlapOpt=='cut':
        toCut = np.sum(tmpMask,axis=1)>1
        # Note that indexing by voxIdx guarantees that there will be no overlap in ROIs
        # (unless there are overlapping assignments to ROIs on the surface), due to 
        # each voxel being assigned only ONE closest vertex
        print(('%d voxels cut'%np.sum(toCut)))
        tmpMask[toCut] = False 
        for ir,roi in enumerate(roiListL):
            mask[tmpMask[:,ir,0]] = -ir-1
            mask[tmpMask[:,ir,1]] = ir+1
            roiIdx[roi] = ir+1
        mask.shape = shape
    elif overlapOpt=='split':
        pass
    return mask,roiIdx

def get_dropout(subject, xfmname, power=20):
    """Create a dropout VolumeData showing where EPI signal
    is very low.
    """
    xfm = surfs.getXfm(subject, xfmname)
    rawdata = xfm.reference.get_data().T

    ## Collapse epi across time if it's 4D
    if rawdata.ndim > 3:
        rawdata = rawdata.mean(0)

    normdata = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min())
    normdata = (1 - normdata) ** power

    from .dataset import VolumeData
    return VolumeData(normdata, subject, xfmname)

def make_movie(stim, outfile, fps=15, size="640x480"):
    import shlex
    import subprocess as sp
    cmd = "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    fcmd = cmd.format(infile=stim, size=size, fps=fps, outfile=outfile)
    sp.call(shlex.split(fcmd))
