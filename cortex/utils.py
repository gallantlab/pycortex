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

def get_ctmpack(subject, types=("inflated",), method="raw", level=0, recache=False, decimated=False):
    ctmform = surfs.getFiles(subject)['ctmcache']
    lvlstr = ("%dd" if decimated else "%d")%level
    ctmfile = ctmform.format(types=','.join(types), method=method, level=lvlstr)
    if os.path.exists(ctmfile) and not recache:
        return ctmfile

    print("Generating new ctm file...")
    from . import brainctm
    ptmap = brainctm.make_pack(ctmfile, subject, types=types, method=method, level=level, decimate=decimated)
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

def add_roi(data, name="new_roi", recache=False, open_inkscape=True, add_path=True, **kwargs):
    import subprocess as sp
    from matplotlib.pylab import imsave
    from .utils import get_roipack
    from . import quickflat
    rois = get_roipack(data.subject)
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()
    quickflat.make_png(fp, data, height=1024, recache=recache, with_rois=False, with_labels=False, **kwargs)
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
    roiList is a list of ROIs (which better be defined in the .svg file)
    poop.
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
    CxMask = (voxDst < Dst).flatten()
    
    #return rois, flat, coords, voxDst, voxIdx ## rois is a list of class svgROI; flat = flat cortex coords; coords = 3D coords
    if roiList is None:
        roiList = rois.names

    if isinstance(roiList, str):
        roiList = [roiList]
    # First: get all roi voxels into 4D volume
    tmpMask = np.zeros((np.prod(shape),len(roiList),2),np.bool)
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
    roiListL = [r.lower() for r in roiList]
    # Kill all overlap btw. "Cortex" and other ROIs
    if 'cortex' in roiListL:
        cIdx = roiListL.index('cortex')
        # Left:
        OtherROIs = tmpMask[:,np.arange(len(roiList))!=cIdx,0] 
        tmpMask[:,cIdx,0] = np.logical_and(np.logical_not(np.any(OtherROIs,axis=1)),tmpMask[:,cIdx,0])
        # Right:
        OtherROIs = tmpMask[:,np.arange(len(roiList))!=cIdx,1]
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
        for ir,roi in enumerate(roiList):
            mask[tmpMask[:,ir,0]] = -ir-1
            mask[tmpMask[:,ir,1]] = ir+1
            roiIdx[roi] = ir+1
        mask.shape = shape
    elif overlapOpt=='split':
        pass
    return mask,roiIdx

def get_curvature(subject, smooth=20.0, **kwargs):
    from . import polyutils
    curvs = []
    for pts, polys in surfs.getSurf(subject, "fiducial"):
        surf = polyutils.Surface(pts, polys)
        curv = surf.smooth(surf.mean_curvature(), smooth)
        curvs.append(curv)
    return curvs


def get_flatmap_distortion(sub, type="areal", smooth=20.0):
    """Computes distortion of flatmap relative to fiducial surface. Several different
    types of distortion are available:
    
    'areal': computes the areal distortion for each triangle in the flatmap, defined as the
    log ratio of the area in the fiducial mesh to the area in the flat mesh. Returns
    a per-vertex value that is the average of the neighboring triangles.
    See: http://brainvis.wustl.edu/wiki/index.php/Caret:Operations/Morphing
    
    'metric': computes the linear distortion for each vertex in the flatmap, defined as
    the mean squared difference between distances in the fiducial map and distances in
    the flatmap, for each pair of neighboring vertices. See Fishl, Sereno, and Dale, 1999.
    """
    from polyutils import Distortion, Surface
    distortions = []
    for hem in ["lh", "rh"]:
        fidvert, fidtri = surfs.getSurf(sub, "fiducial", hem)
        flatvert, flattri = surfs.getSurf(sub, "flat", hem)
        surf = Surface(fidvert, fidtri)

        dist = getattr(Distortion(flatvert, fidvert, flattri), type)
        smdist = surf.smooth(dist, smooth)
        distortions.append(smdist)

    return distortions

def get_tissots_indicatrix(sub, radius=10, spacing=50, maxfails=100):
    from . import polyutils
    
    tissots = []
    allcenters = []
    for hem in ["lh", "rh"]:
        fidpts, fidpolys = surfs.getSurf(sub, "fiducial", hem)
        #G = make_surface_graph(fidtri)
        surf = polyutils.Surface(fidpts, fidpolys)
        nvert = fidpts.shape[0]
        tissot_array = np.zeros((nvert,))

        centers = [np.random.randint(nvert)]
        cdists = [surf.geodesic_distance(centers)]
        while True:
            ## Find possible vertices
            mcdist = np.vstack(cdists).min(0)
            possverts = np.nonzero(mcdist > spacing)[0]
            #possverts = np.nonzero(surf.geodesic_distance(centers) > spacing)[0]
            if not len(possverts):
                break
            ## Pick random vertex
            centervert = possverts[np.random.randint(len(possverts))]
            centers.append(centervert)
            print("Adding vertex %d.." % centervert)
            dists = surf.geodesic_distance([centervert])
            cdists.append(dists)

            ## Find appropriate set of vertices
            selverts = dists < radius
            tissot_array[selverts] = 1

        tissots.append(tissot_array)
        allcenters.append(np.array(centers))

    return tissots, allcenters

def get_dropout(subject, xfmname, power=20):
    """Returns a dropout VolumeData showing where EPI signal
    is very low."""
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
