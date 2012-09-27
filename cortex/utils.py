import os
import sys
import binascii
import cStringIO
import numpy as np
from db import surfs

def unmask(mask, data):
    '''unmask(mask, data)

    "Unmasks" the data, assuming it's been masked.

    Parameters
    ----------
    mask : array_like
        The data mask
    data : array_like
        Actual MRI data to unmask
    '''
    assert isinstance(mask, np.ndarray)

    output = np.zeros(mask.shape, dtype=data.dtype)
    output[mask > 0] = data
    return output

def detrend_volume_median(data, kernel=15):
    from scipy.signal import medfilt
    lowfreq = medfilt(data, [1, kernel, kernel])
    return data - lowfreq

def detrend_volume_gradient(data, diff=3):
    return (np.array(np.gradient(data, 1, diff, diff))**2).sum(0)

def detrend_volume_poly(data, polyorder = 10, mask=None):
    from scipy.special import legendre
    polys = [legendre(i) for i in range(polyorder)]
    s = data.shape
    b = data.ravel()[:,np.newaxis]
    lins = np.mgrid[-1:1:s[0]*1j, -1:1:s[1]*1j, -1:1:s[2]*1j].reshape(3,-1)

    if mask is not None:
        lins = lins[:,mask.ravel() > 0]
        b = b[mask.ravel() > 0]
    
    A = np.vstack([[p(i) for i in lins] for p in polys]).T
    x, res, rank, sing = np.linalg.lstsq(A, b)

    detrended = b.ravel() - np.dot(A, x).ravel()
    if mask is not None:
        filled = np.zeros_like(mask)
        filled[mask > 0] = detrended
        return filled
    else:
        return detrended.reshape(*s)


def mosaic(data, xy=(6, 5), trim=10, skip=1, show=True, **kwargs):
    '''mosaic(data, xy=(6, 5), trim=10, skip=1)

    Turns volume data into a mosaic, useful for quickly viewing volumetric data
    IN RADIOLOGICAL COORDINATES (LEFT SIDE OF FIGURE IS RIGHT SIDE OF SUBJECT)

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

def get_roipack(subject, remove_medial=False):
    import svgroi
    flat, polys, norms = surfs.getVTK(subject, "flat", merge=True, nudge=True)
    if remove_medial:
        valid = np.unique(polys)
        flat = flat[valid]
    svgfile = surfs.getFiles(subject)['rois']
    if not os.path.exists(svgfile):
        with open(svgfile, "w") as fp:
            fp.write(svgroi.make_svg(flat.copy(), polys))
    rois = svgroi.ROIpack(flat[:,:2], svgfile)
    if remove_medial:
        return rois, valid
    return rois

def get_ctmpack(subject, xfmname, types=("inflated",), method="raw", level=0, recache=False):
    ctmform = surfs.getFiles(subject)['ctmcache']
    ctmfile = ctmform.format(xfmname=xfmname, types=','.join(types), method=method, level=level)
    if os.path.exists(ctmfile) and not recache:
        return ctmfile

    print "Generating new ctm file..."
    import vtkctm
    return vtkctm.make_pack(ctmfile, subject, xfmname, types, method, level)

def get_cortical_mask(subject, xfmname):
    import nibabel
    shape = nibabel.load(surfs.getXfm(subject, xfmname)[1]).shape[::-1]
    if len(shape) > 3:
        shape = shape[1:]
    
    data = np.zeros(shape, dtype=bool)
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    pts, polys, norms = surfs.getVTK(subject, "flat", merge=True)
    coords = coords[np.unique(polys)]
    d1 = np.logical_and(0 <= coords[:,0], coords[:,0] < shape[2])
    d2 = np.logical_and(0 <= coords[:,1], coords[:,1] < shape[1])
    d3 = np.logical_and(0 <= coords[:,2], coords[:,2] < shape[0])
    valid = np.logical_and(d1, np.logical_and(d2, d3))
    data.T[tuple(coords[valid].T)] = True
    
    return data


def get_vox_dist(subject, xfmname):
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
    import nibabel
    from scipy.spatial import cKDTree
    shape = nibabel.load(surfs.getXfm(subject, xfmname)[1]).shape[::-1]
    if len(shape) > 3:
        shape = shape[1:]

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


def get_hemi_masks(subject, xfmname):
    '''Returns a binary mask of the left and right hemisphere
    surface voxels for the given subject.
    '''
    import nibabel
    shape = nibabel.load(surfs.getXfm(subject, xfmname)[1]).shape[::-1]
    if len(shape) > 3:
        shape = shape[1:]

    lco, rco = surfs.getCoords(subject, xfmname)
    lmask = np.zeros(shape, dtype=np.bool)
    lmask.T[tuple(lco.T)] = True
    rmask = np.zeros(shape, dtype=np.bool)
    rmask.T[tuple(rco.T)] = True
    return lmask, rmask

def add_roi(data, subject, xfmname, name="new_roi", recache=False, **kwargs):
    import subprocess as sp
    from matplotlib.pylab import imsave
    from utils import get_roipack
    import quickflat
    rois = get_roipack(subject)
    im = quickflat.make(data, subject, xfmname, height=1024, recache=recache, with_rois=False)
    fp = cStringIO.StringIO()
    imsave(fp, im, **kwargs)
    fp.seek(0)
    rois.add_roi(name, binascii.b2a_base64(fp.read()))
    return sp.call(["inkscape", '-f', rois.svgfile])

def get_roi_mask(subject, xfmname, roi=None):
    '''Return a bitmask for the given ROIs'''
    import nibabel
    shape = nibabel.load(surfs.getXfm(subject, xfmname)[1]).shape[::-1]
    if len(shape) > 3:
        shape = shape[1:]

    rois, valid = get_roipack(subject, remove_medial=True)
    
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    coords = coords[valid]

    if roi is None:
        roi = rois.names

    if isinstance(roi, str):
        mask = np.zeros(shape, dtype=bool)
        mask.T[tuple(coords[rois.get_roi(roi)].T)] = True
        return mask
    elif isinstance(roi, list):
        roidict = dict()
        for name in roi:
            mask = np.zeros(shape, dtype=np.bool)
            mask.T[tuple(coords[rois.get_roi(name)].T)] = True
            roidict[name] = mask
        return roidict

def get_roi_verts(subject, roi=None):
    '''Return vertices for the given ROIs'''
    rois, valid = get_roipack(subject)

    if roi is None:
        roi = rois.names

    if isinstance(roi, str):
        raise Exception
    elif isinstance(roi, list):
        roidict = dict()
        for name in roi:
            #roidict[name] = tuple(coords[rois.get_roi(name)].T)
            roidict[name] = rois.get_roi(name)
        return roidict


def get_roi_masks(subject,xfmname,roiList=None,Dst=2,overlapOpt='cut'):
    '''
    Return a numbered mask + dictionary of roi numbers
    roiList is a list of ROIs (which better be defined in the .svg file)

    '''
    # Get ROIs from inkscape SVGs
    rois, vertIdx = get_roipack(subject, remove_medial=True)

    # Retrieve shape from the reference
    import nibabel
    shape = nibabel.load(surfs.getXfm(subject, xfmname)[1]).shape[::-1]
    if len(shape) > 3:
        shape = shape[1:]
    
    # Get 3D coords
    coords = np.vstack(surfs.getCoords(subject, xfmname))
    nVerts = np.max(coords.shape)
    coords = coords[vertIdx]
    nValidVerts = np.max(coords.shape)
    # Get voxDst,voxIdx (voxIdx has NOT had invalid 2-D vertices removed by "vertIdx" index)
    voxDst,voxIdx = get_vox_dist(subject,xfmname)
    voxIdxF = voxIdx.flatten()
    # Get L,R hem separately
    L,R = surfs.getVTK(subject, "flat", merge=False, nudge=True)
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

def get_curvature(subject, smooth=8, neighborhood=2):
    from tvtk.api import tvtk
    curvs = []
    for hemi in surfs.getVTK(subject, "fiducial"):
        pd = tvtk.PolyData(points=hemi[0], polys=hemi[1])
        curv = tvtk.Curvatures(input=pd, curvature_type="mean")
        curv.update()
        curv = curv.output.point_data.scalars.to_array()
        if smooth == 0:
            curvs.append(curv)
        else:
            faces = dict()
            for poly in hemi[1]:
                for pt in poly:
                    if pt not in faces:
                        faces[pt] = set()
                    faces[pt] |= set(poly)

            def getpts(pt, n):
                if pt in faces:
                    for p in faces[pt]:
                        if n == 0:
                            yield p
                        else:
                            for q in getpts(p, n-1):
                                yield q

            curvature = np.zeros(len(hemi[0]))
            for i, pt in enumerate(hemi[0]):
                neighbors = list(set(getpts(i, neighborhood)))
                if len(neighbors) > 0:
                    g = np.exp(-(((hemi[0][neighbors] - pt)**2) / (2*smooth**2)).sum(1))
                    curvature[i] = (g * curv[neighbors]).mean()
                
                if i % 1000 == 0:
                    print "\r%d"%i ,
                    sys.stdout.flush()
            curvs.append(curvature)

    return curvs
