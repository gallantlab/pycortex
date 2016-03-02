import io
import os
import binascii
import numpy as np
from importlib import import_module
from .database import db
from .volume import mosaic, unmask, anat2epispace
from .options import config

class DocLoader(object):
    def __init__(self, func, mod, package):
        self._load = lambda: getattr(import_module(mod, package), func)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattribute__(self, name):
        if name != "_load":
            return getattr(self._load(), name)
        else:
            return object.__getattribute__(self, name)


def get_roipack(*args, **kwargs):
    import warnings
    warnings.warn('Please use db.get_overlay instead', DeprecationWarning)
    return db.get_overlay(*args, **kwargs)

get_mapper = DocLoader("get_mapper", ".mapper", "cortex")

def get_ctmpack(subject, types=("inflated",), method="raw", level=0, recache=False,
                decimate=False, disp_layers=['rois'],extra_disp=None):
    """Creates ctm file for the specified input arguments.

    This is a cached file that specifies (1) the surfaces between which
    to interpolate (`types` argument), (2) the `method` to interpolate 
    between surfaces, (3) the display layers to include (rois, sulci, etc)
    """   
    lvlstr = ("%dd" if decimate else "%d")%level
    # Generates different cache files for each combination of disp_layers
    ctmcache = "%s_[{types}]_{method}_{level}_{layers}{extra}.json"%subject
    # Mark any ctm file containing extra_disp as unique (will be over-written every time)
    ctmcache = ctmcache.format(types=','.join(types),
                               method=method,
                               level=lvlstr,
                               layers=repr(sorted(disp_layers)),
                               extra='' if extra_disp is None else '_xx')
    ctmfile = os.path.join(db.get_cache(subject), ctmcache)

    if os.path.exists(ctmfile) and not recache: # and extra_disp is None:
        # (never load cache with extra_disp, which is based on files outside pycortex)
        return ctmfile

    print("Generating new ctm file...")
    from . import brainctm
    ptmap = brainctm.make_pack(ctmfile,
                               subject,
                               types=types,
                               method=method, 
                               level=level,
                               decimate=decimate,
                               disp_layers=disp_layers,
                               extra_disp=extra_disp)
    return ctmfile

def get_ctmmap(subject, **kwargs):
    from scipy.spatial import cKDTree
    from . import brainctm
    jsfile = get_ctmpack(subject, **kwargs)
    ctmfile = os.path.splitext(jsfile)[0]+".ctm"
    
    try:
        left, right = db.get_surf(subject, "pia")
    except IOError:
        left, right = db.get_surf(subject, "fiducial")
    
    lmap, rmap = cKDTree(left[0]), cKDTree(right[0])
    left, right = brainctm.read_pack(ctmfile)
    lnew = lmap.query(left[0])[1]
    rnew = rmap.query(right[0])[1]
    return lnew, rnew

def get_cortical_mask(subject, xfmname, type='nearest'):
    if type == 'cortical':
        ppts, polys = db.get_surf(subject, "pia", merge=True, nudge=False)
        wpts, polys = db.get_surf(subject, "wm", merge=True, nudge=False)
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


def get_vox_dist(subject, xfmname, surface="fiducial", max_dist=np.inf):
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
    max_dist : nonnegative float, optional
        Limit computation to only voxels within `max_dist` mm of the surface.
        Makes computation orders of magnitude faster for high-resolution 
        volumes.

    Returns
    -------
    dist : ndarray
        Distance (in mm) to the closest point on the surface

    argdist : ndarray
        Point index for the closest point
    """
    from scipy.spatial import cKDTree

    fiducial, polys = db.get_surf(subject, surface, merge=True)
    xfm = db.get_xfm(subject, xfmname)
    z, y, x = xfm.shape
    idx = np.mgrid[:x, :y, :z].reshape(3, -1).T
    mm = xfm.inv(idx)

    tree = cKDTree(fiducial)
    dist, argdist = tree.query(mm, distance_upper_bound=max_dist)
    dist.shape = (x,y,z)
    argdist.shape = (x,y,z)
    return dist.T, argdist.T

def get_hemi_masks(subject, xfmname, type='nearest'):
    '''Returns a binary mask of the left and right hemisphere
    surface voxels for the given subject.
    '''
    return get_mapper(subject, xfmname, type=type).hemimasks

def add_roi(data, name="new_roi", open_inkscape=True, add_path=True, **kwargs):
    """Add new flatmap image to the ROI file for a subject.

    (The subject is specified in creation of the data object)

    Creates a flatmap image from the `data` input, and adds that image as
    a sub-layer to the data layer in the rois.svg file stored for 
    the subject  in the pycortex database. Most often, this is data to be 
    used for defining a region (or several regions) of interest, such as a 
    localizer contrast (e.g. a t map of Faces > Houses). 

    Use the **kwargs inputs to specify 

    Parameters
    ----------
    data : DataView
        The data used to generate the flatmap image. 
    name : str, optional
        Name that will be assigned to the `data` sub-layer in the rois.svg file
            (e.g. 'Faces > Houses, t map, p<.005' or 'Retinotopy - Rotating Wedge')
    open_inkscape : bool, optional
        If True, Inkscape will automatically open the ROI file.
    add_path : bool, optional
        If True, also adds a sub-layer to the `rois` new SVG layer will automatically
        be created in the ROI group with the same `name` as the overlay.
    kwargs : dict
        Passed to cortex.quickflat.make_png
    """
    import subprocess as sp
    from . import quickflat
    from . import dataset

    dv = dataset.normalize(data)
    if isinstance(dv, dataset.Dataset):
        raise TypeError("Please specify a data view")

    rois = db.get_overlay(dv.subject)
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()

    quickflat.make_png(fp, dv, height=1024, with_rois=False, with_labels=False, **kwargs)
    fp.seek(0)
    rois.add_roi(name, binascii.b2a_base64(fp.read()), add_path)
    
    if open_inkscape:
        return sp.call(["inkscape", '-f', rois.svgfile])

def get_roi_verts(subject, roi=None):
    """Return vertices for the given ROIs, or all ROIs if none are given.

    Parameters
    ----------
    subject : str
        Name of the subject
    roi : str, list or None, optional
        ROIs to fetch. Can be ROI name (string), a list of ROI names, or
        None, in which case all ROIs will be fetched.

    Returns
    -------
    roidict : dict
        Dictionary of {roi name : roi verts}. ROI verts are for both
        hemispheres, with right hemisphere vertex numbers sequential
        after left hemisphere vertex numbers.
    """
    # Get ROIpack
    rois = db.get_overlay(subject)

    # Get flat surface so we can figure out which verts are in medial wall
    # or in cuts
    # This assumes subject has flat surface, which they must to have ROIs..
    pts, polys = db.get_surf(subject, "flat", merge=True)
    goodpts = np.unique(polys)

    if roi is None:
        roi = rois.names

    roidict = dict()
    if isinstance(roi, str):
        roi = [roi]

    for name in roi:
        roidict[name] = np.intersect1d(rois.get_roi(name), goodpts)

    return roidict

def get_roi_mask(subject, xfmname, roi=None, projection='nearest'):
    '''Return a bitmask for the given ROI'''

    mapper = get_mapper(subject, xfmname, type=projection)
    rois = get_roi_verts(subject, roi=roi)
    output = dict()
    for name, verts in list(rois.items()):
        left, right = mapper.backwards(verts)
        output[name] = left + right
        
    return output

def get_aseg_mask(subject, xfmname, aseg_id, **kwargs):
    """Return an epi space mask of the given ID from freesurfer's automatic segmentation

    For aseg_id's, see https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """
    aseg = db.get_anat(subject, type="aseg").get_data().T

    if isinstance(aseg_id, (list, tuple)):
        mask = np.zeros(aseg.shape)
        for idx in aseg_id:
            mask = np.logical_or(mask, aseg == idx)
    else:
        mask = aseg == aseg_id
    return anat2epispace(mask.astype(float), subject, xfmname, **kwargs)

def get_roi_masks(subject,xfmname,roi_list=None,dst=2,fail_for_missing_rois=False):
    '''Return a numbered mask + dictionary of roi numbers

    This function returns a single 3D array with a separate numerical index for each ROI, 

    Parameters
    ----------
    subject : string
        pycortex subject ID
    xfmname : string
        pycortex transformation name
    roi_list : list or None
        list of names of ROIs to retrieve (e.g. ['FFA','OFA','EBA']). Names should match the 
        ROI layers in the rois.svg file for the `subject` specified. If None is provided (default),
        all available ROIs for the subject are returned.
    dst : scalar or string
        Distance from fiducial surface to define ROI. Reasonable values are ~1 (very conservative)
        to ~3 (lots of brain that isn't cortex). Alternatively, you can specify one of the available 
        masks as the dst value ('thick','thin', or 'cortical'). 'cortical' will only work if the 
        surface for `subject` was created using Freesurfer.
    fail_for_missing_rois : bool
        Whether to fail if one or more of the rois specified in roi_list are not available. If set to
        False (default behavior), requested rois that are not present will be ignored (and not included
        in the outputs, obviously)
    
    Returns
    -------
    mask : array
        Single 3D array with a separate numerical index value for each ROI. ROI index values in the 
        left hemisphere are negative. (For example, if V1 in the right hemisphere is 1, then V1 in 
        the left hemisphere will be -1). 
    roi_index : dict
        Mapping of roi names to index values (e.g. {'V1': 1}). 
    '''
    # Get ROIs from inkscape SVGs
    rois, vert_idx = db.get_overlay(subject, remove_medial=True)

    # Retrieve shape from the reference
    shape = db.get_xfm(subject, xfmname).shape
    
    # Get 3D coordinates
    coords = np.vstack(db.get_coords(subject, xfmname)) # UGH. Replace with a mapper object (IDK what this comment is...)
    n_verts = np.max(coords.shape)
    coords = coords[vert_idx]
    n_valid_vertices = np.max(coords.shape)
    # Get distance of each voxel from fiducial surface (vox_dst) and index for each voxel showing 
    # which surface vertex is closest to it (vox_idx). Note that vox_idx includes vertices on the 
    # medial wall, which later need to be excluded from the ROIs.
    vox_dst,vox_idx = get_vox_dist(subject,xfmname)
    vox_idx_flat = vox_idx.flatten()
    # Get L,R hem separately
    L,R = db.get_surf(subject, "flat", merge=False, nudge=True)
    nL = len(np.unique(L[1]))
    # Mask for left hemisphere
    Lmask = (vox_idx < nL).flatten()
    Rmask = np.logical_not(Lmask)
    if type(dst) in (str,unicode):
        cx_mask = db.get_mask(subject,xfmname,dst).flatten()
    else:
        cx_mask = (vox_dst < dst).flatten()
    
    if roi_list is None:
        roi_list = rois.names
    else:
        roi_list = [r for r in roi_list if r in ['Cortex','cortex']+rois.names]
        if fail_for_missing_rois:
            fails = [r for r in roi_list if not r in ['Cortex','cortex']+rois.names]
            if any(fails):
                for f in fails:
                    print("No ROI exists for %s"%f)
                raise ValueError("Invalid ROIs requested!")

    if isinstance(roi_list, str):
        roi_list = [roi_list]
    # First: get all roi voxels into 4D volume
    tmp_mask = np.zeros((np.prod(shape),len(roi_list),2),np.bool)
    drop_roi = []
    for ir,roi in enumerate(roi_list):
        if roi.lower()=='cortex':
            roi_idx_bin3 = np.ones(Lmask.shape)>0
        else:
            # Complicated indexing to remove vertices/voxels on the medial wall
            roi_idx_bin1 = np.zeros((n_valid_vertices,),np.bool) # binary index 1
            # This step is the time-consuming step, which can't be avoided.
            roi_idx_sub1 = rois.get_roi(roi) # substitution index 1 (in valid vertex space)
            roi_idx_bin1[roi_idx_sub1] = True
            roi_idx_bin2 = np.zeros((n_verts,),np.bool) # binary index 2
            roi_idx_bin2[vert_idx] = roi_idx_bin1
            roi_idx_sub2 = np.nonzero(roi_idx_bin2)[0] # substitution index 2 (in ALL fiducial vertex space)
            roi_idx_bin3 = np.in1d(vox_idx_flat,roi_idx_sub2) # binary index to 3D volume (flattened, though)
        tmp_mask[:,ir,0] = np.all(np.array([roi_idx_bin3,Lmask,cx_mask]),axis=0)
        tmp_mask[:,ir,1] = np.all(np.array([roi_idx_bin3,Rmask,cx_mask]),axis=0)
        if not np.any(tmp_mask[:,ir]):
            drop_roi += [ir]
    # Cull rois with no voxels
    keep_roi = np.array([not ir in drop_roi for ir in range(len(roi_list))],dtype=np.bool)
    # Cull rois requested, but not avialable in pycortex
    roi_list_L = [r for ir,r in enumerate(roi_list) if not ir in drop_roi]
    tmp_mask = tmp_mask[:,keep_roi,:]
    # Kill all overlap btw. "cortex" and other ROIs
    roi_list_L_lower = [xx.lower() for xx in roi_list_L]
    if 'cortex' in roi_list_L_lower:
        cx_idx = roi_list_L_lower.index('cortex')
        # Left:
        other_rois = tmp_mask[:,np.arange(len(roi_list_L))!=cx_idx,0] 
        tmp_mask[:,cx_idx,0] = np.logical_and(np.logical_not(np.any(other_rois,axis=1)),tmp_mask[:,cx_idx,0])
        # Right:
        other_rois = tmp_mask[:,np.arange(len(roi_list_L))!=cx_idx,1]
        tmp_mask[:,cx_idx,1] = np.logical_and(np.logical_not(np.any(other_rois,axis=1)),tmp_mask[:,cx_idx,1])

    # Second: 
    mask = np.zeros(np.prod(shape),dtype=np.int64)
    roi_idx = {}
    to_cut = np.sum(tmp_mask,axis=1)>1
    # Note that indexing by vox_idx guarantees that there will be no overlap in ROIs
    # (unless there are overlapping assignments to ROIs on the surface), due to 
    # each voxel being assigned only ONE closest vertex
    print('%d overlapping voxels were cut'%np.sum(to_cut))
    tmp_mask[to_cut] = False 
    for ir,roi in enumerate(roi_list_L):
        mask[tmp_mask[:,ir,0]] = -ir-1
        mask[tmp_mask[:,ir,1]] = ir+1
        roi_idx[roi] = ir+1
    mask.shape = shape

    return mask,roi_idx

def get_dropout(subject, xfmname, power=20):
    """Create a dropout Volume showing where EPI signal
    is very low.
    """
    xfm = db.get_xfm(subject, xfmname)
    rawdata = xfm.reference.get_data().T

    ## Collapse epi across time if it's 4D
    if rawdata.ndim > 3:
        rawdata = rawdata.mean(0)

    rawdata[rawdata==0] = np.mean(rawdata[rawdata!=0])
    normdata = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min())
    normdata = (1 - normdata) ** power

    from .dataset import Volume
    return Volume(normdata, subject, xfmname)

def make_movie(stim, outfile, fps=15, size="640x480"):
    import shlex
    import subprocess as sp
    cmd = "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    fcmd = cmd.format(infile=stim, size=size, fps=fps, outfile=outfile)
    sp.call(shlex.split(fcmd))

def vertex_to_voxel(subject):
    max_thickness = db.get_surfinfo(subject, "thickness").data.max()

    # Get distance from each voxel to each vertex on each surface
    fid_dist, fid_verts = get_vox_dist(subject, "identity", "fiducial", max_thickness)
    wm_dist, wm_verts = get_vox_dist(subject, "identity", "wm", max_thickness)
    pia_dist, pia_verts = get_vox_dist(subject, "identity", "pia", max_thickness)

    # Get nearest vertex on any surface for each voxel
    all_dist, all_verts = fid_dist, fid_verts
    
    wm_closer = wm_dist < all_dist
    all_dist[wm_closer] = wm_dist[wm_closer]
    all_verts[wm_closer] = wm_verts[wm_closer]

    pia_closer = pia_dist < all_dist
    all_dist[pia_closer] = pia_dist[pia_closer]
    all_verts[pia_closer] = pia_verts[pia_closer]

    return all_verts

def get_cmap(name):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    # unknown colormap, test whether it's in pycortex colormaps
    cmapdir = config.get('webgl', 'colormaps')
    colormaps = os.listdir(cmapdir)
    colormaps = sorted([c for c in colormaps if '.png' in c])
    colormaps = dict((c[:-4], os.path.join(cmapdir, c)) for c in colormaps)
    if name in colormaps:
        I = plt.imread(colormaps[name])
        cmap = colors.ListedColormap(np.squeeze(I))
        plt.cm.register_cmap(name,cmap)
    else:
        try: 
            cmap = plt.cm.get_cmap(name)
        except:
            raise Exception('Unkown color map!')
    return cmap