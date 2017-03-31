"""Contain utility functions
"""
import io
import os
import binascii
import numpy as np
from six import string_types
from importlib import import_module
from .database import db
from .volume import anat2epispace
from .options import config
from .freesurfer import fs_aseg_dict

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
                decimate=False):
    """Creates ctm file for the specified input arguments.

    This is a cached file that specifies (1) the surfaces between which
    to interpolate (`types` argument), (2) the `method` to interpolate 
    between surfaces
    
    Parameters
    ----------
    subject : str
        Name of subject in pycortex stored
    types : tuple
        Surfaces between which to interpolate.
    method : str
        
    level : 
    
    recache : bool
        Recache intermediate files? Can resolve some errors but is slower.
    
    decimate : bool
    
    Returns
    -------
    ctmfile :
    """
    lvlstr = ("%dd" if decimate else "%d")%level
    # Generates different cache files for each combination of disp_layers
    ctmcache = "%s_[{types}]_{method}_{level}_v3.json"%subject
    ctmcache = ctmcache.format(types=','.join(types),
                               method=method,
                               level=lvlstr)
    ctmfile = os.path.join(db.get_cache(subject), ctmcache)

    if os.path.exists(ctmfile) and not recache: 
        return ctmfile

    print("Generating new ctm file...")
    from . import brainctm
    ptmap = brainctm.make_pack(ctmfile,
                               subject,
                               types=types,
                               method=method, 
                               level=level,
                               decimate=decimate)
    return ctmfile

def get_ctmmap(subject, **kwargs):
    """
    Parameters
    ----------
    subject : str
        Subject name
    
    Returns
    -------
    lnew :
    
    rnew :
    """
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
    """Gets the cortical mask for a particular transform
    
    Parameters
    ----------
    subject : str
        Subject name
    xfmname : str
        Transform name
    type : str
        Mask type, one of {'cortical','thin','thick', 'nearest'}. 'cortical' is exactly the 
        cortical ribbon, between the freesurfer-estimated white matter and pial 
        surfaces; 'thin' is < 2mm away from fiducial surface; 'thick' is < 8mm 
        away from fiducial surface. 
        'nearest' is nearest voxel only (??)
        
    Returns
    -------
    mask : array
        boolean mask array for cortical voxels in functional space
    """
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
    
    Parameters
    ----------
    subject : str
        Name of subject
    xfmname : str
        Name of transform
    type : str
    
    Returns
    -------
    
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

    svg = db.get_overlay(dv.subject)
    fp = io.BytesIO()

    quickflat.make_png(fp, dv, height=1024, with_rois=False, with_labels=False, **kwargs)
    fp.seek(0)
    svg.rois.add_shape(name, binascii.b2a_base64(fp.read()).decode('utf-8'), add_path)
    
    if open_inkscape:
        return sp.call(["inkscape", '-f', svg.svgfile])

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
    # Get overlays
    svg = db.get_overlay(subject)

    # Get flat surface so we can figure out which verts are in medial wall
    # or in cuts
    # This assumes subject has flat surface, which they must to have ROIs.
    pts, polys = db.get_surf(subject, "flat", merge=True)
    goodpts = np.unique(polys)

    if roi is None:
        roi = svg.rois.shapes.keys()

    roidict = dict()
    if isinstance(roi, string_types):
        roi = [roi]

    for name in roi:
        roidict[name] = np.intersect1d(svg.rois.get_mask(name), goodpts)

    return roidict

def get_roi_mask(subject, xfmname, roi=None, projection='nearest'):
    """Return a mask for the given ROI(s)
    
    Deprecated - use get_roi_masks()

    Parameters
    ----------
    subject : str
        Name of subject
    xfmname : str
        Name of transform
    roi : tuple
        Name of ROI(s) to get masks for. None gets all of them.
    projection : str
        Which mapper to use.    
    Returns
    -------
    output : dict
        Dict of ROIs and their masks
    """
    import warnings
    warnings.DeprecationWarning('Deprecated! Use get_roi_mask')

    mapper = get_mapper(subject, xfmname, type=projection)
    rois = get_roi_verts(subject, roi=roi)
    output = dict()
    for name, verts in list(rois.items()):
        left, right = mapper.backwards(verts)
        output[name] = left + right
        
    return output

def get_aseg_mask(subject, aseg_name, xfmname=None, order=1, threshold=None, **kwargs):
    """Return an epi space mask of the given ID from freesurfer's automatic segmentation
    
    Parameters
    ----------
    subject : str
        pycortex subject ID
    aseg_name : str or list
        Name of brain partition or partitions to return. See freesurfer web site for partition names:
        https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
        ... or inspect `cortex.freesurfer.fs_aseg_mask.keys()` Currently (2017.03) only the first
        256 indices in the freesurfer lookup table are supported. If a name is provided that does not 
        exactly match any of the freesurfer partitions, the function will search for all partitions
        that contain that name (caps are ignored). For example, 'white-matter' will generate a mask 
        that combines masks for the following partitions: 'Right-Cerebral-White-Matter', 
        'Left-Cerebellum-White-Matter', 'Right-Cerebellum-White-Matter', and 'Left-Cerebral-White-Matter')
    xfmname : str
        Name for transform of mask to functional space. If `None`, anatomical-space
        mask is returned.
    order : int, [0-5]
        Order of spline interpolation for transform from anatomical to functional space
        (ignored if xfmname is None). 0 is like nearest neighbor; 1 returns bilinear
        interpolation of mask from anatomical space. To convert either of these volumes to
        a binary mask for voxel selection, set the `threshold` argument.
        Setting order > 1 is not recommended, as it will give values outside the range of 0-1.
    threshold : scalar
        Threshold value for aseg mask. If None, function returns result of spline 
        interpolation of mask as transformed to functional space (will have continuous 
        float values from 0-1)
    
    Returns
    -------
    mask : array
        array with float or boolean values denoting the location of the requested cortical 
        partition.

    Notes
    -----
    See also get_anat(subject, type='aseg')
    
    """
    aseg = db.get_anat(subject, type="aseg").get_data().T

    if not isinstance(aseg_name, (list, tuple)):
        aseg_name = [aseg_name]

    mask = np.zeros(aseg.shape)
    for name in aseg_name:
        if name in fs_aseg_dict:
            tmp = aseg==fs_aseg_dict[name]
        else:
            # Combine all masks containing `name` (e.g. all masks with 'cerebellum' in the name)
            keys = [k for k in fs_aseg_dict.keys() if name.lower() in k.lower()]
            if len(keys) == 0:
                raise ValueError('Unknown aseg_name!')
            tmp = np.any(np.array([aseg==fs_aseg_dict[k] for k in keys]), axis=0)
        mask = np.logical_or(mask, tmp)
    if xfmname is not None:
        mask = anat2epispace(mask.astype(float), subject, xfmname, order=order, **kwargs)
    if threshold is not None:
        mask = mask > threshold
    return mask


def get_roi_masks(subject, xfmname, roi_list=None, gm_sampler='cortical', split_lr=False, 
                  allow_overlap=False, fail_for_missing_rois=False, return_dict=True):
    """Return a dictionary of roi masks

    This function returns a single 3D array with a separate numerical index for each ROI, 

    Parameters
    ----------
    subject : string
        pycortex subject ID
    xfmname : string
        pycortex transformation name
    roi_list : list or None
        List of names of ROIs to retrieve (e.g. ['FFA','OFA','EBA']). Names should match 
        the ROI layers in the rois.svg file for the `subject` specified. If None is 
        provided (default), all available ROIs for the subject are returned.
    gm_sampler : scalar or string
        How to sample the cortical gray matter. Options are: 
        integer - Distance from fiducial surface to define ROI. Reasonable values ~ [1-3]
        The following will only work if you have used Freesurfer to define the subject's 
        surface, and so have separate pial and white matter surfaces:
        'cortical' - selection of all voxels with centers within the cortical ribbon
            (directly computed from distance of each voxel from fiducial surface)
        'cortical-liberal' - selection of all voxels that have any part within the 
            cortical ribbon ('line_nearest' mapper)
        'cortical-conservative' - selection of only the closest voxel to each surface 
            vertex ('nearest' mapper)
    split_lr : bool
        Whether to separate ROIs in to left and right hemispheres (e.g., 'V1' becomes 
        'V1_L' and 'V1_R')
    allow_overlap : bool
        Whether to allow ROIs to include voxels in other ROIs (default:False)
    fail_for_missing_rois : bool
        Whether to fail if one or more of the rois specified in roi_list are not avail-
        able. If set to False (default behavior), requested rois that are not present 
        will be ignored and excluded from the output
    return_dict : bool
        If True (default), function returns a dictionary of ROI masks; if False, a volume 
        with integer indices for each ROI (similar to Freesurfer's aseg masks) and a 
        dictionary of how the indices map to ROI names are returned. 
    
    Returns
    -------
    roi_masks : dict
        Dictionary of arrays; keys are ROI names, values are roi masks.
    - OR - 
    (index_volume, index_labels) : tuple of array & dict
        `index_volume` is a 3D array with a separate numerical index value for each ROI. Index values
        in the left hemisphere are negative. (For example, if V1 in the right hemisphere is 1, then V1 in 
        the left hemisphere will be -1). `index_labels` is a dict that maps roi names to index values 
        (e.g. {'V1': 1}). 
    """

    # Start with vertices
    tmp_list = [r for r in roi_list if not r=='Cortex']
    roi_verts = get_roi_verts(subject, roi=tmp_list)
    if roi_list is None:
        roi_list = list(roi_verts.keys())
    if not isinstance(roi_list, (list, tuple)):
        roi_list = [roi_list]
    if fail_for_missing_rois:
        missing = [r for r in roi_list if not r in roi_verts.keys()+['Cortex']]
        if len(missing) > 0:
            raise ValueError('One or more ROIs {} not found in overlays.svg file!'.format(missing))

    mapper_dict = {'cortical-conservative':'nearest',
                   'cortical-liberal':'line_nearest'}

    
    if do_mapper:
        ### 
        # Need more mapper types
        mapper = get_mapper(subject, xfmname, type=mapper_dict[gm_sampler])
        for name, verts in list(roi_verts.items()):
            left, right = mapper.backwards(verts)
            if split_lr:
                output[name+'_L'] = left > 0
                output[name+'_R'] = right > 0
            else:
                output[name] = (left>0) | (right>0)
        
    #return output

    elif (gm_sampler == 'cortical') or not isinstance(gm_sampler, string_types):
        # Get map of vertex indices for each voxel
        vox_dst, vox_idx = get_vox_dist(subject, xfmname)
        if isinstance(dst, string_types):
            cortex_mask = db.get_mask(subject, xfmname, type=dst)
        else:
            cortex_mask = vox_dst <= dst
        # Select roi voxels by combination of vertex indices & distance from fiducial surface
        roi_voxels = {}
        for roi in roi_list:
            if roi not in roi_verts:
                if not roi=='Cortex':
                    print("ROI {} not found...".format(roi))
                continue
            tmp = np.in1d(vox_idx.flatten(), roi_verts[roi]).reshape(vox_idx.shape)
            roi_mask = tmp & cortex_mask
            if np.any(roi_mask):
                roi_voxels[roi] = roi_mask
            else:
                print("ROI {} is empty...".format(roi))
        all_mask = np.array(roi_voxels.values()).sum(0)
        if 'Cortex' in roi_list:
            roi_voxels['Cortex'] = (all_mask==0) & cortex_mask
        # Optionally cull voxels assigned to > 1 ROI due to partly overlapping shapes in inkscape file:
        if not allow_overlap:
            print('Cutting {} overlapping voxels (should be < ~50)'.format(np.sum(all_mask>1)))
            for roi in roi_list:
                roi_voxels[roi][all_mask>1] = False
        # Split left / right hemispheres if desired
        # There ought to be a more succinct way to do this - get_hemi_masks only does the cortical
        # ribbon, and is not guaranteed to have all voxels in the cortex_mask specified in this fn
        left_verts, right_verts = db.get_surf(subject, "flat", merge=False, nudge=True)
        left_mask = vox_idx < len(np.unique(left_verts[1]))
        right_mask = np.logical_not(left_mask)
        if split_lr:
            roi_voxels_lr = {}
            for roi in roi_list:
                roi_voxels_lr[roi+'_L'] = roi_voxels[roi] & left_mask
                roi_voxels_lr[roi+'_R'] = roi_voxels[roi] & right_mask
            output = roi_voxels_lr
        else:
            output = roi_voxels
    # Support alternative outputs for backward compatibility
    if return_dict:
        return output
    else:
        idx_vol = np.zeros(vox_idx.shape, dtype=np.int64)
        idx_labels = {}
        for iroi, roi in enumerate(roi_list):
            idx_vol[roi_voxels[roi]] = iroi
            idx_labels[roi] = iroi
        idx_vol[left_mask] *= -1
        return idx_vol, idx_labels

def get_dropout(subject, xfmname, power=20):
    """Create a dropout Volume showing where EPI signal
    is very low.
    
    Parameters
    ----------
    subject : str
        Name of subject
    xfmname : str
        Name of transform
    power :
    
    Returns
    -------
    volume : dataview
        Pycortex volume of low signal locations
    """
    xfm = db.get_xfm(subject, xfmname)
    rawdata = xfm.reference.get_data().T.astype(np.float32)

    # Collapse epi across time if it's 4D
    if rawdata.ndim > 3:
        rawdata = rawdata.mean(0)

    rawdata[rawdata==0] = np.mean(rawdata[rawdata!=0])
    normdata = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min())
    normdata = (1 - normdata) ** power

    from .dataset import Volume
    return Volume(normdata, subject, xfmname)

def make_movie(stim, outfile, fps=15, size="640x480"):
    """Makes an .ogv movie

    A simple wrapper for ffmpeg. Calls:
    "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    
    Parameters
    ----------
    stim : 
    
    outfile : str
    
    fps : float
        refresh rate of the stimulus
    size : str
        resolution of the movie out
        
    Returns
    -------
    
    """
    import shlex
    import subprocess as sp
    cmd = "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    fcmd = cmd.format(infile=stim, size=size, fps=fps, outfile=outfile)
    sp.call(shlex.split(fcmd))

def vertex_to_voxel(subject):
    """
    Parameters
    ----------
    subject : str
        Name of subject
        
    Returns
    -------
    """
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
    """Gets a colormaps
    
    Parameters
    ----------
    name : str
        Name of colormap to get
        
    Returns
    -------
    cmap : ListedColormap
        Matplotlib colormap object
    """
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

def add_cmap(cmap, name, cmapdir=None):
    """Add a colormap to pycortex
    
    This stores a matplotlib colormap in the pycortex filestore, such that it can 
    be used in the webgl viewer in pycortex. See [] for more information about how
    to generate colormaps in matplotlib

    Parameters
    ----------
    cmap : matplotlib colormap
        Color map to be saved
    name : 
        Name for colormap, e.g. 'jet', 'blue_to_yellow', etc. This will be a file name,
        so no weird characters. This name will also be used to specify this colormap in 
        future calls to cortex.quickflat.make_figure() or cortex.webgl.show()
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    x = np.linspace(0, 1, 256)
    cmap_im = cmap(x).reshape((1,256,4))
    if cmapdir is None:
        # Probably won't work due to permissions...
        cmapdir = config.get('webgl', 'colormaps')
    plt.imsave(os.path.join(cmapdir, name), cmap_im)

