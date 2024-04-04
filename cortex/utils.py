"""Contain utility functions
"""
import binascii
import copy
import io
import os
import shutil
import tarfile
import tempfile
import warnings
from distutils.version import LooseVersion
from importlib import import_module

import h5py
import numpy as np
import wget

from . import formats
from .database import db
from .freesurfer import fs_aseg_dict
from .options import config
from .polyutils import Surface
from .testing_utils import INKSCAPE_VERSION
from .volume import anat2epispace


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
    warnings.warn('Please use db.get_overlay instead', DeprecationWarning)
    return db.get_overlay(*args, **kwargs)

get_mapper = DocLoader("get_mapper", ".mapper", "cortex")

def get_ctmpack(subject, types=("inflated",), method="raw", level=0, recache=False,
                decimate=False, external_svg=None,
                overlays_available=None):
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
        string specifying method of how inverse transforms for
        labels are computed (determines how labels are displayed
        on 3D viewer) one of ['mg2','raw']
    recache : bool
        Whether to re-generate .ctm files. Can resolve some errors
        but takes more time to re-generate cached files.
    decimate : bool
        whether to decimate the mesh geometry of the hemispheres
        to reduce file size
    external_svg : str or None
        file string for .svg file containing alternative overlays
        for brain viewer. If None, the `overlays.svg` file for this
        subject (in the pycortex_store folder for the subejct) is used.
    overlays_available: tuple or None
        Which overlays in the svg file to include in the viewer. If
        None, all layers in the relevant svg file are included.

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
                               decimate=decimate,
                               external_svg=external_svg,
                               overlays_available=overlays_available)
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
        Mask type, one of {"cortical", "thin", "thick", "nearest", "line_nearest"}.
          - 'cortical' includes voxels contained within the cortical ribbon, 
          between the freesurfer-estimated white matter and pial surfaces. 
          - 'thin' includes voxels that are < 2mm away from the fiducial surface. 
          - 'thick' includes voxels that are < 8mm away from the fiducial surface.
          - 'nearest' includes only the voxels overlapping the fiducial surface.
          - 'line_nearest' includes all voxels that have any part within the cortical 
            ribbon.

    Returns
    -------
    mask : array
        boolean mask array for cortical voxels in functional space

    Notes
    -----
    "nearest" is a conservative "cortical" mask, while "line_nearest" is a liberal 
    "cortical" mask.
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
    max_dist : nonnegative float, optional
        Limit computation to only voxels within `max_dist` mm of the surface.
        Makes computation orders of magnitude faster for high-resolution volumes.

    Returns
    -------
    dist : ndarray (z, y, x)
        Array with the same shape as the reference image of `xfmname` containing
        the distance (in mm) of each voxel to the closest point on the surface.

    argdist : ndarray (z, y, x)
        Array with the same shape as the reference image of `xfmname` containing
        for each voxel the index of the closest point on the surface.
    """
    from scipy.spatial import cKDTree

    fiducial, polys = db.get_surf(subject, surface, merge=True)
    xfm = db.get_xfm(subject, xfmname)
    z, y, x = xfm.shape
    idx = np.mgrid[:x, :y, :z].reshape(3, -1).T
    mm = xfm.inv(idx)

    tree = cKDTree(fiducial)
    dist, argdist = tree.query(mm, distance_upper_bound=max_dist)
    dist.shape = (x, y, z)
    argdist.shape = (x, y, z)
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

def add_roi(data, name="new_roi", open_inkscape=True, add_path=True,
            overlay_file=None, **kwargs):
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
    overlay_file : str, optional
        Custom overlays.svg file to use instead of the default one for this
        subject (if not None). Default None.
    kwargs : dict
        Passed to cortex.quickflat.make_png
    """
    import subprocess as sp

    from . import dataset, quickflat

    dv = dataset.normalize(data)
    if isinstance(dv, dataset.Dataset):
        raise TypeError("Please specify a data view")

    svg = db.get_overlay(dv.subject, overlay_file=overlay_file)
    fp = io.BytesIO()

    quickflat.make_png(fp, dv, height=1024, with_rois=False, with_labels=False, **kwargs)
    fp.seek(0)
    svg.rois.add_shape(name, binascii.b2a_base64(fp.read()).decode('utf-8'), add_path)

    if open_inkscape:
        inkscape_cmd = config.get('dependency_paths', 'inkscape')
        if LooseVersion(INKSCAPE_VERSION) < LooseVersion('1.0'):
            cmd = [inkscape_cmd, '-f', svg.svgfile]
        else:
            cmd = [inkscape_cmd, svg.svgfile]
        return sp.call(cmd)


def _get_neighbors_dict(polys):
    """Return a dictionary of {vertex : set(neighbor vertices)} for the given polys"""
    neighbors_dict = {}
    for poly in polys:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            neighbors_dict.setdefault(poly[i], set()).add(poly[j])
            neighbors_dict.setdefault(poly[j], set()).add(poly[i])
    return neighbors_dict


def get_roi_verts(subject, roi=None, mask=False, overlay_file=None):
    """Return vertices for the given ROIs, or all ROIs if none are given.

    Parameters
    ----------
    subject : str
        Name of the subject
    roi : str, list or None, optional
        ROIs to fetch. Can be ROI name (string), a list of ROI names, or
        None, in which case all ROIs will be fetched.
    mask : bool
        if True, return a logical mask across vertices for the roi
        if False, return a list of indices for the ROI
    overlay_file : None or str
        Pass another overlays file instead of the default overlays.svg

    Returns
    -------
    roidict : dict
        Dictionary of {roi name : roi verts}. ROI verts are for both
        hemispheres, with right hemisphere vertex numbers sequential
        after left hemisphere vertex numbers.
    """
    # Get overlays
    svg = db.get_overlay(subject, overlay_file=overlay_file)

    # Get flat surface so we can figure out which verts are in medial wall
    # or in cuts
    # This assumes subject has flat surface, which they must to have ROIs.
    pts, polys = db.get_surf(subject, "flat", merge=True)
    goodpts = np.unique(polys)

    # Load also the pts and polys of the full surface without cuts, to recover
    # vertices that were removed from the flat surface
    _, polys_full = db.get_surf(subject, "fiducial", merge=True)
    neighbors_dict = _get_neighbors_dict(polys_full)

    if roi is None:
        roi = svg.rois.shapes.keys()

    roidict = dict()
    if isinstance(roi, str):
        roi = [roi]

    for name in roi:
        roi_idx = np.intersect1d(svg.rois.get_mask(name), goodpts)
        # Now we want to include also the vertices that were removed from the flat 
        # surface that is, for every vertex in roi_idx we want to add the pts that are 
        # not in goodpts but that are in pts_full
        # to do that, we need to find the neighboring indices from polys_full
        extra_idx = set()
        for idx in roi_idx:
            extra_idx.update(ii for ii in neighbors_dict[idx] if ii not in goodpts)
        if extra_idx:
            roi_idx = np.unique(np.concatenate((roi_idx, list(extra_idx)))).astype(int)

        if mask:
            roidict[name] = np.zeros(pts.shape[:1], dtype=bool)
            if np.any(roi_idx):
                roidict[name][roi_idx] = True
            else:
                warnings.warn("No vertices found in {}!".format(name))
        else:
            roidict[name] = roi_idx

    return roidict


def get_roi_surf(subject, surf_type, roi, overlay_file=None):
    """Similar to get_roi_verts, but gets both the points and the polys for an roi.

    Parameters
    ----------
    subject : str
        Name of subject
    surf_type : str
        Type of surface to return, probably in (fiducial, inflated, veryinflated, hyperinflated,
        superinflated, flat)
    roi : str
        Name of ROI to get the surface geometry for.
    overlay_file : None or str
        Pass another overlays file instead of the default overlays.svg

    Returns
    -------
    pts, polys : (array, array)
        The points, specified in 3D space, as well as indices into pts specifying the polys.
    """
    roi_verts_mask = get_roi_verts(subject, roi, mask=True, overlay_file=overlay_file)
    pts, polys = db.get_surf(subject, surf_type, merge=True, nudge=True)
    vert_idx = np.where(roi_verts_mask[roi])[0]
    vert_set = set(vert_idx)
    roi_polys = []
    for i in range(np.shape(polys)[0]):
        if np.array(list(map(lambda x: x in vert_set, polys[i, :]))).all():
            roi_polys.append(polys[i, :])
    reindexed_polys = []
    vert_rev_hash_idx = {}
    for i, v in enumerate(vert_idx):
        vert_rev_hash_idx[v] = i
    for poly in roi_polys:
        reindexed_polys.append(list(map(vert_rev_hash_idx.get, poly)))
    return pts[vert_idx], np.array(reindexed_polys)


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
    warnings.warn('Deprecated! Use get_roi_masks')

    mapper = get_mapper(subject, xfmname, type=projection)
    rois = get_roi_verts(subject, roi=roi, mask=True)
    output = dict()
    for name, verts in list(rois.items()):
        # This is broken; unclear when/if backward mappers ever worked this way.
        #left, right = mapper.backwards(vert_mask)
        #output[name] = left + right
        output[name] = mapper.backwards(verts.astype(float))
        # Threshold?
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
    from .freesurfer import fs_aseg_dict
    aseg = db.get_anat(subject, type="aseg").get_fdata().T

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
                  allow_overlap=False, fail_for_missing_rois=True, exclude_empty_rois=False,
                  threshold=None, return_dict=True, overlay_file=None):
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
        the ROI layers in the overlays.svg file for the `subject` specified. If None is
        provided (default), all available ROIs for the subject are returned. If 'Cortex'
        is included in roi_list*, a mask of all cortical voxels NOT included in other
        requested rois is included in the output.
        * works for gm_sampler = 'cortical', 'think', 'thick', or (any scalar value);
        does not work for mapper-based gray matter samplers.
    gm_sampler : scalar or string
        How to sample the cortical gray matter. Options are:
        <an integer> - Distance from fiducial surface to define ROI. Reasonable values
            for this input range from 1-3.
        The following will only work if you have used Freesurfer to define the subject's
        surface, and so have separate pial and white matter surfaces:
        'cortical' - selection of all voxels with centers within the cortical ribbon
            (directly computed from distance of each voxel from fiducial surface)
        'thick' - selection of voxels within 'thick' mask (see cortex.get_mask())
        'thin' - selection of voxels within 'thin' mask (see cortex.get_mask())
        'cortical-liberal' - selection of all voxels that have any part within the
            cortical ribbon ('line_nearest' mapper)
        'cortical-conservative' - selection of only the closest voxel to each surface
            vertex ('nearest' mapper)
        mapper-based gm_samplers will return floating point values from 0-1 for each
        voxel (reflecting the fraction of that voxel inside the ROI) unless a threshold
        is provided.
    threshold : float [0-1]
        value used to convert probabilistic ROI values to a boolean mask for the ROI.
    split_lr : bool
        Whether to separate ROIs in to left and right hemispheres (e.g., 'V1' becomes
        'V1_L' and 'V1_R')
    allow_overlap : bool
        Whether to allow ROIs to include voxels in other ROIs (default:False). This
        should only be relevant if (a) spline shapes defining ROIs in overlays.svg
        overlap at all, or (b) a low threshold is set for a mapper-based gm_sampler
    fail_for_missing_rois : bool
        Whether to fail if one or more of the rois specified in roi_list are not
        defined in the overlays.svg file
    exclude_empty_rois : bool
        Whether to fail if an ROI that is present in the overlays.svg file contains no
        voxels due to the scan not targeting that region of the brain.
    return_dict : bool
        If True (default), function returns a dictionary of ROI masks; if False, a volume
        with integer indices for each ROI (similar to Freesurfer's aseg masks) and a
        dictionary of how the indices map to ROI names are returned.
    overlay_file : str or None
        If None, use the default `overlays.svg` file. Otherwise, use the passed
        overlay file to look for the ROIs.

    Returns
    -------
    roi_masks : dict
        Dictionary of arrays; keys are ROI names, values are roi masks.
    - OR -
    index_volume, index_labels : array, dict
        `index_volume` is a 3D array with a separate numerical index value for each ROI. Index values
        in the left hemisphere are negative. (For example, if V1 in the right hemisphere is 1, then V1 in
        the left hemisphere will be -1). `index_labels` is a dict that maps roi names to index values
        (e.g. {'V1': 1}).

    Notes
    -----
    Some gm_samplers may fail if you have very high-resolution data (i.e., with voxels on the
    order of the spacing between vertices in your cortical mesh). In such cases, there may be
    voxels in the middle of your ROI that are not assigned to the ROI (because no vertex falls
    within that voxel). For such cases, it is recommended to use 'cortical', 'thick', or
    'thin' as your `gm_sampler`.
    """
    # Convert mapper names to pycortex sampler types
    mapper_dict = {'cortical-conservative':'nearest',
                   'cortical-liberal':'line_nearest'}
    # Method
    use_mapper = gm_sampler in mapper_dict
    use_cortex_mask = (gm_sampler in ('cortical', 'thick', 'thin')) or not isinstance(gm_sampler, str)
    if not (use_mapper or use_cortex_mask):
        raise ValueError('Unknown gray matter sampler (gm_sampler)!')
    # Initialize
    roi_voxels = {}
    pct_coverage = {}
    # Catch single-ROI input
    if isinstance(roi_list, str):
        roi_list = [roi_list]
    if not return_dict:
        split_lr = True
        if use_mapper and threshold is None:
            raise ValueError(
                f"You must set a threshold for gm_sampler={gm_sampler} if you want an "
                "indexed volume output"
            )
    # Start with vertices
    if roi_list is None:
        roi_verts = get_roi_verts(subject, mask=use_mapper, overlay_file=overlay_file)
        roi_list = list(roi_verts.keys())
    else:
        tmp_list = [r for r in roi_list if not r=='Cortex']
        try:
            roi_verts = get_roi_verts(subject, roi=tmp_list, mask=use_mapper, overlay_file=overlay_file)
        except KeyError as key:
            if fail_for_missing_rois:
                raise KeyError("Requested ROI {} not found in overlays.svg!".format(key))
            else:
                roi_verts = get_roi_verts(subject, roi=None, mask=use_mapper, overlay_file=overlay_file)
                missing = [r for r in roi_list if not r in roi_verts.keys()+['Cortex']]
                roi_verts = dict((roi, verts) for roi, verts in roi_verts.items() if roi in roi_list)
                roi_list = list(set(roi_list)-set(missing))
                print('Requested ROI(s) {} not found in overlays.svg!'.format(missing))
    # Get (a) indices for nearest vertex to each voxel
    # and (b) distance from each voxel to nearest vertex in fiducial surface
    if (use_cortex_mask or split_lr) or (not return_dict):
        vox_dst, vox_idx = get_vox_dist(subject, xfmname)
    if use_mapper:
        mapper = get_mapper(subject, xfmname, type=mapper_dict[gm_sampler])
    elif use_cortex_mask:
        if isinstance(gm_sampler, str):
            cortex_mask = db.get_mask(subject, xfmname, type=gm_sampler)
        else:
            cortex_mask = vox_dst <= gm_sampler
    # Loop over ROIs to map vertices to volume, using mapper or cortex mask + vertex indices
    for roi in roi_list:
        if roi not in roi_verts:
            if not roi=='Cortex':
                print("ROI {} not found...".format(roi))
            continue
        if use_mapper:
            roi_voxels[roi] = mapper.backwards(roi_verts[roi].astype(float))
            # Optionally threshold probabilistic values returned by mapper
            if threshold is not None:
                roi_voxels[roi] = roi_voxels[roi] > threshold
            # Check for partial / empty rois:
            vert_in_scan = np.hstack([np.array((m>0).sum(1)).flatten() for m in mapper.masks])
            vert_in_scan = vert_in_scan[roi_verts[roi]]
        elif use_cortex_mask:
            vox_in_roi = np.in1d(vox_idx.flatten(), roi_verts[roi]).reshape(vox_idx.shape)
            roi_voxels[roi] = vox_in_roi & cortex_mask
            # This is not accurate... because vox_idx only contains the indices of the *nearest*
            # vertex to each voxel, it excludes many vertices. I can't think of a way to compute
            # this accurately for non-mapper gm_samplers for now... ML 2017.07.14
            vert_in_scan = np.in1d(roi_verts[roi], vox_idx[cortex_mask])
        # Compute ROI coverage
        pct_coverage[roi] = vert_in_scan.mean() * 100
        if use_mapper:
            print("Found %0.2f%% of %s"%(pct_coverage[roi], roi))

    # Create cortex mask
    all_mask = np.array(list(roi_voxels.values())).sum(0)
    if 'Cortex' in roi_list:
        if use_mapper:
            # cortex_mask isn't defined / exactly definable if you're using a mapper
            print("Cortex roi not included b/c currently not compatible with your selection for gm_sampler")
            _ = roi_list.pop(roi_list.index('Cortex'))
        else:
            roi_voxels['Cortex'] = (all_mask==0) & cortex_mask
    # Optionally cull voxels assigned to > 1 ROI due to partly overlapping ROI splines
    # in inkscape overlays.svg file:
    if not allow_overlap:
        print('Cutting {} overlapping voxels (should be < ~50)'.format(np.sum(all_mask > 1)))
        for roi in roi_list:
            roi_voxels[roi][all_mask > 1] = False
    # Split left / right hemispheres if desired
    if split_lr:
        # Use the fiducial surface because we need to have all vertices
        left_verts, _ = db.get_surf(subject, "fiducial", merge=False, nudge=True)  
        left_mask = vox_idx < len(np.unique(left_verts[1]))
        right_mask = np.logical_not(left_mask)
        roi_voxels_lr = {}
        for roi in roi_list:
            # roi_voxels may contain float values if using a mapper, therefore we need
            # to manually set the voxels in the other hemisphere to False. Then we let
            # numpy do the conversion False -> 0. 
            roi_voxels_lr[roi + '_L'] = copy.copy(roi_voxels[roi])
            roi_voxels_lr[roi + '_L'][right_mask] = False
            roi_voxels_lr[roi + '_R'] = copy.copy(roi_voxels[roi])
            roi_voxels_lr[roi + '_R'][left_mask] = False
        output = roi_voxels_lr
    else:
        output = roi_voxels

    # Check percent coverage / optionally cull empty ROIs
    for roi in set(roi_list)-set(['Cortex']):
        if pct_coverage[roi] < 100:
            # if not np.any(mask) : reject ROI
            if pct_coverage[roi]==0:
                warnings.warn('ROI %s is entirely missing from your scan protocol!'%(roi))
                if exclude_empty_rois:
                    if split_lr:
                        _ = output.pop(roi+'_L')
                        _ = output.pop(roi+'_R')
                    else:
                        _ = output.pop(roi)
            else:
                # I think this is the only one for which this works correctly...
                if gm_sampler=='cortical-conservative':
                    warnings.warn('ROI %s is only %0.2f%% contained in your scan protocol!'%(roi, pct_coverage[roi]))

    # Support alternative outputs for backward compatibility
    if return_dict:
        return output
    else:
        idx_vol = np.zeros(vox_idx.shape, dtype=np.int64)
        idx_labels = {}
        for iroi, roi in enumerate(roi_list, 1):
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
    rawdata = xfm.reference.get_fdata().T.astype(np.float32)

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

def vertex_to_voxel(subject):  # Am I deprecated in favor of mappers??? Maybe?
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


def _set_edge_distance_graph_attribute(graph, pts, polys):
    '''
    adds the attribute 'edge distance' to a graph
    '''
    import networkx as nx

    l2_distance = lambda v1, v2: np.linalg.norm(pts[v1] - pts[v2])
    heuristic = l2_distance # A* heuristic

    if not nx.get_edge_attributes(graph, 'distance'): # Add edge distances as an attribute to this graph if it isn't there
        edge_distances = dict()
        for x,y,z in polys:
            edge_distances[(x,y)] = heuristic(x,y)
            edge_distances[(y,x)] = heuristic(y,x)
            edge_distances[(y,z)] = heuristic(y,z)
            edge_distances[(z,y)] = heuristic(z,y)
            edge_distances[(x,z)] = heuristic(x,z)
            edge_distances[(z,x)] = heuristic(z,x)
        nx.set_edge_attributes(graph, edge_distances, name='distance')


def get_shared_voxels(subject, xfmname, hemi="both", merge=True, use_astar=True):
    '''Return voxels that are shared by multiple vertices, and for each such voxel,
       also returns the mutually farthest pair of vertices mapping to the voxel
    Parameters
    ----------
    subject : str
        Name of the subject
    xfmname : str
        Name of the transform
    hemi : str, optional
        Which hemisphere to return. For now, only 'lh' or 'rh'
    merge : bool, optional
        Join the hemispheres, if requesting both
    use_astar: bool, optional
        Toggle to decide whether to use A* search or geodesic paths for the
        shortest paths

    Returns
    -------
    vox_vert_array: np.array,
    array of dimensions # voxels X 3, columns being: (vox_idx, farthest_pair[0],
    farthest_pair[1])
    '''

    import networkx as nx
    from scipy.sparse import find as sparse_find
    Lmask, Rmask = get_mapper(subject, xfmname).masks  # Get masks for left and right hemisphere
    if hemi == 'both':
        hemispheres = ['lh', 'rh']
    else:
        hemispheres = [hemi]
    out = []
    for hem in hemispheres:
        if hem == 'lh':
            mask = Lmask
        else:
            mask = Rmask

        all_voxels = mask.tolil().transpose().rows  # Map from voxels to verts
        vert_to_vox_map = dict(zip(*(sparse_find(mask)[:2])))  # From verts to vox

        pts_fid, polys_fid = db.get_surf(subject, 'fiducial', hem)  # Get the fiducial surface
        surf = Surface(pts_fid, polys_fid) #Get the fiducial surface
        graph = surf.graph

        _set_edge_distance_graph_attribute(graph, pts_fid, polys_fid)

        l2_distance = lambda v1, v2: np.linalg.norm(pts_fid[v1] - pts_fid[v2])
        heuristic = l2_distance  # A* heuristic

        if use_astar:
            shortest_path = lambda a, b: nx.astar_path(graph, a, b, heuristic=heuristic, weight='distance') # Find approximate shortest paths using A* search
        else:
            shortest_path = surf.geodesic_path  # Find shortest paths using geodesic distances

        vox_vert_list = []
        for vox_idx, vox in enumerate(all_voxels):
            if len(vox) > 1:  # If the voxel maps to multiple vertices
                vox = np.array(vox).astype(int)
                for v1 in range(vox.size-1):
                    vert1 = vox[v1]
                    if vert1 in vert_to_vox_map:  # If the vertex is a valid vertex
                        for v2 in range(v1+1, vox.size):
                            vert2 = vox[v2]
                            if vert2 in vert_to_vox_map:  # If the vertex is a valid vertex
                                path = shortest_path(vert1, vert2)
                                # Test whether any vertex in path goes out of the voxel
                                stays_in_voxel = all([(v in vert_to_vox_map) and (vert_to_vox_map[v] == vox_idx) for v in path])
                                if not stays_in_voxel:
                                    vox_vert_list.append([vox_idx, vert1, vert2])

        tmp =  np.array(vox_vert_list)
        # Add offset for right hem voxels
        if hem=='rh':
            tmp[:, 1:3] += Lmask.shape[0]
        out.append(tmp)
    if hemi in ('lh', 'rh'):
        return out[0]
    else:
        if merge:
            return np.vstack(out)
        else:
            return tuple(out)


def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.
    """
    import scipy.sparse
    with h5py.File(fname) as hf:
        data = (hf['%s_data'%varname], hf['%s_indices'%varname], hf['%s_indptr'%varname])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape'%varname])
    return sparsemat


def save_sparse_array(fname, data, varname, mode='a'):
    """Save a numpy sparse array to an hdf file

    Results in relatively smaller file size than numpy.savez

    Parameters
    ----------
    fname : string
        file name to save
    data : sparse array
        data to save
    varname : string
        name of variable to save
    mode : string
        write / append mode set, one of ['w','a'] (passed to h5py.File())
    """
    import scipy.sparse
    if not isinstance(data, scipy.sparse.csr.csr_matrix):
        data_ = scipy.sparse.csr_matrix(data)
    else:
        data_ = data
    with h5py.File(fname, mode=mode) as hf:
        # Save indices
        hf.create_dataset(varname + '_indices', data=data_.indices, compression='gzip')
        # Save data
        hf.create_dataset(varname + '_data', data=data_.data, compression='gzip')
        # Save indptr
        hf.create_dataset(varname + '_indptr', data=data_.indptr, compression='gzip')
        # Save shape
        hf.create_dataset(varname + '_shape', data=data_.shape, compression='gzip')


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
        try:
            plt.cm.register_cmap(name,cmap)
        except:
            print(f"Color map {name} is already registered.")
    else:
        try:
            cmap = plt.cm.get_cmap(name)
        except:
            raise Exception('Unkown color map!')
    return cmap

def add_cmap(cmap, name, cmapdir=None):
    """Add a colormap to pycortex.

    This stores a matplotlib colormap in the pycortex filestore, such that it can
    be used in the webgl viewer in pycortex. See 
    https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html 
    for more information about how to generate colormaps in matplotlib.

    Parameters
    ----------
    cmap : matplotlib colormap
        Color map to be saved
    name : str
        Name for colormap, e.g. 'jet', 'blue_to_yellow', etc. The name will be used
        to generate a filename for the colormap stored in the pycortex store, 
        so avoid illegal characters for a filename. This name will also be used to 
        specify this colormap in future calls to `cortex.quickflat.make_figure()`
        or `cortex.webgl.show()`.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 256)
    cmap_im = cmap(x).reshape((1, 256, 4))
    if cmapdir is None:
        # Probably won't work due to permissions...
        cmapdir = config.get("webgl", "colormaps")
    # Make sure name ends with png
    name = name if name.endswith(".png") else f"{name}.png"
    plt.imsave(os.path.join(cmapdir, name), cmap_im, format="png")


def download_subject(subject_id='fsaverage', url=None, pycortex_store=None,
                     download_again=False):
    """Download subjects to pycortex store

    Parameters
    ----------
    subject_id : string
        subject identifying string in pycortex. This assumes that
        the file downloaded from some URL is of the form <subject_id>.tar.gz
    url: string or None
        URL from which to download. Not necessary to specify for subjects
        known to pycortex (None is OK). Known subjects will have a default URL.
        Currently,the only known subjects is 'fsaverage', but there are plans
        to add more in the future. If provided, URL overrides `subject_id`
    pycortex_store : string or None
        Directory to which to put the new subject folder. If None, defaults to
        current filestore in use (`cortex.db.filestore`).
    download_again : bool
        Download the data again even if the subject id is already present in
        the pycortex's database.
    """
    if subject_id in db.subjects and not download_again:
        warnings.warn(
            "{} is already present in the database. "
            "Set download_again to True if you wish to download "
            "the subject again.".format(subject_id))
        return
    # Map codes to URLs; more coming eventually
    id_to_url = dict(fsaverage='https://ndownloader.figshare.com/files/17827577?private_link=4871247dce31e188e758',
                     )
    if url is None:
        if not subject_id in id_to_url:
            raise ValueError('Unknown subject_id!')
        url = id_to_url[subject_id]
    print("Downloading from: {}".format(url))
    # Download to temp dir
    tmp_dir = tempfile.gettempdir()
    wget.download(url, tmp_dir)
    print('Downloaded subject {} to {}'.format(subject_id, tmp_dir))
    # Un-tar to pycortex store
    if pycortex_store is None:
        # Default location is current filestore in cortex.db
        pycortex_store = db.filestore
    pycortex_store = os.path.expanduser(pycortex_store)
    with tarfile.open(os.path.join(tmp_dir, subject_id + '.tar.gz'), "r:gz") as tar:
        print("Extracting subject {} to {}".format(subject_id, pycortex_store))
        tar.extractall(path=pycortex_store)

    # reload all subjects from the filestore
    db.reload_subjects()


def rotate_flatmap(surf_id, theta, plot=False):
    """Rotate flatmap to be less V-shaped
    
    Parameters
    ----------
    surf_id : str
        pycortex surface identifier
    theta : scalar
        angle in degrees to rotate flatmaps (rotation is clockwise 
        for right hemisphere and counter-clockwise for left)
    plot : bool
        Whether to make a coarse plot to visualize the changes
    """
    # Lazy load of matplotlib
    import matplotlib.pyplot as plt
    paths = db.get_paths(surf_id)['surfs']['flat']
    theta = np.radians(theta)
    if plot:
        fig, axs = plt.subplots(2, 2)
    for j, hem in enumerate(('lh','rh')):
        this_file = paths[hem]
        pts, polys = formats.read_gii(this_file)
        # Rotate clockwise (- rotation) for RH, counter-clockwise (+ rotation) for LH
        if hem == 'rh':
            rtheta = - theta
        else:
            rtheta = copy.copy(theta)
        rotation_mat = np.array([[np.cos(rtheta), -np.sin(rtheta)], [np.sin(rtheta), np.cos(rtheta)]])
        rotated = rotation_mat.dot(pts[:, :2].T).T
        pts_new = pts.copy()
        pts_new[:, :2] = rotated
        new_file, bkup_num = copy.copy(this_file), 0
        while os.path.exists(new_file):
            new_file = this_file.replace('.gii', '_rotbkup%02d.gii'%bkup_num)
            bkup_num += 1
        print('Backing up file at %s...' % new_file)
        shutil.copy(this_file, new_file)
        formats.write_gii(this_file, pts_new, polys)
        print('Overwriting %s...' % this_file)
        if plot:
            axs[0,j].plot(*pts[::100, :2].T, marker='r.')
            axs[0,j].axis('equal')
            axs[1,j].plot(*pts_new[::100, :2].T, marker='b.')
            axs[1,j].axis('equal')
    # Remove and back up overlays file
    overlay_file = db.get_paths(surf_id)['overlays']
    shutil.copy(overlay_file, overlay_file.replace('.svg', '_rotbkup%02d.svg'%bkup_num))
    os.unlink(overlay_file)
    # Regenerate file
    svg = db.get_overlay(surf_id)
