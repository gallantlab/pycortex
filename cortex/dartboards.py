from matplotlib import pyplot as plt
from matplotlib import transforms as mtrans
from matplotlib.colors import Normalize
import matplotlib
import numpy as np
import time
import os

from . import db
from . import options
from . import polyutils
from . import quickflat

from .dataset.views import Vertex, Vertex2D
from scipy import interpolate
from scipy.spatial import ConvexHull, distance
from .utils import get_cmap

try:
    CACHE_DIR = options.config.get('basic', 'cache')
except:
    CACHE_DIR = options.config.get('basic','filestore')

## Utils
from sklearn.metrics import pairwise_distances as _pairwise_distances
# def _pairwise_distances(array, metric='euclidean'):
#     """Simple wrapper for scipy distance function"""
#     d = distance.pdist(array, metric=metric)
#     return distance.squareform(d)

def flatten_list_of_lists(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list
    

def sph2cart(r, az, elev):
    """Convert spherical to cartesian coordinates

    Parameters
    ----------
    r : scalar or array-like
        radius
    az : scalar or array-like
        azimuth angle in degrees
    elev : scalar or array-like
        elevation angle in degrees
    """
    z = r * np.sin(np.radians(elev))
    rcoselev = r * np.cos(np.radians(elev))
    x = rcoselev * np.cos(np.radians(az))
    y = rcoselev * np.sin(np.radians(az))
    return x, y, z


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return phi, rho


def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def nonzero_coords(coords):
    return coords[np.any(coords != 0, axis=1)]


def outline_to_polar(outline, centroid, theta_direction='ccw'):
    phi, rho = cart2pol(*(outline.T-centroid[:, np.newaxis]))
    if theta_direction not in (-1, 0, 'counter-clockwise', 'ccw'):
        phi = np.pi - phi
    return np.stack([phi, rho], axis=0)


def interpolate_angles(angles, angle_sub_bins, period=2*np.pi):
    """interpolate between anchor angles

    CONSIDER UPDATING. 

    Parameters
    ----------
    angles : list
        list of angles over which to interpolate. Should be ordered, in radians,...
    angle_sub_bins : _type_
        _description_
    period : _type_, optional
        _description_, by default 2*np.pi
    """    
    def single_interpolate(angles):
        angles = list(angles)
        if angles[-1] % period != angles[0] % period:
            angles.append(angles[0])
        for i in range(1, len(angles)):
            if angles[i] < angles[i-1]:
                angles[i] += period
        interpolated_angles = []
        for i in range(len(angles)-1):
            interpolated_angles += [a for a in np.linspace(
                angles[i], angles[i+1], angle_sub_bins, endpoint=False)]
        interpolated_angles.append(angles[-1])
        return np.array(interpolated_angles) % period
    if np.array(angles).ndim == 2:
        return np.array([single_interpolate(hemi_angles) for hemi_angles in angles])
    else:
        return single_interpolate(angles)


def warp_phis(phis, current_anchors, new_anchors=None, period=2*np.pi):
    if isinstance(new_anchors, type(None)):
        new_anchors = np.linspace(0,period,len(current_anchors),endpoint=False)
    current_anchors = np.append(current_anchors, current_anchors[0]+period)
    new_anchors = np.append(new_anchors, new_anchors[0]+period)
    return np.interp(phis, current_anchors, new_anchors, period=round(period,14))


def angle_between(start_angle, end_angle, check_angle, period=2*np.pi):
    start_angle, end_angle, check_angle = start_angle % period, end_angle % period, check_angle % period
    if start_angle > end_angle:
        adjustment = period - start_angle
        start_angle += adjustment
        end_angle += adjustment
        check_angle += adjustment
    start_angle, end_angle, check_angle = start_angle % period, end_angle % period, check_angle % period
    return (start_angle <= check_angle) * (check_angle < end_angle)


def vertical_flip_path(path, overlay):
    if isinstance(path, matplotlib.path.Path):
        path.vertices[:, 1] = overlay.svgshape[1] - path.vertices[:, 1]
    elif isinstance(path, np.ndarray):
        path[:, 1] = overlay.svgshape[1] - path[:, 1]
    else:
        raise ValueError(
            "path should be either a matplotlib.path.Path or an np.ndarray")
    return path


def determine_path_hemi(overlay, path, percent_cutoff=.4):
    if isinstance(path, matplotlib.path.Path):
        path = path.vertices
    middle_x = overlay.svgshape[0]/2
    return np.mean(middle_x < path[:, 0]) >= percent_cutoff


def center_of_mass(X):
    """Calculate the center of mass of a closed polygon

    Parameters
    ----------
    X : array
        array of x, y coordinates that is (n_points x 2)

    Returns
    -------
    center_of_mass
        array of shape (2,) that contains (x, y) center of mass
    """    
    # calculate center of mass of a closed polygon
    x = X[:, 0]
    y = X[:, 1]
    g = (x[:-1]*y[1:] - x[1:]*y[:-1])
    A = 0.5*g.sum()
    cx = ((x[:-1] + x[1:])*g).sum()
    cy = ((y[:-1] + y[1:])*g).sum()
    return 1./(6*A)*np.array([cx, cy])


def split_axis(ax, fig=None, split='horizontal', gap=0, **kwargs):
    if fig is None:
        fig = ax.figure
    pos = ax.get_position()
    ax.axis('off')
    width = pos.x1 - pos.x0
    height = pos.y1 - pos.y0
    if split == 'horizontal':
        subax0 = fig.add_axes([
            pos.x0, 
            pos.y0, 
            width/2 - width*gap/2, 
            height], 
            **kwargs)
        subax1 = fig.add_axes([
            pos.x0 + width/2 + width*gap/2, 
            pos.y0, 
            width/2 - width*gap/2, 
            height], 
            **kwargs)
    elif split == 'vertical':
        subax0 = fig.add_axes([
            pos.x0, 
            pos.y0, 
            width, 
            height/2 - height*gap/2], 
            **kwargs)
        subax1 = fig.add_axes([
            pos.x0, 
            pos.y0 + height/2 + height*gap/2, 
            width, 
            height/2 - height*gap/2], 
            **kwargs)
    return subax0, subax1


def overlay_axis(ax, fig=None, polar=False):
    if fig is None:
        fig = ax.figure
    rect = ax.get_position()
    overlay_ax = fig.add_axes(rect, polar=polar, frameon=False)
    return overlay_ax


def get_num_hemi_vertices(overlay, surface_type='flat'):
    """Returns the number of vertices in the left and right hemispheres.
    
    Parameters
    ----------
    overlay (cortex.svgoverlay.SVGOverlay): Pycortex overlay for subject
    
    Returns
    -------
    list
        List of two values: [n_vertices_left, n_vertices_right]
    """
    subject_id = overlay.svgfile.split('/')[-2]
    surfs_flat = [polyutils.Surface(*d)
                  for d in db.get_surf(subject_id, surface_type)]
    return [len(s.pts) for s in surfs_flat]


def colormap_2d(
    cmap,
    data=None,
    vmin0=None,
    vmax0=None,
    vmin1=None,
    vmax1=None,
    map_to_uint8=False,
):
    """Map values in two dimensions to color according to a 2D color map image

    Parameters
    ----------
    data0 : array (1d)
        First dimension of data to map
    data1 : array (1d)
        Second dimension of data to map
    cmap : array (3d)
        image of values to use for 2D color map
    vmin0 : scalar
        vmin for first dimension
    vmin1 : scalar
        vmin for second dimension
    vmax0 : scalar
        vmax for first dimension
    vmax1 : scalar
        vmax for second dimension
    map_to_uint8 : bool
        whether to map color values to uint8 values (0-255)
    """
    if isinstance(cmap, str):
        # load pycortex 2D colormap
        cmapdir = options.config.get('webgl', 'colormaps')
        colormaps = os.listdir(cmapdir)
        colormaps = sorted([c for c in colormaps if '.png' in c])
        colormaps = dict((c[:-4], os.path.join(cmapdir, c)) for c in colormaps)
        cmap = plt.imread(colormaps[cmap])

    norm0 = Normalize(vmin0, vmax0)
    norm1 = Normalize(vmin1, vmax1)

    def func(data):
        data0, data1 = data
        d0 = np.clip(norm0(data0), 0, 1)
        d1 = np.clip(1 - norm1(data1), 0, 1)
        dim0 = np.round(d0 * (cmap.shape[1] - 1))
        # Nans in data seemed to cause weird interaction with conversion to int
        dim0 = np.nan_to_num(dim0).astype(int)
        dim1 = np.round(d1 * (cmap.shape[0] - 1))
        dim1 = np.nan_to_num(dim1).astype(int)

        colored = cmap[dim1.ravel(), dim0.ravel()]
        # May be useful to map r, g, b, a values between 0 and 255
        # to avoid problems with diff plotting functions...?
        if map_to_uint8:
            colored = (colored * 255).astype(np.uint8)
        return colored
    if isinstance(data, type(None)):
        return func
    else:
        return func(data)

def _distances_from_vertex(overlay, start_idx, target_idxs=None, surf_type='fiducial'):
    """Compute distance from vertex to XX

    Parameters
    ----------
    overlay : SVGOverlay object
        svg overlay object for a subject
    start_idx : int
        start index
    target_idxs : list, optional
        other indices idk, by default None
    surf_type : str, optional
        type of surface on which to compute distance, by default 'fiducial'

    Returns
    -------
    _type_
        _description_
    """    """"""
    n_vertices_left, n_vertices_right = get_num_hemi_vertices(overlay)
    subject_id = overlay.svgfile.split('/')[-2]
    surfs = [polyutils.Surface(*d) for d in db.get_surf(subject_id, surf_type)]
    hemi = start_idx >= n_vertices_left
    if surf_type=='flat':
        diffs = surfs[hemi].pts - surfs[hemi].pts[start_idx-n_vertices_left*hemi]
        distances = np.linalg.norm(diffs[...,:2], axis=-1)
    else:
        distances = surfs[hemi].geodesic_distance(start_idx-hemi*n_vertices_left)
    if not isinstance(target_idxs, type(None)):
        hemi_target_idxs = [idx-hemi*n_vertices_left for idx in target_idxs if (hemi==0 and idx < n_vertices_left) or (hemi==1 and idx >= n_vertices_left)]
        if len(hemi_target_idxs) > 0:
            distances = distances[hemi_target_idxs]
    return distances

def _angles_from_vertex(overlay, start_idx, target_idxs=None, period=2*np.pi, theta_direction='ccw'):
    """compute angle between two points

    Parameters
    ----------
    overlay : SVGOverlay object
        svg overlay object for a subject
    start_idx : int
        index for center vertex from which to compute angles
    target_idxs : list, optional
        index for anchor vertex to which to compute angles, by default None
    period : scalar, optional
        max angle, by default 2*np.pi
    theta_direction : str, optional
        direction for positive angles, by default 'ccw'

    Returns
    -------
    _type_
        _description_
    """    
    diff = overlay.coords - overlay.coords[start_idx]
    angles = np.arctan2(diff[:,1], diff[:,0]) + period/4
    if theta_direction not in (-1, 0, 'counter-clockwise', 'ccw'):
        angles = (-angles)%period
    if target_idxs is not None:
        angles = angles[target_idxs]
    return (angles + period*3/4) % period


def get_roi_paths(overlay, roi, cleaned=True, filter_zeros=True, vertical_flip=True, overlay_file=None):
    if roi in list(overlay.rois.shapes.keys()):
        paths = overlay.rois[roi].splines
    elif roi in list(overlay.sulci.shapes.keys()):
        paths = overlay.sulci[roi].splines
    else:
        raise ValueError(
            f"{roi} is not a valid ROI or sulcus in the overlay file.")
    if cleaned:
        paths = [path.cleaned() for path in paths]
    if filter_zeros:
        for path in paths:
            path.vertices = nonzero_coords(path.vertices)
    if vertical_flip:
        paths = [vertical_flip_path(path, overlay) for path in paths]
    hemi_paths = [[], []]
    for path in paths:
        hemi = determine_path_hemi(overlay, path)
        hemi_paths[hemi].append(path)
    return hemi_paths

def get_roi_centroids(overlay, roi, return_indices=True):
    """get centroids of a given ROI in both hemispheres

    Parameters
    ----------
    overlay : SVGOverlay object
        svg for a given subject
    roi : str
        name for region or sulcus to choose
    return_indices : bool, optional
        whether to return indices (True) or coordinates (False),
        by default True

    Returns
    -------
    centroids : list
        coordinate or index for centroids
    """    
    centroids = []
    for hemi_paths in get_roi_paths(overlay, roi):
        centroid = center_of_mass(np.concatenate([path.vertices for path in hemi_paths]))
        if return_indices:
            centroid = get_closest_vertices(overlay, centroid).item()
        centroids.append(centroid)
    return centroids


def get_roi_outlines(overlay, roi, distance_metric='euclidean', return_indices=False):
    paths = get_roi_paths(overlay, roi)
    outlines=[[],[]]
    for hemi in range(2):
        for path in paths[hemi]:
            path_vertices = path.vertices
            if return_indices:
                path_vertices = get_closest_vertices(overlay, path_vertices, distance_metric=distance_metric)
            outlines[hemi].append(path_vertices)
    return outlines


def get_closest_vertices(overlay, coords, hemi='both', distance_metric='euclidean'):
    """Get vertex idxs nearest to the points in a set of coords (e.g. from overlay.sulci or overlay.rois)

    Parameters
    ----------
    overlay : cortex.svgoverlay.SVGOverlay
        Pycortex overlay for subject
    coords : matplotlib.path.Path or np.ndarray
        Set of coords on flatmap
    metric : str, optional
        Distance metric to pass to sklearn for choosing nearest vertices. 
        Defaults to 'euclidean'.

    Returns
    -------
    np.ndarray
        Array of idxs for the vertex nearest to each point in coords.
    """  
    if isinstance(coords, matplotlib.path.Path):
        coords = coords.vertices
    elif not isinstance(coords, np.ndarray):
        raise ValueError("coords should be a matplotlib.path.Path or an np.ndarray")
    if coords.ndim == 1:
        coords = coords[np.newaxis]
    overlay_coords = overlay.coords.copy()
    if hemi!='both':
        n_vertices_left, n_vertices_right = get_num_hemi_vertices(overlay, 'flat')
        if hemi in [0,'l','left']:
            overlay_coords = overlay_coords[:n_vertices_left]
        elif hemi in [1,'r','right']:
            overlay_coords = overlay_coords[n_vertices_left:]
        else:
            raise ValueError("hemi must be 0, 1, 'left', 'right' or 'both'.")
    min_idxs = np.nanargmin(_pairwise_distances(coords,overlay_coords,metric=distance_metric), axis=1)
    if hemi in [1,'r','right']:
        min_idxs += n_vertices_left
    return min_idxs


def _get_closest_vertex_to_roi(overlay, roi, comparison_roi, roi_full=False, comparison_roi_full=False, distance_metric='euclidean', return_indices=True):
    all_coords = overlay.coords
    roi_verts = _get_roi_verts(overlay, roi, full=roi_full)
    comparison_roi_verts = _get_roi_verts(
        overlay, comparison_roi, full=comparison_roi_full)
    nearest_vertices = []
    for hemi in range(2):
        hemi_roi_coords = all_coords[roi_verts[hemi]]
        hemi_comparison_roi_coords = all_coords[comparison_roi_verts[hemi]]
        distances = _pairwise_distances(
            hemi_roi_coords, hemi_comparison_roi_coords, metric=distance_metric)
        nearest_vertex = np.nanargmin(np.nanmin(distances, axis=1))
        if return_indices:
            nearest_vertices.append(roi_verts[hemi][nearest_vertex])
        else:
            nearest_vertices.append(hemi_roi_coords[nearest_vertex, :])
    return nearest_vertices


def _get_roi_verts(overlay, roi, full=True, distance_metric='euclidean', overlay_file=None):
    """Like the one in cortex.utils, but works for sulci and gyri too.

    Parameters
    ----------
    overlay : SVGOverlay
        Pycortex overlay for subject
    roi : str, list or None, optional
        ROIs to fetch. Can be ROI name (string), a list of ROI names, or
        None, in which case all ROIs will be fetched.
    mask : bool
        if True, return a logical mask across vertices for the roi
        if False, return a list of indices for the ROI

    Returns
    -------
    roi_dict : dict
        Dictionary of {roi name : roi verts}. ROI verts are for both
        hemispheres, with right hemisphere vertex numbers sequential
        after left hemisphere vertex numbers.
    """
    # Get overlays
    n_vertices_left, n_vertices_right = get_num_hemi_vertices(overlay)

    if full:
        if roi in list(overlay.rois.shapes.keys()):
            roi_verts = overlay.rois.get_mask(roi)
        elif roi in list(overlay.sulci.shapes.keys()):
            roi_verts = overlay.sulci.get_mask(roi)
        else:
            raise ValueError(f"ERROR: {roi} is not a valid ROI or sulcus.")
    else:
        paths = get_roi_paths(overlay, roi)
        roi_verts = np.concatenate(
            [get_closest_vertices(overlay, path, distance_metric=distance_metric)
             for path in flatten_list_of_lists(paths)]
        )
    hemi_roi_verts = [
        roi_verts[roi_verts < n_vertices_left],
        roi_verts[roi_verts >= n_vertices_left],
    ]
    return hemi_roi_verts

## Core functions
def compute_eccentricity_angle_masks(overlay, centroids, eccentricities, angles):
    """
    Compute dartboard-style masks for both hemispheres.

    Parameters
    ----------
    overlay : str
        pycortex overlay file for subject
    centroids : list or tuple
        Center vertices indices, one for each hemisphere.
    eccentricities : list or tuple
        Two lists of eccentricities for eccentricity bins, one for each hemisphere.
        Should include lower and upper bounds, e.g. (0, 10, 20) will yield two bins from 0-10 and 10-20.
    angles : list
        Two lists of angles (degrees) for angle bins, one for each hemisphere.
        Should include repeat of start angle, e.g., (0, 180, 270, 360) will yield three bins from
        0-180, 180-270, 270-360.

    Returns
    -------
    np.ndarray
        Vertex masks, one for each bin and hemisphere.
        Shape will be (2, len(eccentricities), len(angles), len(vertices)).
    """
    n_vertices_left, n_vertices_right = get_num_hemi_vertices(overlay)
    vertex_distances = [_distances_from_vertex(overlay, centroids[hemi]) for hemi in range(2)]
    vertex_angles = [_angles_from_vertex(overlay, centroids[hemi], theta_direction=hemi) for hemi in range(2)]
    masks = np.zeros((2, np.array(eccentricities).shape[-1]-1,
                      np.array(angles).shape[-1]-1, n_vertices_left+n_vertices_right), dtype=bool)
    for ecc_i in range(np.array(eccentricities).shape[-1]-1):
        for angle_i in range(np.array(angles).shape[-1]-1):
            angle_mask = angle_between(angles[0][angle_i], angles[0][angle_i+1], vertex_angles[0][:n_vertices_left], period=2*np.pi)
            eccentricity_mask = (eccentricities[0][ecc_i] <= vertex_distances[0]) & (vertex_distances[0] < eccentricities[0][ecc_i+1])
            masks[0, ecc_i, angle_i, :n_vertices_left] =  angle_mask & eccentricity_mask
            angle_mask = angle_between(angles[1][angle_i], angles[1][angle_i+1], vertex_angles[1][n_vertices_left:], period=2*np.pi)
            eccentricity_mask = (eccentricities[1][ecc_i] <= vertex_distances[1]) & (vertex_distances[1] < eccentricities[1][ecc_i+1])
            masks[1, ecc_i, angle_i, n_vertices_left:] =  angle_mask & eccentricity_mask
    return masks.astype(bool)


def apply_masks(data, masks, mean_func=np.nanmean, cutoff=None):
    """Given vertex data and eccentricity-angle masks, extracts and averages data for each mask.

    Parameters
    ----------
    data : np.ndarray
        Array of vertex data. Can have any number of dimensions, as long as the last dimension matches that of the masks.
    masks : np.ndarray
        Vertex masks for each bin and hemisphere, such as the output of get_eccentricity_angle_masks. Shape should be (2, eccentricities, angles, vertices).
    mean_func : function, optional
        Function to average values of vertices within a bin. Defaults to np.nanmean.
    cutoff : int or float, optional
        Cutoff value for the minimum number of vertices for a bin. If an int, the threshold will be based on the absolute number of vertices. If a float, the threshold will be based on the percentage of vertices included per bin. Bins with fewer vertices than the cutoff will return np.nan. Defaults to None.

    Returns
    -------
    np.ndarray
        Average vertex values per bin. Will be of shape (data.shape[:-1], # vertices).
    """
    values = np.zeros((*data.shape[:-1], *masks.shape[:-1]))
    for hemi in range(masks.shape[0]):
        for eccentricity in range(masks.shape[1]):
            for angle in range(masks.shape[2]):
                mask = masks[hemi, eccentricity, angle]
                if isinstance(cutoff, type(None)):
                    set_to_nan = False
                elif isinstance(cutoff, float):
                    print((~np.isnan(data[..., mask])).mean())
                    set_to_nan = (~np.isnan(data[..., mask])).mean() < cutoff
                elif isinstance(cutoff, int):
                    set_to_nan = (~np.isnan(data[..., mask])).sum() < cutoff
                if set_to_nan:
                    values[..., hemi, eccentricity, angle] = np.nan
                else:
                    values[..., hemi, eccentricity, angle] = mean_func(
                        data[..., mask], axis=-1)
    return values


def show_dartboard(data, 
        data2=None,
        axis=None,
        image_resolution=500,
        cmap=plt.cm.inferno,
        vmin=None,
        vmax=None,
        vmin2=None,
        vmax2=None,
        theta_direction=-1,
        show_grid=True,
        max_radius=None,
        grid_linewidth=0.5,
        grid_linecolor='lightgray'):
    """Given values masked by angle and eccentricity, shows them as a radial grid ('dartboard'-style visualization).

    Parameters
    ----------
    data : np.ndarray
        Average vertex values per bin. Should be of shape (#eccentricities, #angles).
    data2 : np.ndarray, optional
        Average vertex values per bin. Should be of shape (#eccentricities, #angles).
    axis : plt.Axes, optional
        Matplotlib axis on which to plot. If None, a new axis will be created. Defaults to None.
    image_resolution : int, optional
        Resolution of dartboard figure. Defaults to 500.
    linewidth : float, optional
        Linecolor for bin outlines. Defaults to 0.5.
    cmap : plt.colors.ListedColormap or str, optional
        Colormap for data. Can be a colormap object or a string, which will be loaded through
        matplotlib or pycortex. Can be 2d. Defaults to plt.cm.inferno.
    vmin : float, optional
        vmin for the first data dimension. Defaults to None.
    vmax : float, optional
        vmax for the first data dimension. Defaults to None.
    vmin2 : float, optional
        vmin for the optional second data dimension. Defaults to None.
    vmax2 : float, optional
        vmax for the optional second data dimension. Defaults to None.
    theta_direction : int or str, optional
        Direction by which data wraps around the dartboard. 'clockwise' = 'cw' = 1,
        'counter-clockwise' = 'ccw' = -1. Defaults to 1.
    show_grid : bool, optional
        Whether to show the grid between bins. Defaults to True.

    Returns
    -------
    plt.Axes
        Matplotlib axis in which data is plotted.
    """
    if max_radius is None:
        max_radius = 1
    data = np.array(data).astype(np.float)
    if isinstance(data2, np.ndarray):
        data = np.stack([data, data2], axis=0)
    else:
        data = data[np.newaxis]
    if isinstance(cmap, str):
        if data.shape[0] == 1:
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError:
                cmap = get_cmap(cmap)
        elif data.shape[0] == 2:
            cmap = colormap_2d(cmap, vmin0=0, vmin1=0, vmax0=1, vmax1=1,)
    if vmin is None:
        vmin = np.nanmin(data[0])
    if vmax is None:
        vmax = np.nanmax(data[0])
    nrm = Normalize(vmin=vmin, vmax=vmax)
    data[0] = np.clip(nrm(data[0]), 0, 1)
    if data.shape[0] == 2:
        if vmin2 is None:
            vmin2 = np.nanmin(data[1])
        if vmax2 is None:
            vmax2 = np.nanmax(data[1])
        nrm = Normalize(vmin=vmin2, vmax=vmax2)
        data[1] = np.clip(nrm(data[1]), 0, 1)
    if axis is None:
        _, axis = plt.subplots()
    n_radii, n_angles = data[0].shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(0, max_radius, n_radii + 1)
    dartboard = np.full((image_resolution, image_resolution, 4), np.nan)
    # Define image
    image_t = np.linspace(-max_radius, max_radius, image_resolution)
    image_x, image_y = np.meshgrid(image_t, image_t)
    radius = np.sqrt(image_x**2 + image_y**2)
    theta = np.arctan2(image_x, image_y)
    theta = np.pi * 2-np.mod(theta + np.pi * 5/2, np.pi*2)
    # Define lines
    x_lines, y_lines, _ = sph2cart(
        np.ones_like(angles) * max_radius, np.degrees(angles), np.zeros_like(angles))
    # Loop to fill dartboard
    for radius_i, (radius_small, radius_large) in enumerate(zip(radii[:-1], radii[1:])):
        for angle_i, angle in enumerate(angles):
            ang_i = ((theta-angle) >= 0)
            rad_i = (radius <= radius_large) & (radius > radius_small)
            patch_color = cmap(data[:, radius_i, angle_i])
            dartboard[ang_i & rad_i, :] = patch_color
    if theta_direction in (-1, 'counter-clockwise', 'ccw'):
        dartboard = np.flip(dartboard, axis=1)
    axis.imshow(dartboard, extent=(-max_radius,
                max_radius, -max_radius, max_radius))
    if show_grid:
        n_lines = int(n_angles/2)
        for i in range(n_lines):
            if i % (n_angles/4) == 0:
                axis.plot(
                    [x_lines[i], x_lines[i+n_lines]
                     ], [y_lines[i], y_lines[i+n_lines]],
                    grid_linecolor, lw=grid_linewidth*1.25
                )
            else:
                axis.plot(
                    [x_lines[i], x_lines[i+n_lines]
                     ], [y_lines[i], y_lines[i+n_lines]],
                    'grey', lw=grid_linewidth,  #but why grey
                )
        for radius in radii[:-1]:
            circ = plt.Circle((0, 0), radius=radius,
                              edgecolor='grey', facecolor='none', lw=grid_linewidth)
            axis.add_patch(circ)
    circ = plt.Circle(
        (0, 0), radius=radii[-1]*.99, edgecolor=grid_linecolor, facecolor='none', lw=grid_linewidth*2, linestyle='--')
    axis.add_patch(circ)
    axis.axis("off")
    return axis


def interpolate_outlines(phi, rho, resolution=50):
    new_phi = np.linspace(0,2*np.pi,resolution)
    new_rho = np.interp(new_phi, phi, rho, period=2*np.pi)
    return new_phi, new_rho


def resample_roi_outline(
        angles,
        distances,
        resolution=100,
        every_n=5,
        even_sampling_over='polar angle',
):
    """resample roi outline to smooth outline
    
    Relies on angles being monotonically increasing

    Parameters
    ----------
    angles : _type_
        _description_
    even_sampling_over : str, optional
        _description_, by default 'polar angle'
    resolution : int, optional
        number of points to sample along the ROI border, by default 100

    Returns
    -------
    _type_
        _description_
    """
    delta = 1e-5
    # Periodic linear resample (upsample) to assure values from 0 to 2pi
    if even_sampling_over == 'polar angle':
        # Get evenly spaced (resampled) version of angles and distances
        angles_linear_sampling = np.linspace(0, 2 * np.pi - delta, len(angles))
        dist_linear_sampling = np.interp(
            angles_linear_sampling, angles, distances, period=2 * np.pi)
        if every_n is None:
            return angles_linear_sampling, dist_linear_sampling
        # Smooth: Fit a new spline with points sampled at `every_n`
        angle_input = angles_linear_sampling[::every_n]
        angle_input[-1] = angles_linear_sampling[-1]
        dist_input = dist_linear_sampling[::every_n]
        dist_input[-1] = dist_linear_sampling[-1]
        smooth = interpolate.interp1d(
            angle_input, dist_input, kind='cubic', assume_sorted=True, )
        # Get evenly spaced target angles
        angles_out = np.linspace(0, 2 * np.pi - delta, resolution)
        dist_out = smooth(angles_out)
        return angles_out, dist_out

    if even_sampling_over == 'path length':
        ## Resample
        xy = np.array(pol2cart(angles, distances)).T
        path_dist = np.cumsum(
            np.hstack([0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]))
        max_dist = np.max(path_dist)
        interp = interpolate.interp1d(
            path_dist, xy, axis=0, kind='linear', )
        # TO DO: sample evenly between anchor points?
        regular_distances = np.linspace(
            0, max_dist, len(angles))
        xy_resampled = interp(regular_distances)

        if every_n is None:
            # Map back to angles for output
            angles_out, dist_out = cart2pol(*xy_resampled.T)
            return angles_out, dist_out
        ## Smooth
        # Inputs
        xy_input = xy_resampled[::every_n]
        xy_input[-1] = xy_resampled[-1]
        dist_input = regular_distances[::every_n]
        dist_input[-1] = regular_distances[-1]
        # Smooth x,y values
        smooth = interpolate.interp1d(dist_input,
                                      xy_input,
                                      axis=0, kind='cubic', )
        path_dist_out = np.linspace(0, max_dist, resolution)
        xy_out = smooth(path_dist_out)
        # Convert back to angles
        angles_out, dist_out = cart2pol(*xy_out.T)
        return angles_out, dist_out


def _get_interpolated_outlines(overlay, 
    outline_roi, 
    geodesic_distances=True,
    resolution=100,
    even_sampling_over='polar angle',
    every_n=5,
    recache=False,
    verbose=False,
    **dartboard_spec):
    """compute outline of roi for a given dartboard plot

    Parameters
    ----------
    overlay : SVGOverlay object
        svgoverlay for a given subject
    outline_roi : str
        name of ROI to plot (must sexist in overlays.svg)
    center_roi : str
        name of center ROI for dartboard
    anchor_angles : array
        array of angles ??
    geodesic_distances : bool, optional
        Flag for whether to compute geodesic distances for eccentricity bins,
        by default True
    resolution : int, optional
        number of points estimated for outline, by default 100

    Returns
    -------
    dict
        dictionary of roi names (keys) & outlines (x, y coordinates of boundary)
    """
    center_roi = dartboard_spec['center_roi']
    subject = overlay.svgfile.split('/')[-2]    
    dartboard_str = _get_dartboard_str(**dartboard_spec)
    fname = f'{subject}_dartboard_{outline_roi}_outline_{dartboard_str}_resampleby_{even_sampling_over}_sm{every_n}_res{resolution}_geo{geodesic_distances}.npy'
    fpath = os.path.expanduser(os.path.join(CACHE_DIR, subject, 'cache', fname))
    if os.path.exists(fpath) and not recache:
        if verbose:
            print(f'Loading {center_roi} outline from {fpath}')
        interpolated_outlines = np.load(fpath)
        return interpolated_outlines
    else:
        if verbose:
            print(f'Computing dartboard ROI outline for {center_roi}')
    # Compute interpolated outlines, no cache present
    raw_outlines = get_roi_outlines(overlay, outline_roi)
    # Compute coordinates of overall center
    center_coords = get_roi_centroids(overlay, center_roi, return_indices=False)
    center_idxs = get_roi_centroids(overlay, center_roi, return_indices=True)
    # Get outline center coordinates
    outline_centroids_coords = get_roi_centroids(overlay, outline_roi, return_indices=False)
    # Get centroids and angles to anchors
    center_name, centroids, anchor_angles_dict = _compute_centroids_angles_from_spec(
        overlay, verbose=verbose, **dartboard_spec)
    anchor_angles = np.array([anchor_angles_dict[roi[0]] for roi in dartboard_spec['anchors']]).T
    # Above still has redundancy with getting ROI centroids...
    interpolated_outlines = []
    for hemi in range(2):
        raw_outline = raw_outlines[hemi][0]
        raw_outline_centroid = outline_centroids_coords[hemi]
        # Subtract overall center and convert to polar based on center roi centroids
        phis, rhos = cart2pol(*(raw_outline-center_coords[hemi]).T)
        centroid_phi, centroid_rho = cart2pol(*(raw_outline_centroid-center_coords[hemi]))
        # Flip right hemi
        if hemi:
            phis = np.pi - phis
            centroid_phi = np.pi - centroid_phi
        # Change rhos to geodesic distances
        if geodesic_distances:
            outline_centroid_idxs = get_closest_vertices(overlay, raw_outline_centroid)
            centroid_rho = _distances_from_vertex(overlay, center_idxs[hemi], outline_centroid_idxs)
            raw_outline_idxs = get_closest_vertices(overlay, raw_outline)
            rhos = _distances_from_vertex(overlay, center_idxs[hemi], raw_outline_idxs)
        # Warp angles based on anchor centroids
        warped_phis = warp_phis(phis, anchor_angles[hemi])
        warped_centroid_phi = warp_phis(centroid_phi, anchor_angles[hemi])
        # Convert back to cartesian
        x, y = pol2cart(warped_phis, rhos)
        centroid_x, centroid_y = pol2cart(warped_centroid_phi, centroid_rho)
        # Subtract own centroid
        x -= centroid_x
        y -= centroid_y
        # Convert to polar relative to these centroids
        relative_phi, relative_rho = cart2pol(x,y)
        # Even angular interpolation
        ## CHANGED HERE: May require re-ordering of angles, may rest on assumption
        ## about what angle is first
        #relative_phi_interp, relative_rho_interp = interpolate_outlines(relative_phi, relative_rho, resolution=resolution)
        relative_phi_interp, relative_rho_interp = resample_roi_outline(
            relative_phi, 
            relative_rho,
            resolution=resolution,
            every_n=every_n,
            even_sampling_over=even_sampling_over,
            )
        # Convert interpolated points back to cartesian
        x_interp, y_interp = pol2cart(relative_phi_interp, relative_rho_interp)
        # Add back own centroid
        x_interp += centroid_x
        y_interp += centroid_y
        interpolated_outlines.append(np.stack(cart2pol(x_interp,y_interp), axis=1))
    # cache result
    if verbose:
        print(f'Saving {center_roi} outline to {fpath}')
    np.save(fpath, interpolated_outlines)

    return interpolated_outlines

def show_outlines_on_dartboard(
    outlines, axis=None, colors=None, polar=True, hemi=0, rmax=None, as_overlay=False, 
    **plot_kwargs):
    """Plot outlines of regions of interest on dartboard plots.

    Parameters
    ----------
    outlines : _type_
        _description_
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis on which to plot.
    colors : _type_, optional
        List of colors to iterate through when plotting outlines. If None, will iterate through
        the default pyplot color cycle. Defaults to None.
    polar : bool, optional
        Whether outlines are in polar coordinates. Defaults to True.
    hemi : int, optional
        Hemisphere of the brain from which outlines are drawn.
        0 = left, 1 = right. Defaults to 0.
    rmax : int or float, optional
        Polar maximum radius. Defaults to None.
    as_overlay : bool, optional
        Whether to create a new axis on top of the existing axis, to avoid interfering with
        previously-plotted data. Defaults to False.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis in which data is plotted.
    """
    if axis is None:
        fig, axis = plt.subplots(
            subplot_kw={'projection': 'polar' if polar else 'rectilinear'})
    elif as_overlay:
        fig = axis.figure
        rect = axis.get_position()
        axis = fig.add_axes(rect, polar=polar, frameon=False)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    elif isinstance(colors, str):
        colors = [colors]*len(outlines)
    for outline, color in zip(outlines, colors):
        axis.plot(*outline.T, c=color, **plot_kwargs)
    axis.grid(False)
    if polar:
        axis.spines['polar'].set_visible(False)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    if polar:
        if hemi in ['right', 'rh', 'r', 1]:
            axis.set_theta_zero_location('W')
            axis.set_theta_direction(-1)
        if rmax is not None:
            axis.set_rmax(rmax)
    return axis

# Need all inputs to spec dartboard. Make dartboard_spec a thing? a dict?
def show_dartboard_pair(dartboard_data, 
                        dartboard_data2=None,
                        subject=None,
                        axes=None,
                        rois=None,
                        # plot parameters
                        cmap=plt.cm.viridis,
                        vmin=None,
                        vmax=None,
                        vmin2=None,
                        vmax2=None,
                        image_resolution=500,
                        figsize=(8, 4),
                        # grid line parameters
                        show_grid=True,
                        grid_linecolor='gray',
                        grid_linewidth=0.5,
                        # roi path line parameters
                        geodesic_distances=True,
                        path_resolution=100,
                        every_n=5,
                        even_sampling_over='path length',
                        roi_linewidth=1,
                        roi_color='r',
                        recache=False,
                        verbose=False,
                        **dartboard_spec,
    ):
    """Make two dartboard plots (one for each hemisphere) of dartboard data

    Includes many options for plotting, including for plotting of
    ROI outlines over dartboard plot. There are several choices to
    make here, since the ROI outlines must be warped appropriately.

    For lower-level functions, see:
    `show_dartboard` # 
    `_get_interpolated_outlines` # get roi outlines
    `show_outlines_on_dartboard` # plot roi outlines

    Parameters
    ----------
    dartboard_data : array or pycortex Vertex object
        if array, should be (); if vertex object, vertex data will be masked
        according to `dartboard_spec` kwargs
    dartboard_data2 : array or pycortex Vertex object, optional
        optional second dimension of data for 2D colormap plots; 
        see `dartboard_data` for format, by default None
    subject : str, optional
        string identifier for pycortex subject; not necessary if a vertex object
        is provided for dartboard_data, by default None
    axes : matplotlib axis, optional
        axis into which to plot; if None, a new plot is created, by default None
    rois : list, optional
        list of string names for ROIs to plot on dartboard space, by default None
    cmap : str or cmap, optional
        colormap or name of colormap to use to colormap data. string names for 
        pycortex colormaps, including 2D colormaps, are allowable, by default plt.cm.viridis
    vmin : scalar, optional
        vmin for first dimension of data to be plotted on dartboard, by default None
    vmax : scalar, optional
        vmax for first dimension of data to be plotted on dartboard, by default None
    vmin2 : scalar, optional
        vmin for second dimension of data to be plotted on dartboard, by default None
    vmax2 : scalar, optional
        vmax for second dimension of data to be plotted on dartboard, by default None
    image_resolution : int, optional
        resolution of dartboard images, by default 500
    figsize : tuple, optional
        size in inches of resulting plot; should be twice as wide as it is tall, by default (8, 4)
    show_grid : bool, optional
        whether to show lines between dartboard bins, by default True
    grid_linecolor : str, optional
        colorspec for grid lines, by default 'gray'
    grid_linewidth : float, optional
        width of grid lines, by default 0.5
    geodesic_distances : bool, optional
        whether to compute distances along folded cortical surface (True), or to
        simply compute distances in flattened space (False). True is slower, by default True
    path_resolution : int, optional
        resolution of ROI paths plotted on dartboard in points, by default 100
    every_n : int, optional
        sampling along ROI path for smoothing; ROI paths are warped into dartboard
        space, a little smoothing usually helps aesthetically. 1 is no smoothing
        , by default 5
    even_sampling_over : str, optional
        How to resample ROI paths, by angle or along path length. 'angle' is perhaps slightly
        more principled for convex ROIs, but 'path_length' gives better results for non-convex
        ROIs, by default 'path length'
    roi_linewidth : int, optional
        width of lines for drawn ROIs, by default 1
    roi_color : list or str, optional
        color of lines for drawn ROIs; if list is provided, ROIs are each colored
        in order of the colors, by default 'r'
    recache : bool, optional
        recache mask and ROI outline data (data is cached in the pycortex cache
        to speed up processing), by default False
    verbose : bool, optional
        verbose output, by default False

    Returns
    -------
    list of matplotlib.axes
        list of axes into which data was plotted

    Raises
    ------
    ValueError
        _description_
    """    """"""
    if subject is None:
        if isinstance(dartboard_data, Vertex):
            subject = dartboard_data.subject
        else:
            if rois is not None:
                raise ValueError("Must provide either `subject` argument or a Vertex object as `dartboard_data` if you wish to display rois")

    if isinstance(dartboard_data, Vertex):
        _, data0 = get_dartboard_data(dartboard_data, **dartboard_spec)
    else:
        data0 = dartboard_data.copy()
    if dartboard_data2 is None:
        # None for left hem, None for right hem
        data1 = (None, None)
    elif isinstance(dartboard_data2, Vertex):
        _, data1 = get_dartboard_data(dartboard_data2, **dartboard_spec)
    else:
        data1 = dartboard_data2.copy()
        
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Loop over hemispheres
    # Hemisphere index goes (0, 1) = (left, right)
    directions = [-1, 1]
    max_radii = dartboard_spec['max_radii']
    if not isinstance(max_radii, (list, tuple)):
        max_radii = [max_radii, max_radii]
    for hemi_index, data in enumerate(zip(data0, data1)):
        max_radius = max_radii[hemi_index]
        axis = axes[hemi_index]
        d0, d1 = data
        _ = show_dartboard(d0, data2=d1,
            axis=axis,
            theta_direction=directions[hemi_index],
            image_resolution=image_resolution,
            cmap=cmap,
            show_grid=show_grid,
            grid_linewidth=grid_linewidth,
            grid_linecolor=grid_linecolor,
            max_radius=max_radius,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            )
    # optionally plot ROIs
    if rois is not None:
        overlay = db.get_overlay(subject)
        outlines = {}
        if not isinstance(roi_color, (list, tuple)):
            roi_color = [roi_color] * len(rois)
        for roi in rois:
            outlines[roi] = _get_interpolated_outlines(overlay,
                                                        roi,
                                                        geodesic_distances=geodesic_distances,
                                                        resolution=path_resolution,
                                                        even_sampling_over=even_sampling_over,
                                                        every_n=every_n,
                                                        recache=recache,
                                                        verbose=verbose,
                                                        **dartboard_spec)
        for hemi_index in range(2):
            hemi_axis = axes[hemi_index]
            outlines_to_plot = np.array([v for v in outlines.values()])[:, hemi_index]
            if isinstance(dartboard_spec['max_radii'], tuple):
                rmax = dartboard_spec['max_radii'][hemi_index]
            else:
                rmax = dartboard_spec['max_radii']
            show_outlines_on_dartboard(
                outlines_to_plot,
                hemi_axis,
                hemi=hemi_index,
                rmax=rmax,
                as_overlay=True,  # As overlay
                linewidth=roi_linewidth,
                colors=roi_color)
    return axes


def draw_mask_outlines(
    overlay, masks, axis=None, eccentricities=True, angles=True, 
    outer_line=True, as_overlay=True, **plot_kwargs):
    """Given eccentricity-angle masks, fits and draws outlines of bins on the cortex.

    Args:
        masks np.ndarray: Vertex masks for each bin and hemisphere, such as the output of
            get_eccentricity_angle_masks. Shape should be (2, eccentricities, angles, vertices).
        axis (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlb axis on which to plot.
        eccentricities (bool, optional): Whether to draw outlines for separate eccentricities.
            Defaults to True.
        angles (bool, optional): Whether to draw outlines for separate angles.
            Defaults to True.
        outer_line (bool, optional): Whether to draw outline of outermost eccentricity.
            Defaults to True.
        as_overlay (bool, optional): Whether to create new axis on top of existing axis, to avoid
            interfering with previously-plotted data. Defaults to False.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Matplotlib axis in which data is plotted.
    """
    if axis is None:
        _, axis = plt.subplots()
    elif as_overlay:
        axis = overlay_axis(axis)

    def fit_and_draw_hull(masks):
        for mask in masks:
            coords = overlay.coords[mask == 1]
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                axis.plot(coords[simplex, 0],
                          coords[simplex, 1], **plot_kwargs)
    if eccentricities:
        fit_and_draw_hull(masks[:, :-1].sum(2).reshape(-1, masks.shape[-1]))
    if angles:
        fit_and_draw_hull(masks.sum(1).reshape(-1, masks.shape[-1]))
    if outer_line:
        fit_and_draw_hull(masks[:, -1].sum(1).reshape(-1, masks.shape[-1]))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0, overlay.svgshape[0])
    axis.set_ylim(0, overlay.svgshape[1])
    axis.set_aspect(1)
    return axis


def draw_mask_bins(
    overlay, masks, axis=None, eccentricities=True, angles=True, 
    as_overlay=True, cmap='gray', alpha=.5, **plot_kwargs):
    """
    Given eccentricity-angle masks, visualizes bins on the cortex as alternating colors on the cortex.

    Parameters
    ----------
    masks : np.ndarray
        Vertex masks, one for each bin and hemisphere.
        Shape should be (2, eccentricities, angles, vertices).
    axis : matplotlib.axes._subplots.AxesSubplot or None, optional
        Matplotlib axis on which to plot.
    eccentricities : bool, optional
        Whether to color bins for separate eccentricities. Defaults to True.
    angles : bool, optional
        Whether to color bins for separate angles. Defaults to True.
    as_overlay : bool, optional
        Whether to create a new axis on top of the existing axis, to avoid interfering with previously-plotted data.
        Defaults to False.
    cmap : matplotlib.colors.ListedColormap or str, optional
        Colormap to set colors for bins, which will have values of either 0 or 1. Defaults to 'gray'.
    alpha : float, optional
        Transparency value, for overlaying on flatmaps. Defaults to 0.5.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis in which data is plotted.
    """
    if axis is None:
        _, axis = plt.subplots()
    elif as_overlay:
        axis = overlay_axis(axis)
    # Sum over hemispheres
    masks = masks.sum(0)
    if not eccentricities:
        masks = masks.sum(0)
        mod = 1
    else:
        mod = masks.shape[1]
    if not angles:
        masks = masks.sum(-2)
    masks = masks.reshape(-1, masks.shape[-1])
    bins = np.full(masks.shape[-1], np.nan)
    for i, mask in enumerate(masks):
        bins[mask == 1] = (i + i//mod) % 2
    bins_vx = Vertex(bins, overlay.svgfile.split('/')[-2], vmin=0, vmax=1)
    img_grid, extent = quickflat.composite.make_flatmap_image(bins_vx)
    axis.imshow(img_grid, extent=extent, alpha=alpha, cmap=cmap, **plot_kwargs)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect(1)
    return axis


def draw_anchor_lines(
    overlay, center_idxs, anchor_idxs, axis=None, as_overlay=True, **plot_kwargs):
    """
    Draw lines over the cortex stretching from the center of one ROI to the centers of other 'anchor' ROIs.

    Parameters
    ----------
    overlay : str
        Subject identifier string, e.g. 'S1fs'.
    center_roi : str
        ROI from which to draw lines.
    anchor_rois : list or tuple
        ROIs to which to draw lines from center_roi.
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis on which to plot.
    as_overlay : bool, optional
        Whether to create a new axis on top of the existing axis to avoid interfering with previously-plotted data. Defaults to False.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis in which data is plotted.
    """
    default_kwargs = {
        'color': 'lightgray',
        'linewidth': 1.5,
    }
    default_kwargs.update(plot_kwargs)
    if axis is None:
        _, axis = plt.subplots()
    elif as_overlay:
        axis = overlay_axis(axis)
    for hemi in range(2):
        center_coords = overlay.coords[center_idxs[hemi]]
        for anchor_idx in anchor_idxs[hemi]:
            anchor_coords = overlay.coords[anchor_idx]
            axis.plot(
                [center_coords[0], anchor_coords[0]],
                [center_coords[1], anchor_coords[1]],
                **default_kwargs)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0, overlay.svgshape[0])
    axis.set_ylim(0, overlay.svgshape[1])
    axis.set_aspect(1)
    return axis


def generate_dartboard_vertex_object(dartboard_mask,
                                     subject,
                                     type='grid',
                                     bg_value=np.nan,
                                     cmap='gray'):
    """Make a vertex object containing index values for parts of the dartboard grid
    
    Parameters
    ----------
    dartboard_mask : array
        array of masks for each region of dartboard grid, of size 
        (2, n_eccentricites, n_angles, n_vertices). 2 is hemispheres. 
    subject : str
        pycortex subject surface ID
    type : str
        one of: 'grid', 'solid', 'eccentricity', 'angle'
    n_eccentricities : scalar int
        number of eccentricity bins
    n_angles : scalar int
        number of radial bins
    bg_value : float
        value for non-dartboard locations
    cmap : str
        color map to use for pycortex vertex object returned
    
    TODO:
    ----
    Optionally split hemispheres? 
    """
    assert dartboard_mask.ndim == 4, \
        'dartboard mask must be (2 x eccentricities x angles x vertices)'
    _, n_eccentricities, n_angles, n_vertices = dartboard_mask.shape
    mask_img = np.zeros((n_vertices,), ) * bg_value
    for ecc_i in range(n_eccentricities):
        #ang_i = j % n_angles
        #ecc_i = j // n_angles
        for ang_i in range(n_angles):
            if type == 'grid':
                if ecc_i % 2:
                    value = ang_i % 2
                else:
                    value = 1 - (ang_i % 2)
            elif type == 'solid':
                value = 1
            elif type == 'eccentricity':
                value = ecc_i
            elif type == 'angle':
                # Maybe revisit me
                value = ang_i
            #value = ang_i
            for hem in [0, 1]:  # left, right
                mask_img[dartboard_mask[hem, ecc_i, ang_i, :]] = value
    vx = Vertex(mask_img, subject, vmin=0,
                   vmax=np.nanmax(mask_img), cmap=cmap)
    return vx

def overlay_dartboards(
    data, data_2=None, axis=None, fig=None, position_x=.5, position_y=.75, scale=.25, **dartboard_kwargs):
    """
    Show dartboard data in a position overlaid on an existing axis.  Primarily intended for
    overlaying dartboards on corresponding flatmaps.

    Parameters
    ----------
    data : np.ndarray
        Average vertex values per bin. Should be of shape (# eccentricities, # angles) for single-/cross-hemi data, and (#hemis, # eccentricities, # angles) for two-hemi data.
    data_2 : np.ndarray, optional
        Average vertex values per bin. Should be of shape (# eccentricities, # angles) for single-/cross-hemi data, and (#hemis, # eccentricities, # angles) for two-hemi data. Defaults to None.
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        Matplotlib axis on which to plot.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure in which data is plotted.
    x : float, optional
        X position in the range of (0,1) for the center of dartboards. Defaults to 0.5.
    y : float, optional
        Y position in the range of (0,1) for the center of dartboards. Defaults to 0.5.
    scale : float, optional
        Scale of dartboards relative to the axis size. Defaults to 0.25.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis in which data is plotted.
    """
    overlaid_axis = axis.figure.add_axes(
        mtrans.Bbox([
            [position_x-scale/2, position_y-scale/2], 
            [position_x+scale/2, position_y+scale/2]]),
        frameon=False
    )

    if data.ndim == 2:
        show_dartboard(data, data_2, axis=overlaid_axis,
                        **dartboard_kwargs)
        return overlaid_axis
    elif data.ndim == 3:
        axis_left, axis_right = split_axis(overlaid_axis, fig)
        show_dartboard(
            data[0], data_2[0] if not isinstance(data_2, type(None)) else None, axis=axis_left,
            theta_direction=-1, **dartboard_kwargs)
        show_dartboard(
            data[1], data_2[1] if not isinstance(data_2, type(None)) else None, axis=axis_right,
            theta_direction=1, **dartboard_kwargs)
        return axis_left, axis_right


def _get_dartboard_str(**dartboard_spec):
    mx = dartboard_spec['max_radii']
    if not isinstance(mx, tuple):
        mx = (mx, mx)
    strs = []
    for a in dartboard_spec['anchors']:
        if isinstance(a, tuple):
            anchor_name, anchor_type = a
        else:
            anchor_name, anchor_type = a, 'centroid'
        strs.append('{}_{}'.format(anchor_name, anchor_type[0]))
    anchor_str = 'a' + '_'.join(strs)
    fmt = dict(
        center_roi=dartboard_spec['center_roi'],
        max_radii=mx,
        n_angles=dartboard_spec['n_angles'],
        n_eccentricities=dartboard_spec['n_eccentricities'],
        anchor_str=anchor_str,
        rad_str='mx%d_%d' % tuple(mx),
    )
    fname = '_center{center_roi}-anchors_{anchor_str}-{n_angles}ang-{n_eccentricities}ecc-{rad_str}'
    return fname.format(**fmt)


def _get_anchor_points(svg, center_roi, anchors, return_indices=True):
    centroids = {}
    for j, anchor in enumerate([center_roi] + anchors):
        if isinstance(anchor, tuple):
            anchor_name, anchor_type = anchor
        else:
            anchor_name, anchor_type = anchor, 'centroid'
        if j == 0:
            center_name, center_type = anchor_name, anchor_type
        if anchor_type == 'nearest':
            centroids[anchor_name] = _get_closest_vertex_to_roi(svg, anchor_name, center_name, return_indices=return_indices)
        elif anchor_type == 'centroid':
            centroids[anchor_name] = get_roi_centroids(svg, anchor_name, return_indices=return_indices)
        else:
            raise ValueError("unknown anchor type specified: %s\n(Must be 'nearest' or 'centroid')"%(anchor_type))    
    return centroids


def _compute_centroids_angles_from_spec(svg, center_roi, anchors, verbose=False, **kwargs):
    """kwargs catches extra inputs from dartboard_spec. 
    Perhaps not the most principled.
    functional.
    """
    # Compute centroids of each ROI
    t0 = time.time()
    # Compute centroids
    centroids = _get_anchor_points(svg, center_roi, anchors)
    t1 = time.time()
    if verbose:
        print('Time to get centroids:', t1 - t0)
    if isinstance(center_roi, tuple) and (len(center_roi) == 2):
        center_name, _ = center_roi
    else:
        center_name = center_roi
    # Compute angles from center ROI to each of the anchors
    anchor_angles_dict = {}
    for anchor in anchors:
        if isinstance(anchor, tuple):
            anchor_name, anchor_type = anchor
        else:
            anchor_name, anchor_type = anchor, 'centroid'
        anchor_angles_dict[anchor_name] = [_angles_from_vertex(svg, centroids[center_name][hemi], centroids[anchor_name][hemi], theta_direction=hemi) for hemi in range(2)]
    t2 = time.time()
    if verbose:
        print('Time to compute angles:', t2 - t1)
    return center_name, centroids, anchor_angles_dict

def get_dartboard_data(vertex_obj,
    center_roi,
    anchors,
    max_radii=(50, 50),
    n_angles=16,
    n_eccentricities=8,
    eccentricities=None,
    mean_func=np.nanmean,
    cutoff = None,
    recache=False,
    verbose=True,
    ):
    """retrieve dartboard data for a given vertex object and dartboard parameters

    Parameters
    ----------
    vertex_obj : cortex.Vertex
        vertex object containing data to be averaged over the defined radial bins
    center_roi : str or tuple
        name of roi (defined in overlays.svg for this subject) 
    anchors : list
        list of strings or tuples specifying anchor points. If strings are provided,
        code assumes strings to be names for regions of interest defined for this 
        pycortex subject (e.g. 'FFA'). Anchor points are computed as centroids of 
        the named ROIs. If tuples are provided, the first element of the tuple is the
        name of the ROI, and the second element must be 'centroid' or 'nearest', 
        indicating whether the anchor point shoudl be the centroi of the named ROI, or
        the nearest point in or on the named region to the center_roi. Sulci or other
        labeled anatomical markers in other layers of overlays.svg for the subject may
        be used, e.g. ('STS', 'nearest') for the nearest point to the center_roi that
        falls on STS. 

        Important: anchors must be specified counter clockwise from the RIGHT with
        respect to the RIGHT hemisphere:
               2
              /\
        3 <- center -> 1
              \/
               4
    max_radii : int or tuple
        maximum radius to extend from center ROI (in mm in fiducial space). Separate
        values for each hemisphere can be provided as a tuple (Lhem, Rhem).
    mean_func : function, optional
        The function used to average all vertices in a given bin, by default np.nanmean
        Consider what is being averaged to make an appropriate choice here (e.g.
        correlations should be Fischer z transformed before averaging, which may 
        require a custom function)
    eccentricities : _type_, optional
        _description_, by default np.linspace(0, max_radii, 8+1)
    cutoff : int or float, optional
        Cutoff value for the minimum number of vertices for a bin. If an int, the threshold
        will be based on the absolute number of vertices. If a float, the threshold will be
        based on the percentage of vertices included per bin. Bins with fewer vertices than
        the cutoff will return np.nan. Defaults to None.
    
    Returns
    -------
    masks : array
        Masks for each bin of dartboard histogram
    data : array
        One value for each bin of dartboard histogram. Array is
        (hemispheres, eccentricities, angles) starting from 0 degrees
        for first value (VERIFY ME FIX)
    """    
    # Subject
    subject = vertex_obj.subject
    # SVG overlay object
    svg = db.get_overlay(subject)
    # Handle max_radii (should be tuple)
    if not isinstance(max_radii, tuple):
        max_radii = (max_radii, max_radii)
    # Allow manually specified eccentricities to override linear spacing
    if eccentricities is None:
        eccentricities = [np.linspace(0, mr, n_eccentricities + 1) for mr in max_radii]
    # Check for cached files:
    dartboard_spec = dict(center_roi=center_roi,
                          anchors=anchors,
                          n_angles=n_angles,
                          n_eccentricities=n_eccentricities,
                          eccentricities=eccentricities,
                          max_radii=max_radii,)
    dartboard_str = _get_dartboard_str(**dartboard_spec)
    fname = f'{subject}_dartboard_mask_{dartboard_str}.npy'
    fpath = os.path.expanduser(os.path.join(CACHE_DIR, subject, 'cache', fname))
    if os.path.exists(fpath) and not recache:
        if verbose:
            print(f'Loading masks from {fpath}')
        masks = np.load(fpath)
    else:
        # Compute dartboard masks
        if verbose:
            print("Computing dartboard masks...")
        center_name, centroids, anchor_angles_dict = _compute_centroids_angles_from_spec(
            svg, verbose=verbose, **dartboard_spec)

        # Compute the masks, based on specified eccentricity bins, angle bins, and the previously-computed variables
        t2 = time.time()
        # NOTE: magic number here, not great. 4 sub-bins only works specifically 
        # for 4 anchors and 16 total anglular bins. This could be re-computed with
        # an assumption of even spacing, or fundamentally changed by specifying 
        # the desired angles of the anchor points
        masks = compute_eccentricity_angle_masks(
            svg, centroids[center_name],
            eccentricities=eccentricities,
            angles=[
                interpolate_angles(np.array([v for v in anchor_angles_dict.values()])[
                                :, 0], angle_sub_bins=4),
                interpolate_angles(np.array([v for v in anchor_angles_dict.values()])[
                                :, 1], angle_sub_bins=4)
            ],
        )
        t3 = time.time()
        if verbose:
            print('Time to compute masks:', t3-t2)
            print(f'Saving dartboard masks to {fpath}')
        np.save(fpath, masks)
    output = apply_masks(vertex_obj.data, masks, mean_func, cutoff)

    return masks, output


def dartboard_on_flatmap(vertex_data,
                         fn=np.nanmean,
                         vmin=None,
                         vmax=None,
                         cmap=None,
                         # Dartboard args
                         center_roi=None,
                         anchors=None,
                         rois=None,
                         n_angles=16,
                         n_eccentricities=8,
                         max_radii=(50, 50),
                         eccentricities=None,
                         #surf_type='inflated', # Was for choosing which 
                         # Plotting args
                         figsize=(12, 6),
                         dartboard_axes_dist_from_midline=0.15,
                         dartboard_axes_bottom=0.25,
                         dartboard_axes_width=0.1,
                         dartboard_axes_height=0.2,
                         dartboard_display_alpha=0.2,
                         quickflat_kw=None,
                         flatmap_line_linewidth=1.5,
                         flatmap_line_color='c',
                         flatmap_line_style=('--','-','--','-'),
                         show_anchor_lines=None,
                         show_dartboard_grid=True,
                         show_dartboard_edge=True,
                         # ROI outline parameters
                         n_roi_border_points=64,
                         roi_outline_smooth_factor=5,  # every 5th point kept, smoothed with cubic spline
                         roi_border_kw=None,
                         verbose=False,
                         outline_kw=None,
                         ):
    """Make a flatmap with overlaid dartboard plots

    Parameters
    ----------
    vertex_data : cortex.Vertex (or cortex.Vertex2D, or cortex.VertexRGB*) instance
        data to be plotted (VertexRGB still WIP). Note that vmin, vmax of dartboard
        use vmin, vmax of this data.
    masks : array-like, optional
        masks for , by default None
    fn : function, optional
        function to use to collapse over vertices within each mask, by default np.nanmean
    center_roi : str
        center ROI
    anchors : list
        list of strings or tuples specifying anchor points. If strings are provided,
        code assumes strings to be names for regions of interest defined for this 
        pycortex subject (e.g. 'FFA'). Anchor points are computed as centroids of 
        the named ROIs. If tuples are provided, the first element of the tuple is the
        name of the ROI, and the second element must be 'centroid' or 'nearest', 
        indicating whether the anchor point shoudl be the centroi of the named ROI, or
        the nearest point in or on the named region to the center_roi. Sulci or other
        labeled anatomical markers in other layers of overlays.svg for the subject may
        be used, e.g. ('STS', 'nearest') for the nearest point to the center_roi that
        falls on STS. 

        Important: anchors must be specified counter clockwise from the RIGHT with
        respect to the RIGHT hemisphere:
               2
              /\
        3 <- center -> 1
              \/
               4        
    n_eccentricities : int, optional
        number of radial bins, by default 5; should be consistent with number of radial
        bins in `masks`, if `masks` is provided
    n_angles : int, optional
        number of polar angle bins, by default 16; will currently break with anything but
        16 and 4 anchors...
    dartboard_axes_dist_from_midline : float, optional
        fraction of the figure by which dartboard axes are displaced from the figure midline,
        by default 0.15
    dartboard_axes_bottom : float, optional
        fraction of the figure by which dartboard axes are displaced from the figure bottom,
        by default 0.25
    dartboard_axes_width : float, optional
        width of dartboard axes as a fraction of the figure, by default 0.1
    dartboard_axes_height : float, optional
        height of dartboard axes as a fraction of the figure, by default 0.2
    verbose : bool
        whether or not to display verbose output 
    """
    if verbose:
        print("Getting masks...")
    # Allow manually specified eccentricities to override linear spacing
    if eccentricities is None:
        eccentricities = [np.linspace(
            0, mr, n_eccentricities + 1) for mr in max_radii]
    dartboard_spec = dict(center_roi=center_roi,
                          anchors=anchors,
                          n_angles=n_angles,
                          n_eccentricities=n_eccentricities,
                          eccentricities=eccentricities,
                          max_radii=max_radii,)

    # Manage inputs
    if isinstance(vertex_data, Vertex):
        if vmin is None:
            vmin = vertex_data.vmin
        if vmax is None:
            vmax = vertex_data.vmax
    elif isinstance(vertex_data, Vertex2D):
        if vmin is None:
            vmin = (vertex_data.vmin, vertex_data.vmin2)
        if vmax is None:
            vmax = (vertex_data.vmax, vertex_data.vmax2)
    else:
        raise NotImplementedError("No VertexRGB yet!")
    if cmap is None:
        cmap = vertex_data.cmap
    if flatmap_line_style is None:
        flatmap_line_style = '-'
    if quickflat_kw is None:
        quickflat_kw = {'with_curvature' : True, }
    if roi_border_kw is None:
        roi_border_kw = {}
    if outline_kw is None:
        outline_kw = {}
    # Load dartboard masks & masked data
    masks, to_plot = get_dartboard_data(vertex_data, **dartboard_spec, mean_func=fn)
    # Vertex flatmap plot
    fig, ax = plt.subplots(figsize=figsize)
    _ = quickflat.make_figure(
        vertex_data, fig=ax, **quickflat_kw)

    # Augment plot
    if show_dartboard_grid:
        # Grid fill for dartboard area
        vx_grid = generate_dartboard_vertex_object(
            masks, vertex_data.subject, type='grid', bg_value=np.nan)
        img_grid, extent = quickflat.composite.make_flatmap_image(vx_grid)
        ax.imshow(img_grid, extent=extent,
                  alpha=dartboard_display_alpha, cmap='gray')
    if show_dartboard_edge:
        # Solid fill for dartboard area
        vx_fill = generate_dartboard_vertex_object(
            masks, vertex_data.subject, type='solid', bg_value=0)
        img_fill, extent = quickflat.composite.make_flatmap_image(vx_fill)
        xt = np.linspace(extent[0], extent[1], img_fill.shape[1])
        yt = np.linspace(extent[3], extent[2], img_fill.shape[0])
        xg, yg = np.meshgrid(xt, yt)
        ax.contour(xg, yg, img_fill, [1],
                   linewidths=[flatmap_line_linewidth],
                   colors=[flatmap_line_color],
                   zorder=10,  # (in front)
                   )
    if show_anchor_lines:
        overlay = db.get_overlay(vertex_data.subject)
        # Draw lines from center of dartboard to each anchor ROI center
        if not isinstance(flatmap_line_style, (list, tuple)):
            flatmap_line_style = [flatmap_line_style] * len(anchors)
        pts, _ = db.get_surf(vertex_data.subject,
                                'flat', merge=True, nudge=True)
        roi_centers = {}
        roi_center_indices = _get_anchor_points(overlay, center_roi, anchors)
        for roi, c in roi_center_indices.items():
            roi_centers[roi] = np.array([pts[c[0]][:2],
                                         pts[c[1]][:2]])
        center_pt = roi_centers.pop(center_roi)
        for lr in [0, 1]:  # left, right
            for pt, ls in zip(roi_centers.keys(), flatmap_line_style):
                x = [center_pt[lr, 0],
                     roi_centers[pt][lr, 0]]
                y = [center_pt[lr, 1],
                     roi_centers[pt][lr, 1]]
                ax.plot(x, y, lw=flatmap_line_linewidth,
                        ls=ls, color=flatmap_line_color)
    # Replace below with...?
    # overlaid_axis = fig.add_axes(
    #     mtrans.Bbox([
    #         [position_x-scale/2, position_y-scale/2],
    #         [position_x+scale/2, position_y+scale/2]]),
    #     frameon=False
    # )

    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.5 - dartboard_axes_dist_from_midline - dartboard_axes_width,
                                   dartboard_axes_bottom,
                                   dartboard_axes_width,
                                   dartboard_axes_height]
    ax_lhem = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.5 + dartboard_axes_dist_from_midline,
                                   dartboard_axes_bottom,
                                   dartboard_axes_width,
                                   dartboard_axes_height]
    ax_rhem = fig.add_axes([left, bottom, width, height])

    show_dartboard_pair(vertex_data,
                   **dartboard_spec,
                   rois=rois,
                   mean_func=fn,
                   axes=np.array([ax_lhem, ax_rhem]),
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap,
                   #outline_kw=outline_kw,
                   verbose=verbose,
                   #**kwargs,
                   ###
                   )
    if show_anchor_lines:
        # Show vertical and horizontal lines on dartboard plots
        # NOTE: Will need changing if anchors are not meant to specify
        # 90 degree ticks around dartboard. For that, should be something
        # like defining regularly spaced angles:
        # angles = np.linspace(0, 2 * np.pi, n_anchor_points)
        # ... then computing max_radii[blah] * sin and cos of each angle for X and Y
        # ... while following same line order as above.
        ax_rhem.plot([-max_radii[1], max_radii[1]], [0, 0], '--', color=flatmap_line_color,
                     lw=flatmap_line_linewidth)
        ax_rhem.plot([0, 0], [-max_radii[1], max_radii[1]], '-', color=flatmap_line_color,
                     lw=flatmap_line_linewidth)
        ax_rhem.axis([-max_radii[1], max_radii[1], -
                     max_radii[1], max_radii[1]])

        ax_lhem.plot([-max_radii[0], max_radii[0]], [0, 0], '--', color=flatmap_line_color,
                     lw=flatmap_line_linewidth)
        ax_lhem.plot([0, 0], [-max_radii[0], max_radii[0]], '-', color=flatmap_line_color,
                     lw=flatmap_line_linewidth)
        ax_lhem.axis([-max_radii[0], max_radii[0], -
                     max_radii[0], max_radii[0]])

    return fig
