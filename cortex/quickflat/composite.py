import copy
import numpy as np
from .. import dataset
from ..database import db
from ..options import config
from ..svgoverlay import get_overlay
from ..utils import get_shared_voxels, get_mapper
from .utils import _get_height, _get_extents, _convert_svg_kwargs, _has_cmap, _get_images, _parse_defaults
from .utils import make_flatmap_image, _make_hatch_image, _return_pixel_pairs, get_flatmask, get_flatcache

### --- Individual compositing functions --- ###

def add_curvature(fig, dataview, extents=None, height=None, threshold=True, contrast=None,
                  brightness=None, smooth=None, cmap='gray', recache=False, curvature_lims=0.5):
    """Add curvature layer to figure

    Parameters
    ----------
    fig : figure
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    threshold : boolean 
        Whether to apply a threshold to the curvature values to create a binary curvature image
        (one shade for positive curvature, one shade for negative). `None` defaults to value 
        specified in the config file
    contrast : float, [0-1] or None 
        Contrast of curvature image. 1 is maximal contrast (given brightness). If brightness is 0.5
        and contrast is 1, and cmap is 'gray', curvature will be black and white. None defaults
        to value in config file.
    brightness : float, [0-1] or None
        How bright to make average value of curvature (0=black, 1=white in gray cmap). None
        defaults to the value in config file.
    curvature_lims : float
        Limits for real curvature values (actual values for cortical curvature are normalized
        within [-`curvature_lims`, +`curvature_lims`] before scaling by `contrast` and shifting
        by `brightness`). 
    smooth : scalar or None
        Width of smoothing to apply to surface curvature. None defaults to no smoothing, or
        whatever the default value for curvature is that is stored in 
        <filestore>/<subject>/surface-info/curvature.npz (for some subjects initiated in old
        versions of pycortex, this may be smoothed too!)
    cmap : string
        name for colormap of curvature
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data

    """
    from matplotlib.colors import Normalize
    if height is None:
        height = _get_height(fig)
    # Get curvature map as image
    default_smoothing = config.get('curvature','smooth')
    if default_smoothing.lower()=='none':
        default_smoothing = None
    else:
        default_smoothing = np.float(default_smoothing)
    if smooth is None:
        # (Might still be None!)
        smooth = default_smoothing
    if smooth is None:
        # If no value for 'smooth' is given in kwargs, db.get_surfinfo returns
        # the default curvature value, whatever that may be. This is the behavior
        # that we want a None in the code to invoke. This is silly and complicated
        # due to backward compatibility issues with some old subjects.
        curv_vertices = db.get_surfinfo(dataview.subject)
    else:
        curv_vertices = db.get_surfinfo(dataview.subject, smooth=smooth)
    curv, _ = make_flatmap_image(curv_vertices, recache=recache, height=height)
    # First, limit to sensible range for flatmap curvature
    norm = Normalize(vmin=-0.5, vmax=0.5)
    curv_im = norm(curv)
    # Option to use thresholded curvature
    default_threshold = config.get('curvature','threshold').lower() in ('true','t','1','y','yes')
    use_threshold_curvature = default_threshold if threshold is None else threshold
    if use_threshold_curvature:
        curv_im = (np.nan_to_num(curv_im) > 0.5).astype(np.float)
        curv_im[np.isnan(curv)] = np.nan
    # Get defaults for brightness, contrast
    if brightness is None:
        brightness = float(config.get('curvature', 'brightness'))
    if contrast is None:
        contrast = float(config.get('curvature', 'contrast'))
    # Scale and shift curvature image
    curv_im = (curv_im - 0.5) * contrast + brightness
    if extents is None:
        extents = _get_extents(fig)
    ax = fig.gca()
    cvimg = ax.imshow(curv_im, 
            aspect='equal', 
            extent=extents, 
            cmap=cmap, 
            vmin=0, 
            vmax=1, 
            label='curvature',
            zorder=0)
    return cvimg

def add_data(fig, braindata, height=1024, thick=32, depth=0.5, pixelwise=True, 
             sampler='nearest', recache=False):
    """Add data to quickflat plot

    Parameters
    ----------
    fig : figure
        Figure into which to plot image of curvature
    braindata : one of: {cortex.Volume, cortex.Vertex, cortex.Dataview)
        Object containing containing data to be plotted, subject (surface identifier), 
        and transform.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    extents : list
        Extents of image [left, right, top, bottom] in figure coordinates
    """    
    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.Dataview):
        # Unclear what this means. Clarify error in terms of pycortex classes 
        # (please provide a [cortex.dataset.Dataview or whatever] instance)
        raise TypeError('Please provide a Dataview, not a Dataset')
    # Generate image (2D array, maybe 3D array)
    im, extents = make_flatmap_image(dataview, recache=recache, pixelwise=pixelwise, sampler=sampler,
                       height=height, thick=thick, depth=depth)
    # Check whether dataview has a cmap instance
    cmapdict = _has_cmap(dataview)
    # Plot
    ax = fig.gca()
    img = ax.imshow(im, 
            aspect='equal', 
            extent=extents, 
            label='data',
            zorder=1,
            **cmapdict)
    return img, extents

def add_rois(fig, dataview, extents=None, height=None, with_labels=True, roi_list=None, **kwargs):
    """Add ROIs layer to a figure

    NOTE: zorder for rois is 3

    Parameters
    ----------
    fig : figure
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels on ROIs
    roi_list : 

    kwargs : 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    """
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)        
    svgobject = db.get_overlay(dataview.subject)
    svg_kws = _convert_svg_kwargs(kwargs)
    layer_kws = _parse_defaults('rois_paths')
    layer_kws.update(svg_kws)
    im = svgobject.get_texture('rois', height, labels=with_labels, shape_list=roi_list, **layer_kws)
    ax = fig.gca()
    img = ax.imshow(im,
        aspect='equal', 
        interpolation='bicubic', 
        extent=extents, 
        label='rois',
        zorder=4)
    return img

def add_sulci(fig, dataview, extents=None, height=None, with_labels=True, **kwargs):
    """Add sulci layer to figure

    Parameters
    ----------
    fig : figure
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels for sulci

    Other Parameters
    ----------------
    kwargs : keyword arguments
        Keywords args govern line appearance in final plot. Allowable kwargs are : linewidth,
        linecolor, 
    
    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    """
    svgobject = db.get_overlay(dataview.subject)
    svg_kws = _convert_svg_kwargs(kwargs)
    layer_kws = _parse_defaults('sulci_paths')
    layer_kws.update(svg_kws)
    sulc = svgobject.get_texture('sulci', height, labels=with_labels, **layer_kws)
    if extents is None:
        extents = _get_extents(fig)
    ax = fig.gca()
    img = ax.imshow(sulc,
                     aspect='equal', 
                     interpolation='bicubic', 
                     extent=extents, 
                     label='sulci',
                     zorder=5)
    return img

def add_hatch(fig, hatch_data, extents=None, height=None, hatch_space=4, hatch_color=(0,0,0),
    sampler='nearest', recache=False):
    """Add hatching to figure at locations specified in hatch_data    

    Parameters
    ----------
    fig : matplotlib figure
        Figure into which to plot the hatches. Should have pycortex flatmap image in it already.
    hatch_data : cortex.Volume
        cortex.Volume object created from data scaled from 0-1; locations with values of 1 will
        have hatching overlaid on them in the resulting image.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. If None, defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. if None, defaults to height of images already present in figure. 
    hatch_space : scalar 
        Spacing between hatch lines, in pixels
    hatch_color : 3-tuple
        (R, G, B) tuple for color of hatching. Values for R,G,B should be 0-1
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted hatch image

    Notes
    -----
    Possibly to add: add hatch_width, hatch_offset arguments.
    """
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)
    hatchim = _make_hatch_image(hatch_data, height, sampler, recache=recache, 
        hatch_space=hatch_space)
    hatchim[:,:,0] = hatch_color[0]
    hatchim[:,:,1] = hatch_color[1]
    hatchim[:,:,2] = hatch_color[2]

    ax = fig.gca()
    img = ax.imshow(hatchim, 
                    aspect="equal", 
                    interpolation="bicubic", 
                    extent=extents, 
                    label='hatch',
                    zorder=2)
    return img

def add_colorbar(fig, cimg, colorbar_ticks=None, colorbar_location=(.4, .07, .2, .04), 
    orientation='horizontal'):
    """Add a colorbar to a flatmap plot

    Parameters
    ----------
    fig : matplotlib Figure object
        Figure into which to insert colormap
    cimg : matplotlib.image.AxesImage object
        Image for which to create colorbar. For reference, matplotlib.image.AxesImage 
        is the output of imshow()
    colorbar_ticks : array-like
        values for colorbar ticks
    colorbar_location : array-like
        Four-long list, tuple, or array that specifies location for colorbar axes 
        [left, top, width, height] (?)
    orientation : string
        'vertical' or 'horizontal'
    """
    cbar = fig.add_axes(colorbar_location)
    fig.colorbar(cimg, cax=cbar, orientation=orientation, ticks=colorbar_ticks)
    return

def add_custom(fig, dataview, svgfile, layer, extents=None, height=None, with_labels=False, 
    shape_list=None, **kwargs):
    """Add a custom data layer
    
    Parameters
    ----------
    fig : matplotlib figure
        Figure into which to plot the hatches. Should have pycortex flatmap image in it already.
    dataview : cortex.Volume
        cortex.Volume object containing 
    svgfile : string
        Filepath for custom svg file to use. Must be formatted identically to overlays.svg
        file for subject in `dataview`
    layer : string
        Layer name within custom svg file to display
    extents : array-like
        4 values for [Left, Right, Bottom, Top] extents of image plotted. If None, defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. if None, defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels on ROIs
    shape_list : list
        list of paths/shapes within svg layer to render, if only a subset of the paths/
        shapes within the layer are desired.

    Other Parameters
    ----------------
    kwargs : 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data

    """
    if height is None:
        height = _get_height(fig)
    if extents is None:
        extents = _get_extents(fig)
    pts_, polys_ = db.get_surf(dataview.subject, "flat", merge=True, nudge=True)
    extra_svg = get_overlay(dataview.subject, svgfile, pts_, polys_)
    svg_kws = _convert_svg_kwargs(kwargs)
    try:
        # Check for layer if it exists
        layer_kws = _parse_defaults(layer+'_paths')
        layer_kws.update(svg_kws)
    except:
        layer_kws = svg_kws
    im = extra_svg.get_texture(layer, height, 
                               labels=with_labels, 
                               shape_list=shape_list, 
                               **layer_kws)
    ax = fig.gca()
    img = ax.imshow(im, 
                    aspect="equal", 
                    interpolation="nearest", 
                    extent=extents,  
                    label='custom',
                    zorder=6)
    return img

def add_connected_vertices(fig, dataview, height=None, extents=None, recache=False, 
                           min_dist=5, max_dist=85, color=(1.0,0.5,0.1,0.6), linewidth=2, 
                           **kwargs):
    """Plot lines btw distant vertices that are within the same voxel

    Notes
    -----
    Replace min_dist, max_dist with more principled values!
    extents is currently unused, but probably should be to scale pix_array
    """
    from matplotlib.collections import LineCollection
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)            
    subject = dataview.subject
    xfmname = dataview.xfmname
    if xfmname is None:
        # Raise error? Just pass?
        raise ValueError("Dataview for add_connected_vertices must be a Volume! You seem to have provided vertex data.")
    # Should cache this value in the db
    within_voxel_vertex_dists = get_shared_voxels(subject, xfmname, **kwargs)
    mapper = get_mapper(subject, xfmname, 'nearest', recache=recache)
    mask, extents = get_flatmask(subject)
    pixmap = get_flatcache(subject, None)
    img = np.nan * np.ones(mask.shape) #Finding a mapping from verts to pixels in flatmap (This is a hack)
    img[mask] = pixmap * (np.arange(mapper.nverts))
    img_arr = img.flatten()
    # I'm pretty sure this should be done with the pixmap / mask variables, but I'm not sure how rn so oh well
    # Two dictionaries, containing x and y values for each vert in the final flatmap. Again, this is a severe hack.
    # Currently, this is also the most time consuming part of the process.
    x_dict = {}
    y_dict = {}
    divisor = img.shape[1]
    for i, elem in enumerate(img_arr):
        if not np.isnan(elem):
            x_dict[int(elem)] = i//divisor
            y_dict[int(elem)] = i%divisor

    # Substitute:
    # flat, polys = db.get_surf(subject, "flat", merge=True, nudge=True)
    # valid = np.unique(polys)    
    pix_array, valid_vert_pairs = _return_pixel_pairs(within_voxel_vertex_dists[:,1:3], x_dict, y_dict)
    # Scaling this coordinates for plot
    pix_array_scaled = (pix_array / (np.array(mask.shape).astype(np.float32)))
    # Mebbe scale colors by distance? Or some such fancy? Not necessary...
    lc = LineCollection(pix_array_scaled, 
                        transform=fig.transFigure, 
                        figure=fig,
                        colors=color, 
                        linewidths=linewidth)
    ax = fig.gca()
    lc_object = ax.add_collection(lc)
    return lc_object

def add_cutout(fig, name, dataview, layers=None, height=None, extents=None):
    """Apply a cutout mask to extant layers in flatmap figure

    Parameters
    ----------
    fig : figure
        figure to which to add cutouts
    name : str
        name of cutout shape within cutouts layer to use to crop the rest of the figure
    dataview : 
    
    layers :
    
    height : int
    
    extents :

    Returns
    -------

    """
    if layers is None:
        layers = _get_images(fig)
    if height is None:
        height = _get_height(fig)
    if extents is None:
        extents  = _get_extents(fig)
    svgobject = db.get_overlay(dataview.subject)
    # Set other cutouts to be invisible
    for co_name, co_shape in svgobject.cutouts.shapes.items():
        co_shape.visible = co_name == name
    # Get cutout image (now all white = 1, black = 0)
    svg_kws = _convert_svg_kwargs(dict(fillcolor="white", 
                                       fillalpha=1.0,
                                       linecolor="white", 
                                       linewidth=2))
    co = svgobject.get_texture('cutouts', height, labels=False, **svg_kws)[..., 0]
    if not np.any(co):
        raise Exception('No pixels in cutout region {}!'.format(name))

    # Bounding box indices
    LL, RR, BB, TT = np.nan, np.nan, np.nan, np.nan
    # Clip each layer to this cutout
    for layer_name, im_layer in layers.items():
        im = im_layer.get_array()
        
        # Reconcile occasional 1-pixel difference between flatmap image layers 
        # that are generated by different functions
        if not all([np.abs(aa - bb) <= 1 for aa, bb in zip(im.shape, co.shape)]):
            raise Exception("Shape mismatch btw cutout and data!")
        if any([np.abs(aa - bb) > 0 and np.abs(aa - bb) < 2 for aa, bb in zip(im.shape, co.shape)]):
            from scipy.misc import imresize
            print('Resizing! {} to {}'.format(co.shape, im.shape[:2]))
            layer_cutout = imresize(co, im.shape[:2]).astype(np.float32)/255.
        else:
            layer_cutout = copy.copy(co)
        
        # Handle different types of alpha layers. Useful for RGBVolumes if nothing else.
        if im.dtype == np.uint8:
            im = np.cast['float32'](im)/255.
            im[:,:,3] *= layer_cutout
            h, w, cdim = [float(v) for v in im.shape]
        else:
            if np.ndim(im)==3:
                im[:,:,3] *= layer_cutout
                h, w, cdim = [float(v) for v in im.shape]
            elif np.ndim(im)==2:
                im[layer_cutout==0] = np.nan
                h, w = [float(v) for v in im.shape]
        y, x = np.nonzero(layer_cutout)
        l, r, b, t = extents
        x_span = np.abs(r-l)
        y_span = np.abs(t-b)
        extents_new = [l + x.min() / w * x_span,
                    l + x.max() / w * x_span,
                    t + y.min() / h * y_span,
                    t + y.max() / h * y_span]    

        # Bounding box indices
        iy, ix = ((y.min(), y.max()), (x.min(), x.max()))
        tmp = im[iy[0]:iy[1], ix[0]:ix[1]]
        im_layer.set_array(tmp)
        im_layer.set_extent(extents_new)
        
        # Track maxima / minima for figure
        LL = np.nanmin([extents_new[0], LL])
        RR = np.nanmax([extents_new[1], RR])
        BB = np.nanmin([extents_new[2], BB])
        TT = np.nanmax([extents_new[3], TT])
        imsize = (np.abs(np.diff(iy))[0], np.abs(np.diff(ix))[0])
    
    # Re-set figure limits
    ax = fig.gca()
    ax.set_xlim(LL, RR)
    ax.set_ylim(BB, TT)    
    inch_size = np.array(imsize)[::-1] / float(fig.dpi)
    fig.set_size_inches(inch_size[0], inch_size[1])

    return