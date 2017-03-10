import io
import os
import glob
import copy
import binascii
import numpy as np

from . import utils
from . import dataset
from .database import db
from .options import config
from .svgoverlay import get_overlay

### --- Individual compositing functions --- ###

# for all: 
"""    linewidth : int, optional
        Width of ROI lines. Defaults to roi options in your local `options.cfg`
    linecolor : tuple of float, optional
        (R, G, B, A) specification of line color
    roifill : tuple of float, optional # CHANGE TO FILLCOLOR - OR, leave as is, to be a unique kwarg...
        (R, G, B, A) specification for the fill of each ROI region
    shadow : int, optional
        Standard deviation of the gaussian shadow. Set to 0 if you want no shadow    
"""


def add_curvature(fig, dataview, extents=None, height=None, threshold=None, contrast=None,
                  brightness=None, cmap='gray', recache=False):
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
    threshold : boolean or None
        Whether to apply a threshold to the curvature values to create a binary curvature image
        (one shade for positive curvature, one shade for negative). `None` defaults to value 
        specified in the config file
    contrast : float, [0-1] or None 
        TBD: None defaults to config value
    brightness : float
        How bright to make average value of curvature. This is not raw brightness (for now); this 
        scales the minimum and maximum of the color scale for curvature, so the units are in curvature.
        Positive values will make the zero curvature value lighter and negative values will make the 
        zero curvatuer value darker. 
    cmap : string
        name for colormap of curvature
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. 
    """
    if height is None:
        height = _get_height(fig)
    # Get curvature map as image
    curv_vertices = db.get_surfinfo(dataview.subject)
    curv, _ = make_flatmap_image(curv_vertices, recache=recache, height=height)
    # Option to use thresholded curvature
    default_threshold = config.get('curvature','threshold').lower() in ('true','t','1','y','yes')
    use_threshold_curvature = default_threshold if threshold is None else threshold
    if use_threshold_curvature:
        curvT = (curv>0).astype(np.float32)
        curvT[np.isnan(curv)] = np.nan
        curv = curvT
    # Still WIP: Compute min / max for display of curvature based on `contrast` and `brightness` inputs
    if contrast is None:
        contrast = float(config.get('curvature', 'contrast'))
    if brightness is None:
        brightness = float(config.get('curvature', 'brightness'))
    cvmin, cvmax = -contrast, contrast
    if brightness < 0:
        cvmin += brightness
    else:
        cvmax += brightness
    if extents is None:
        extents = _get_extents(fig)
    ax = fig.gca()
    cvimg = ax.imshow(curv, 
            aspect='equal', 
            extent=extents, 
            cmap=cmap, 
            vmin=cvmin, #float(config.get('curvature','min')) if cvmin is None else cvmin,
            vmax=cvmax, #float(config.get('curvature','max')) if cvmax is None else cvmax,
            origin='lower',
            label='curvature',
            zorder=0)
    return cvimg

def add_data(fig, braindata, height=1024, thick=32, depth=0.5, pixelwise=True, 
             sampler='nearest', recache=False):
    """Add data to quickflat plot

    Parameters
    ----------
    fig : figure
        figure into which to plot image of curvature
    braindata : one of: {cortex.Volume, cortex.Vertex, cortex.Dataview)
        object containing containing data to be plotted, subject (surface identifier), 
        and transform.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    recache : bool
        If True, recache the flatmap cache. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options

    Returns
    -------
    img : 

    extents : 
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
            origin='lower',
            label='data',
            zorder=1,
            **cmapdict)
    return img, extents

def add_rois(fig, dataview, extents=None, height=None, with_labels=True, roi_list=None, **kwargs):
    """

    NOTE: zorder for rois is 3

    Parameters
    ----------
    fig : 

    dataview : 

    extents : 

    height : 

    with_labels : 

    roi_list : 

    kwargs : 


    """
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)        
    svgobject = db.get_overlay(dataview.subject)
    svg_kws = _convert_svg_kwargs(kwargs)
    im = svgobject.get_texture('rois', height, labels=with_labels, shape_list=roi_list, **svg_kws)
    ax = fig.gca()
    img = ax.imshow(im,
        aspect='equal', 
        interpolation='bicubic', 
        extent=extents, 
        origin='lower',
        label='rois',
        zorder=4)
    return img

def add_sulci(fig, dataview, extents=None, height=1024, with_labels=True, **kwargs):
    """Add sulci layer to figure

    Parameters
    ----------
    linewidth : 

    linecolor : 

    with_labels : 

    labelsize : 

    labelcolor : 

    shadow : 

    kwargs : 

    """
    svgobject = db.get_overlay(dataview.subject)
    svg_kws = _convert_svg_kwargs(kwargs)
    sulc = svgobject.get_texture('sulci', height, labels=with_labels, **svg_kws)
    if extents is None:
        extents = _get_extents(fig)
    ax = fig.gca()
    img = ax.imshow(sulc,
                     aspect='equal', 
                     interpolation='bicubic', 
                     extent=extents, 
                     origin='lower',
                     label='sulci',
                     zorder=5)
    return img

def add_hatch(fig, hatch_data, extents=None, height=None, hatch_space=4, hatch_color=(0,0,0),
    sampler='nearest', recache=False):
    """Add hatching to figure at locations specified in hatch_data

    TODO: add hatch_width, hatch_offset arguments.

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
                    origin='lower',
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
    """
    Parameters
    ----------
    fig

    dataview : 

    svgfile : 

    layer : 

    labelsize : 

    labelcolor : 

    """
    if height is None:
        height = _get_height(fig)
    if extents is None:
        extents = _get_extents(fig)
    pts_, polys_ = db.get_surf(dataview.subject, "flat", merge=True, nudge=True)
    extra_svg = get_overlay(svgfile, pts_, polys_)
    svg_kws = _convert_svg_kwargs(kwargs)
    im = extra_svg.get_texture(layer, height, 
                               labels=with_labels, 
                               shape_list=shape_list, 
                               **svg_kws)
    ax = fig.gca()
    img = ax.imshow(im, 
                    aspect="equal", 
                    interpolation="nearest", 
                    extent=extents, 
                    origin='lower', 
                    label='custom',
                    zorder=6)
    return img

def add_cutout(fig, name, dataview, layers=None, height=None, extents=None):
    """Apply a cutout mask to extant layers in flatmap figure

    Parameters
    ----------

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
    # Set all cutouts to be filled w/ white
    #svgobject.cutouts.set(fill="white", stroke="white", **{'stroke-width':'2'})
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
        raise Exception('No pixels in cutout region {}}!'.format(name))
    # print('orig_extents: {}'.format(extents))
    # l, r, t, b = extents
    # x_span = np.abs(l-r)
    # y_span = np.abs(t-b)
    # y, x = np.nonzero(co)
    # extents_new = [l + x.min() / w * x_span,
    #             l + x.max() / w * x_span,
    #             t + y.min() / h * y_span,
    #             t + y.max() / h * y_span]    

    # # Set extents        
    # print('tmp extents: {}'.format(extents_new))
    # # Bounding box indices

    LL, RR, TT, BB = np.nan, np.nan, np.nan, np.nan
    # Clip each layer to this cutout
    for layer_name, im_layer in layers.items():
        #print('\n=== Clipping %s... ==='%layer_name)
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
        # Handle different types of alpha layers. Unclear if this is still necessary after 
        # switching api to deal with matplotlib images.
        if im.dtype == np.uint8:
            raise Exception("WTF are you doing with uint8 data in a matplotlib Image...")
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
        #print('orig_extents: {}'.format(extents))
        #print('Fak cutout extents are: {}'.format((x.min(), x.max(), y.min(), y.max())))
        #print('height, width: {}'.format((h,w)))
        #print('Cutout shape: {} - {}'.format(co.shape, layer_cutout.shape))
        l, r, t, b = extents
        x_span = np.abs(l-r)
        y_span = np.abs(t-b)
        extents_new = [l + x.min() / w * x_span,
                    l + x.max() / w * x_span,
                    t + y.min() / h * y_span,
                    t + y.max() / h * y_span]    

        # Set extents        
        #print('tmp extents: {}'.format(extents_new))
        # Bounding box indices
        iy, ix = ((y.min(), y.max()), (x.min(), x.max()))
        tmp = im[iy[0]:iy[1], ix[0]:ix[1]]
        im_layer.set_array(tmp)
        im_layer.set_extent(extents_new)
        # Track maxima / minima for figure
        LL = np.nanmin([extents_new[0], LL])
        RR = np.nanmax([extents_new[1], RR])
        BB = np.nanmax([extents_new[2], BB])
        TT = np.nanmin([extents_new[3], TT])
        #print('new extents: {}'.format((LL, RR, BB, TT)))
        imsize = (np.abs(np.diff(iy))[0], np.abs(np.diff(ix))[0])#fig.get_axes()[0].get_images()[0].get_size()
        #print('image size for this cutout: {}'.format(imsize))
    # Re-set figure limits
    ax = fig.gca()
    #print('setting limits to: ')
    #print(LL, RR)
    #print(TT, BB)
    ax.set_xlim(LL, RR)
    ax.set_ylim(TT, BB)
    #
    #if fig_resize:
    #imsize = fig.get_axes()[0].get_images()[0].get_size()
    
    dpi = 100
    inch_size = np.array(imsize)[::-1] / float(dpi)
    #print('Size in inches: ', inch_size)
    fig.set_size_inches(inch_size[0], inch_size[1])
    return #[LL, RR, BB, TT]
        

### --- Main functions --- ###

def make_figure(braindata, recache=False, pixelwise=True, thick=32, sampler='nearest',
                height=1024, dpi=100, depth=0.5, with_rois=True, with_sulci=False,
                with_labels=True, with_colorbar=True, with_borders=False, 
                with_dropout=False, with_curvature=False, extra_disp=None, 
                linewidth=None, linecolor=None, roifill=None, shadow=None,
                labelsize=None, labelcolor=None, cutout=None, cvmin=None,
                cvmax=None, cvthr=None, fig=None, extra_hatch=None,
                colorbar_ticks=None, colorbar_location=(.4, .07, .2, .04), **kwargs):
    """Show a Volume or Vertex on a flatmap with matplotlib. Additional kwargs are passed on to
    matplotlib's imshow command.
    Parameters
    ----------
    braindata : Dataview
        the data you would like to plot on a flatmap
    recache : bool
        If True, recache the flatmap cache. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options
    height : int
        Height of the image to render. Automatically scales the width for the aspect
        of the subject's flatmap
    depth : float
        Value between 0 and 1 for how deep to sample the surface for the flatmap (0 = gray/white matter
        boundary, 1 = pial surface)
    with_rois, with_labels, with_colorbar, with_borders, with_dropout, with_curvature : bool, optional
        Display the rois, labels, colorbar, annotated flatmap borders, or cross-hatch dropout?
    cutout : str
        Name of flatmap cutout with which to clip the full flatmap. Should be the name
        of a sub-layer of the 'cutouts' layer in <filestore>/<subject>/rois.svg
    Other Parameters
    ----------------
    dpi : int
        DPI of the generated image. Only applies to the scaling of matplotlib elements,
        specifically the colormap
    linewidth : int, optional
        Width of ROI lines. Defaults to roi options in your local `options.cfg`
    linecolor : tuple of float, optional
        (R, G, B, A) specification of line color
    roifill : tuple of float, optional
        (R, G, B, A) sepcification for the fill of each ROI region
    shadow : int, optional
        Standard deviation of the gaussian shadow. Set to 0 if you want no shadow
    labelsize : str, optional
        Font size for the label, e.g. "16pt"
    labelcolor : tuple of float, optional
        (R, G, B, A) specification for the label color
    cvmin : float,optional
        Minimum value for curvature colormap. Defaults to config file value.
    cvmax : float, optional
        Maximum value for background curvature colormap. Defaults to config file value.
    cvthr : bool,optional
        Apply threshold to background curvature
    extra_disp : tuple, optional
        Optional extra display layer from external .svg file. Tuple specifies (filename,layer)
        filename should be a full path. External svg file should be structured exactly as 
        rois.svg for the subject. (Best to just copy rois.svg somewhere else and add layers to it)
        Default value is None.
    extra_hatch : tuple, optional
        Optional extra crosshatch-textured layer, given as (DataView, [r, g, b]) tuple. 
    colorbar_location : tuple, optional
        Location of the colorbar! Not sure of what the numbers actually mean. Left, bottom, width, height, maybe?
    """
    from matplotlib import colors,cm, pyplot as plt
    from matplotlib.collections import LineCollection

    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.Dataview):
        # Unclear what this means. Clarify error.
        raise TypeError('Please provide a Dataview, not a Dataset')
    
    if fig is None:
        fig_resize = True
        fig = plt.figure()
    else:
        fig_resize = False
        fig = plt.figure(fig.number)
    ax = fig.add_axes((0, 0, 1, 1))
    # Add data
    data_im, extents = add_data(fig, dataview, pixelwise=pixelwise, thick=thick, sampler=sampler, 
                       height=height, depth=depth, recache=recache)

    layers = dict(data=data_im)
    # Add curvature
    if with_curvature:
        curv_im = add_curvature(fig, dataview, extents)
        layers['curvature'] = curv_im
    # Add dropout
    if with_dropout is not False:
        # Support old api:
        if isinstance(with_dropout, dataset.Dataview):
            hatch_data = with_dropout
        else:
            hatch_data = None
            dropout_power = 20 if with_dropout is True else with_dropout
        if hatch_data is None:
            hatch_data = utils.get_dropout(dataview.subject, dataview.xfmname,
                                         power=dropout_power)

        drop_im = add_hatch(fig, hatch_data, extents=extents, height=height, 
            sampler=sampler)
        layers['dropout'] = drop_im
    # Add extra hatching
    if extra_hatch is not None:
        hatch_data2, hatch_color = extra_hatch
        hatch_im = add_hatch(fig, hatch_data2, extents=extents, height=height, 
            sampler=sampler)
        layers['hatch'] = hatch_im
    # Add rois
    if with_rois:
        roi_im = add_rois(fig, dataview, extents=extents, height=height, linewidth=linewidth, linecolor=linecolor,
             roifill=roifill, shadow=shadow, labelsize=labelsize, labelcolor=labelcolor, with_labels=with_labels)
        layers['rois'] = roi_im
    # Add sulci
    if with_sulci:
        sulc_im = add_sulci(fig, dataview, extents=extents, height=height, linewidth=linewidth, linecolor=linecolor,
             shadow=shadow, labelsize=labelsize, labelcolor=labelcolor, with_labels=with_labels)
        layers['sulci'] = sulc_im
    # Add custom
    if extra_disp is not None:
        svgfile, layer = extra_disp
        custom_im = add_custom(fig, dataview.subject, svgfile, layer, height=height, extents=extents, 
            linewidth=linewidth, linecolor=linecolor, shadow=shadow, labelsize=labelsize, labelcolor=labelcolor, 
            with_labels=with_labels)
        layers['custom'] = custom_im
        
    ax.axis('off')
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[3], extents[2])

    if fig_resize:
        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))

    # Add (apply) cutout of flatmap
    if cutout is not None:
        extents = add_cutout(fig, cutout, dataview, layers)

    return fig


def make_figure_old(braindata, recache=False, pixelwise=True, thick=32, sampler='nearest',
                height=1024, dpi=100, depth=0.5, with_rois=True, with_sulci=False,
                with_labels=True, with_colorbar=True, with_borders=False, 
                with_dropout=False, with_curvature=False, extra_disp=None, 
                linewidth=None, linecolor=None, roifill=None, shadow=None,
                labelsize=None, labelcolor=None, cutout=None, cvmin=None,
                cvmax=None, cvthr=None, fig=None, extra_hatch=None,
                colorbar_ticks=None, colorbar_location=(.4, .07, .2, .04), **kwargs):
   
    from matplotlib import colors,cm, pyplot as plt
    from matplotlib.collections import LineCollection

    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.Dataview):
        # Unclear what this means. Clarify error.
        raise TypeError('Please provide a Dataview, not a Dataset')
    
    if fig is None:
        fig_resize = True
        fig = plt.figure()
    else:
        fig_resize = False
        fig = plt.figure(fig.number)

    im, extents = make_flatmap_image(dataview, recache=recache, pixelwise=pixelwise, sampler=sampler,
                       height=height, thick=thick, depth=depth)

    svgobject = db.get_overlay(dataview.subject)

    if cutout:
        # Set ONLY desired cutout to be white
        svgobject.cutouts.set(fill="white",stroke="white", **{'stroke-width':'2'})
        for co_name, co_shape in svgobject.cutouts.shapes.items():
            sh.visible = co_name == cutout
        #roitex = svgobject.get_texture(height, labels=False)
        #roitex.seek(0)
        co = svgobject.get_texture('cutouts', height, labels=False)
        if not np.any(co):
            raise Exception('No pixels in cutout region %s!'%cutout)

        # STUPID BUT NECESSARY 1-PIXEL CHECK:
        if any([np.abs(aa-bb)>0 and np.abs(aa-bb)<2 for aa,bb in zip(im.shape,co.shape)]):
            from scipy.misc import imresize
            co = imresize(co, im.shape[:2]).astype(np.float32)/255.

        # Alpha
        if im.dtype == np.uint8:
            im = np.cast['float32'](im)/255.
            im[:,:,3]*=co
            h, w, cdim = [float(v) for v in im.shape]
        else:
            im[co==0] = np.nan
            h, w = [float(v) for v in im.shape]

        # set extents
        y,x = np.nonzero(co)
        l,r,t,b = extents
        extents = [x.min()/w * (l-r)+l,
                    x.max()/w * (l-r)+l,
                    y.min()/h * (t-b)+b,
                    y.max()/h * (t-b)+b]

        # bounding box indices
        iy,ix = ((y.min(),y.max()),(x.min(),x.max()))
    else:
        iy,ix = ((0,-1),(0,-1))
    
    if with_curvature:
        curv,ee = make_flatmap_image(db.get_surfinfo(dataview.subject), recache=recache, height=height)
        if cutout: curv[co==0] = np.nan
        axcv = fig.add_axes((0,0,1,1))
        # Option to use thresholded curvature
        use_threshold_curvature = config.get('curvature','threshold').lower() in ('true','t','1','y','yes') if cvthr is None else cvthr
        if use_threshold_curvature:
            curvT = (curv>0).astype(np.float32)
            curvT[np.isnan(curv)] = np.nan
            curv = curvT
        cvimg = axcv.imshow(curv[iy[1]:iy[0]:-1,ix[0]:ix[1]], 
                aspect='equal', 
                extent=extents, 
                cmap=plt.cm.gray,
                vmin=float(config.get('curvature','min')) if cvmin is None else cvmin,
                vmax=float(config.get('curvature','max')) if cvmax is None else cvmax,
                origin='lower')
        axcv.axis('off')
        axcv.set_xlim(extents[0], extents[1])
        axcv.set_ylim(extents[2], extents[3])

    imkws = dict(aspect='equal', 
        extent=extents, 
        origin='lower')
    
    # Check whether dataview has a cmap instance
    cmapdict = _has_cmap(dataview)
    imkws.update(cmapdict)
    print(imkws)

    ax = fig.add_axes((0,0,1,1))
    cimg = ax.imshow(im[iy[1]:iy[0]:-1,ix[0]:ix[1]], **imkws)
    ax.axis('off')
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])


    if with_colorbar and not isinstance(dataview, dataset.Volume2D):
        cbar = fig.add_axes(colorbar_location)
        fig.colorbar(cimg, cax=cbar, orientation='horizontal',
                     ticks=colorbar_ticks)

    if with_dropout is not False:
        if isinstance(with_dropout, dataset.Dataview):
            dropout_data = with_dropout
        else:
            if with_dropout is True:
                dropout_power = 20 # default
            else:
                dropout_power = with_dropout

            dropout_data = utils.get_dropout(dataview.subject, dataview.xfmname,
                                             power=dropout_power)
        
        hatchim = _make_hatch_image(dropout_data, height, sampler, recache=recache)
        if cutout: hatchim[:,:,3]*=co
        dax = fig.add_axes((0,0,1,1))
        dax.imshow(hatchim[iy[1]:iy[0]:-1,ix[0]:ix[1]], aspect="equal",
                   interpolation="nearest", extent=extents, origin='lower')

    if extra_hatch is not None:
        hatch_data, hatch_color = extra_hatch
        hatchim = _make_hatch_image(hatch_data, height, sampler, recache=recache)
        hatchim[:,:,0] = hatch_color[0]
        hatchim[:,:,1] = hatch_color[1]
        hatchim[:,:,2] = hatch_color[2]
        if cutout: hatchim[:,:,3]*=co
        dax = fig.add_axes((0,0,1,1))
        dax.imshow(hatchim[iy[1]:iy[0]:-1,ix[0]:ix[1]], aspect="equal",
                   interpolation="nearest", extent=extents, origin='lower')
    
    if with_borders:
        border = _gen_flat_border(dataview.subject, im.shape[0])
        bax = fig.add_axes((0,0,1,1))
        blc = LineCollection(border[0], linewidths=3.0,
                             colors=[['r','b'][mw] for mw in border[1]])
        bax.add_collection(blc)
    #O = db.get_overlay(dataview.subject)
    overlays = []
    if with_rois:
        #co = svgobject.get_texture(height, 'cutouts', labels=False)
        roi = svgobject.get_texture('rois', height, labels=with_labels, #**kwargs)
                                    linewidth=linewidth,
                                    linecolor=linecolor,
                                    roifill=roifill,
                                    shadow=shadow,
                                    labelsize=labelsize,
                                    labelcolor=labelcolor)
        overlays.append(roi)
    if with_sulci:
        sulc = svgobject.get_texture('sulci', height, labels=with_labels, #**kwargs)
                                     linewidth=linewidth,
                                     linecolor=linecolor,
                                     shadow=shadow,
                                     labelsize=labelsize,
                                     labelcolor=labelcolor)
        overlays.append(sulc)

    if not extra_disp is None:
        raise NotImplementedError("Not yet!")
        svgfile,layer = extra_disp
        if not isinstance(layer,(list,tuple)):
            layer = [layer]
        for extralayer in layer:
            # Allow multiple extra layer overlays
            pts_, polys_ = (0,0)
            O = svgoverlay.get_overlay(svgfile, pts_, polys_)
            disp = None
            # disp = svgoverlay.get_overlay(dataview.subject,
            #                   otype='external',
            #                   shadow=shadow,
            #                   labelsize=labelsize,
            #                   labelcolor=labelcolor,
            #                   layer=extralayer,
            #                   svgfile=svgfile)
            overlays.append(disp)

    for oo in overlays:
        #roitex = oo.get_texture(height, labels=with_labels, size=labelsize)
        oo.seek(0)
        oax = fig.add_axes((0,0,1,1))
        im = plt.imread(oo)
        if cutout: 
            # STUPID BUT NECESSARY 1-PIXEL CHECK:
            if any([np.abs(aa-bb)>0 and np.abs(aa-bb)<2 for aa,bb in zip(im.shape,im.shape)]):
                from scipy.misc import imresize
                co = imresize(co,im.shape[:2]).astype(np.float32)/255.
            im[:,:,3] *= co

        oimg = oax.imshow(im[iy[1]:iy[0]:-1,ix[0]:ix[1]],
            aspect='equal', 
            interpolation='bicubic', 
            extent=extents, 
            zorder=3,
            origin='lower')

    if fig_resize:
        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        
    return fig

def make_png(fname, braindata, recache=False, pixelwise=True, sampler='nearest', height=1024,
             bgcolor=None, dpi=100, **kwargs):
    """Create a PNG of the VertexData or VolumeData on a flatmap.

    Parameters
    ----------
    fname : str
        Filename for where to save the PNG file
    braindata : Dataview
        the data you would like to plot on a flatmap
    recache : bool
        If True, recache the flatmap cache. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data
    height : int
        Height of the image to render. Automatically scales the width for the aspect of
        the subject's flatmap
    depth : float
        Value between 0 and 1 for how deep to sample the surface for the flatmap (0 = gray/white matter
        boundary, 1 = pial surface)        
    with_rois, with_labels, with_colorbar, with_borders, with_dropout : bool, optional
        Display the rois, labels, colorbar, annotated flatmap borders, and cross-hatch dropout?
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options

    Other Parameters
    ----------------
    dpi : int
        DPI of the generated image. Only applies to the scaling of matplotlib elements,
        specifically the colormap
    bgcolor : matplotlib colorspec
        Color of background of image. `None` gives transparent background.
    linewidth : int, optional
        Width of ROI lines. Defaults to roi options in your local `options.cfg`
    linecolor : tuple of float, optional
        (R, G, B, A) specification of line color
    roifill : tuple of float, optional
        (R, G, B, A) sepcification for the fill of each ROI region
    shadow : int, optional
        Standard deviation of the gaussian shadow. Set to 0 if you want no shadow
    labelsize : str, optional
        Font size for the label, e.g. "16pt"
    labelcolor : tuple of float, optional
        (R, G, B, A) specification for the label color
    """
    from matplotlib import pyplot as plt
    fig = make_figure(braindata,
                      recache=recache,
                      pixelwise=pixelwise,
                      sampler=sampler,
                      height=height,
                      **kwargs)
    
    imsize = fig.get_axes()[0].get_images()[0].get_size()
    fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
    if bgcolor is None:
        fig.savefig(fname, transparent=True, dpi=dpi)
    else:
        fig.savefig(fname, facecolor=bgcolor, transparent=False, dpi=dpi)
    fig.clf()
    plt.close(fig)

def make_svg(fname, braindata, with_labels=True, **kwargs): # recache=False, pixelwise=True, sampler='nearest', height=1024, thick=32, depth=0.5, 
    """Save an svg file of the desired flatmap.

    This function creates an SVG file with vector graphic ROIs overlaid on a single png image.
    Ideally, this function would layer different images (curvature, data, dropout, etc), but 
    that has been left to implement at a future date if anyone really wants it. 

    Parameters
    ----------
    fname : string
        file name to save
    braindata : Dataview
        the data you would like to plot on a flatmap
    with_labels : bool
        Whether to display text labels on ROIs

    Other Parameters
    ----------------
    kwargs : see make_figure
        All kwargs are passed to make_png. `with_rois` will be ignored, because by using 
        this function you are basically saying that you want an editable layer of vector 
        graphic ROIs on top of your image. `with_cutouts` is not functional yet.
    """
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()
    from matplotlib.pylab import imsave
    to_cut = ['with_rois','cutouts']
    for cc in to_cut:
        if cc in kwargs: 
            _ = kwargs.pop(cc)
    ## Render PNG file & retrieve image data
    make_png(fp,braindata,with_rois=False,**kwargs) #recache=recache, pixelwise=pixelwise, sampler=sampler, height=height, thick=thick, depth=depth, **kwargs)
    fp.seek(0)
    pngdata = binascii.b2a_base64(fp.read())
    ## Create and save SVG file
    roipack = utils.get_roipack(braindata.subject)
    roipack.get_svg(fname, labels=with_labels, with_ims=[pngdata])

def show(*args, **kwargs):
    """Wrapper for make_figure()"""
    return make_figure(*args, **kwargs)

def make_movie(name, data, subject, xfmname, recache=False, height=1024,
               sampler='nearest', dpi=100, tr=2, interp='linear', fps=30,
               vcodec='libtheora', bitrate="8000k", vmin=None, vmax=None, **kwargs):
    """Create a movie of an 4D data set"""
    raise NotImplementedError
    import sys
    import shlex
    import shutil
    import tempfile
    import subprocess as sp
    import multiprocessing as mp
    
    from scipy.interpolate import interp1d

    # Make the flatmaps
    ims, extents = make_flatmap_image(data, subject, xfmname, recache=recache, height=height, sampler=sampler)
    if vmin is None:
        vmin = np.nanmin(ims)
    if vmax is None:
        vmax = np.nanmax(ims)

    # Create the matplotlib figure
    fig = make_figure(ims[0], subject, vmin=vmin, vmax=vmax, **kwargs)
    fig.set_size_inches(np.array([ims.shape[2], ims.shape[1]]) / float(dpi))
    img = fig.axes[0].images[0]

    # Set up interpolation
    times = np.arange(0, len(ims)*tr, tr)
    interp = interp1d(times, ims, kind=interp, axis=0, copy=False)
    frames = np.linspace(0, times[-1], (len(times)-1)*tr*fps+1)
    
    try:
        path = tempfile.mkdtemp()
        impath = os.path.join(path, "im{:09d}.png")
        for frame, frame_time in enumerate(frames): 
            img.set_data(interp(frame_time))
            fig.savefig(impath.format(frame), transparent=True, dpi=dpi)
        # avconv might not be relevant function for all operating systems. 
        # Introduce operating system check here?
        cmd = "avconv -i {path} -vcodec {vcodec} -r {fps} -b {br} {name}".format(path=impath, vcodec=vcodec, fps=fps, br=bitrate, name=name)
        sp.call(shlex.split(cmd))
    finally:
        shutil.rmtree(path)

### --- Helper functions --- ###

def make_flatmap_image(braindata, height=1024, recache=False, **kwargs):
    """Generate flatmap image from volumetric brain data

    This 

    Parameters
    ----------
    braindata : 

    height : 

    recache : 

    kwargs : wtf

    Returns
    -------
    image : 

    extents :

    """
    mask, extents = get_flatmask(braindata.subject, height=height, recache=recache)
    
    if not hasattr(braindata, "xfmname"):
        pixmap = get_flatcache(braindata.subject,
                               None,
                               height=height,
                               recache=recache,
                               **kwargs)
        
        if isinstance(braindata, dataset.Vertex2D):
            data = braindata.raw.vertices
        else:
            data = braindata.vertices
    else:
        pixmap = get_flatcache(braindata.subject,
                               braindata.xfmname,
                               height=height,
                               recache=recache,
                               **kwargs)
        if isinstance(braindata, dataset.Volume2D):
            data = braindata.raw.volume
        else:
            data = braindata.volume

    if data.shape[0] > 1:
        raise ValueError("Cannot flatten movie views - please provide 3D Volume or 2D Vertex data")

    if data.dtype == np.uint8:
        img = np.zeros(mask.shape+(4,), dtype=np.uint8)
        img[mask] = pixmap * data.reshape(-1, 4)
        return img.transpose(1,0,2)[::-1], extents
    else:
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan*np.ones(mask.shape)).astype(braindata.data.dtype)
        mimg = (np.nan*np.ones(badmask.shape)).astype(braindata.data.dtype)
        mimg[badmask] = (pixmap*data.ravel())[badmask].astype(mimg.dtype)
        img[mask] = mimg

        return img.T[::-1], extents

def get_flatmask(subject, height=1024, recache=False):
    """FARK"""
    cachedir = db.get_cache(subject)
    cachefile = os.path.join(cachedir, "flatmask_{h}.npz".format(h=height))

    if not os.path.exists(cachefile) or recache:
        mask, extents = _make_flatmask(subject, height=height)
        np.savez(cachefile, mask=mask, extents=extents)
    else:
        npz = np.load(cachefile)
        mask, extents = npz['mask'], npz['extents']
        npz.close()

    return mask, extents

def get_flatcache(subject, xfmname, pixelwise=True, thick=32, sampler='nearest',
                  recache=False, height=1024, depth=0.5):
    """DEDARK"""
    cachedir = db.get_cache(subject)
    cachefile = os.path.join(cachedir, "flatverts_{height}.npz").format(height=height)
    if pixelwise and xfmname is not None:
        cachefile = os.path.join(cachedir, "flatpixel_{xfmname}_{height}_{sampler}_{extra}.npz")
        extra = "l%d"%thick if thick > 1 else "d%g"%depth
        cachefile = cachefile.format(height=height, xfmname=xfmname, sampler=sampler, extra=extra)

    if not os.path.exists(cachefile) or recache:
        print("Generating a flatmap cache")
        if pixelwise and xfmname is not None:
            pixmap = _make_pixel_cache(subject, xfmname, height=height, sampler=sampler, thick=thick, depth=depth)
        else:
            pixmap = _make_vertex_cache(subject, height=height)
        np.savez(cachefile, data=pixmap.data, indices=pixmap.indices, indptr=pixmap.indptr, shape=pixmap.shape)
    else:
        from scipy import sparse
        npz = np.load(cachefile)
        pixmap = sparse.csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
        npz.close()

    if not pixelwise and xfmname is not None:
        from scipy import sparse
        mapper = utils.get_mapper(subject, xfmname, sampler)
        pixmap = pixmap * sparse.vstack(mapper.masks)

    return pixmap


### --- Hidden helper functions --- ###

def _color2hex(color):
    """Convert arbitrary color input to hex string"""
    from matplotlib import colors
    cc = colors.ColorConverter()
    rgba = cc.to_rgba(color)
    hexcol = colors.rgb2hex(rgba)
    return hexcol
    
def _convert_svg_kwargs(kwargs):
    """Convert matplotlib-like plotting property names/values to svg object property names/values"""
    svg_style_key_mapping = dict(
        linewidth='stroke-width',
        lw='stroke-width',
        linecolor='stroke',
        lc='stroke',
        #labelcolor='',
        #labelsize='',
        linealpha='stroke-opacity',
        fillcolor='fill',
        roifill='fill',
        fillalpha='fill-opacity',
        dashes='stroke-dasharray'
        #dash_capstyle
        #dash_joinstyle
        )  
    svg_style_value_mapping = dict(
        linewidth=lambda x: x,
        lw=lambda x: x,
        linecolor=lambda x: _color2hex(x), 
        lc=lambda x: _color2hex(x), 
        labelcolor=lambda x: _color2hex(x), 
        linealpha=lambda x: x,
        roifill=lambda x: _color2hex(x),
        fillcolor=lambda x: _color2hex(x),
        fillalpha=lambda x: x,
        dashes=lambda x: '{}, {}'.format(*x),
        )

    out = dict((svg_style_key_mapping[k], svg_style_value_mapping[k](v)) 
               for k,v in kwargs.items() if v is not None)
    return out

def _get_images(fig):
    """Get all images in a given matplotlib axis"""
    from matplotlib.image import AxesImage
    ax = fig.gca()
    images = dict((x.get_label(), x) for x in ax.get_children() if isinstance(x, AxesImage))
    return images

def _get_extents(fig):
    """Get extents of images current in a given matplotlib figure"""
    images = _get_images(fig)
    if 'data' not in images:
        raise ValueError("You must specify `extents` argument if you have not yet plotted a data flatmap!")
    extents = images['data'].get_extent()
    return extents

def _get_height(fig):
    """Get height of images in currently in a given matplotlib figure"""
    images = _get_images(fig)
    if 'data_cutout' in images:
        raise Exception("Can't add plots once cutout has been performed! Do cutouts last!")
    if 'data' in images:
        height = images['data'].get_array().shape[0]
    else:
        # No images, revert to default
        height = 1024 
    return height

def _make_hatch_image(hatch_data, height, sampler='nearest', hatch_space=4, recache=False):
    """Make hatch image

    Parameters
    ----------
    hatch_data : cortex.Dataview
        brain data with values ranging from 0-1, specifying where to show hatch marks (data value
        will be mapped to alpha value of hatch marks)
    height : scalar
        height of image to display
    sampler : string
        pycortex sampler string, {'nearest', ...} (FILL ME IN ??)
    hatch_space : scalar
        space between hatch lines (in pixels)
    recache : boolean


    """
    dmap, _ = make_flatmap_image(hatch_data, height=height, sampler=sampler, recache=recache)
    hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    
    hatchpat = (hx+hy)%(2*hatch_space) < 2
    # Leila code that breaks shit:
    #hatch_size = [0, 4, 4]
    #hatchpat = (hx + hy + hatch_size[0])%(hatch_size[1] * hatch_space) < hatch_size[2]

    hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
    hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
    hatchim[:, : ,3] *= np.clip(dmap, 0, 1).astype(float)

    return hatchim

def _make_flatmask(subject, height=1024):
    from . import polyutils
    from PIL import Image, ImageDraw
    pts, polys = db.get_surf(subject, "flat", merge=True, nudge=True)
    bounds = polyutils.trace_poly(polyutils.boundary_edges(polys))
    left, right = bounds.next(), bounds.next()
    aspect = (height / (pts.max(0) - pts.min(0))[1])
    lpts = (pts[left] - pts.min(0)) * aspect
    rpts = (pts[right] - pts.min(0)) * aspect

    im = Image.new('L', (int(aspect * (pts.max(0) - pts.min(0))[0]), height))
    draw = ImageDraw.Draw(im)
    draw.polygon(lpts[:,:2].ravel().tolist(), fill=255)
    draw.polygon(rpts[:,:2].ravel().tolist(), fill=255)
    extents = np.hstack([pts.min(0), pts.max(0)])[[0,3,1,4]]

    return np.array(im).T > 0, extents

def _make_vertex_cache(subject, height=1024):
    from scipy import sparse
    from scipy.spatial import cKDTree
    flat, polys = db.get_surf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)
    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)

    mask, extents = get_flatmask(subject, height=height)
    assert mask.shape[0] == width and mask.shape[1] == height

    kdt = cKDTree(flat[valid,:2])
    dist, vert = kdt.query(grid.T[mask.ravel()])
    dataij = (np.ones((len(vert),)), np.array([np.arange(len(vert)), valid[vert]]))
    return sparse.csr_matrix(dataij, shape=(mask.sum(), len(flat)))

def _make_pixel_cache(subject, xfmname, height=1024, thick=32, depth=0.5, sampler='nearest'):
    from scipy import sparse
    from scipy.spatial import Delaunay
    flat, polys = db.get_surf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)
    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)
    
    mask, extents = get_flatmask(subject, height=height)
    assert mask.shape[0] == width and mask.shape[1] == height
    
    ## Get barycentric coordinates
    dl = Delaunay(flat[valid,:2])
    simps = dl.find_simplex(grid.T[mask.ravel()])
    missing = simps == -1
    tfms = dl.transform[simps]
    l1, l2 = (tfms[:,:2].transpose(1,2,0) * (grid.T[mask.ravel()] - tfms[:,2]).T).sum(1)
    l3 = 1 - l1 - l2

    ll = np.vstack([l1, l2, l3])
    ll[:,missing] = 0

    from cortex.mapper import samplers
    xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
    sampclass = getattr(samplers, sampler)

    ## Transform fiducial vertex locations to pixel locations using barycentric xfm
    try:
        pia, polys = db.get_surf(subject, "pia", merge=True, nudge=False)
        wm, polys = db.get_surf(subject, "wm", merge=True, nudge=False)
        piacoords = xfm((pia[valid][dl.vertices][simps] * ll[np.newaxis].T).sum(1))
        wmcoords = xfm((wm[valid][dl.vertices][simps] * ll[np.newaxis].T).sum(1))

        valid_p = reduce(np.logical_and, [reduce(np.logical_and, (0 <= piacoords).T), 
            piacoords[:,0] < xfm.shape[2], 
            piacoords[:,1] < xfm.shape[1], 
            piacoords[:,2] < xfm.shape[0]])
        valid_w = reduce(np.logical_and, [reduce(np.logical_and, (0 <= wmcoords).T), 
            wmcoords[:,0] < xfm.shape[2],
            wmcoords[:,1] < xfm.shape[1],
            wmcoords[:,2] < xfm.shape[0]])
        valid = np.logical_and(valid_p, valid_w)
        vidx = np.nonzero(valid)[0]
        mapper = sparse.csr_matrix((mask.sum(), np.prod(xfm.shape)))
        if thick == 1:
            i, j, data = sampclass(piacoords[valid]*depth + wmcoords[valid]*(1-depth), xfm.shape)
            mapper = mapper + sparse.csr_matrix((data / float(thick), (vidx[i], j)),
                                                shape=mapper.shape)
            return mapper

        for t in np.linspace(0, 1, thick+2)[1:-1]:
            i, j, data = sampclass(piacoords[valid]*t + wmcoords[valid]*(1-t), xfm.shape)
            mapper = mapper + sparse.csr_matrix((data / float(thick), (vidx[i], j)),
                                                shape=mapper.shape)
        return mapper

    except IOError:
        fid, polys = db.get_surf(subject, "fiducial", merge=True)
        fidcoords = xfm((fid[valid][dl.vertices][simps] * ll[np.newaxis].T).sum(1))

        valid = reduce(np.logical_and, [reduce(np.logical_and, (0 <= fidcoords).T),
            fidcoords[:,0] < xfm.shape[2],
            fidcoords[:,1] < xfm.shape[1],
            fidcoords[:,2] < xfm.shape[0]])
        vidx = np.nonzero(valid)[0]

        i, j, data = sampclass(fidcoords[valid], xfm.shape)
        csrshape = mask.sum(), np.prod(xfm.shape)
        return sparse.csr_matrix((data, (vidx[i], j)), shape=csrshape)


def _has_cmap(dataview):
    """Checks whether a given dataview has colormap (cmap) information as an
    instance or is an RGB volume and does not have a cmap.
    Returns a dictionary with cmap information for non RGB volumes"""

    from matplotlib import colors, cm, pyplot as plt

    cmapdict = dict()
    if not isinstance(dataview, (dataset.VolumeRGB, dataset.VertexRGB)):
        # Get colormap from matplotlib or pycortex colormaps
        ## -- redundant code, here and in cortex/dataset/views.py -- ##
        if isinstance(dataview.cmap,(str,unicode)):
            if not dataview.cmap in cm.__dict__:
                # unknown colormap, test whether it's in pycortex colormaps
                cmapdir = config.get('webgl', 'colormaps')
                colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
                colormaps = dict(((os.path.split(c)[1][:-4],c) for c in colormaps))
                if not dataview.cmap in colormaps:
                    raise Exception('Unkown color map!')
                I = plt.imread(colormaps[dataview.cmap])
                cmap = colors.ListedColormap(np.squeeze(I))
                # Register colormap while we're at it
                cm.register_cmap(dataview.cmap,cmap)
            else:
                cmap = dataview.cmap
        elif isinstance(dataview.cmap, colors.Colormap):
            # Allow input of matplotlib colormap class
            cmap = dataview.cmap

        cmapdict.update(cmap=cmap, 
                        vmin=dataview.vmin, 
                        vmax=dataview.vmax)

    return cmapdict
