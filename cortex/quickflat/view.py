import io
import os
import tempfile
import binascii
import numpy as np

from .. import utils
from .. import dataset
from .utils import make_flatmap_image
from . import composite


default_colorbar_locations = {
    'left': (.0, .07, .2, .04),
    'center': (.4, .07, .2, .04),
    'right': (.7, .07, .2, .04)
}


def _check_colorbar_location(colorbar_location):
    if isinstance(colorbar_location, (tuple, list)):
        return colorbar_location

    if colorbar_location not in default_colorbar_locations:
        raise ValueError("colorbar_location must be one of {}".format(
            list(default_colorbar_locations.keys())))

    return default_colorbar_locations[colorbar_location]


def make_figure(braindata, recache=False, pixelwise=True, thick=32, sampler='nearest',
                height=1024, dpi=100, depth=0.5, with_rois=True, with_sulci=False,
                with_labels=True, with_colorbar=True, with_borders=False,
                with_dropout=False, with_curvature=False, extra_disp=None,
                with_connected_vertices=False, overlay_file=None,
                linewidth=None, linecolor=None, roifill=None, shadow=None,
                labelsize=None, labelcolor=None, cutout=None, curvature_brightness=None,
                curvature_contrast=None, curvature_threshold=None, fig=None, extra_hatch=None,
                colorbar_ticks=None, colorbar_location='center', roi_list=None,
                nanmean=False, **kwargs):
    """Show a Volume or Vertex on a flatmap with matplotlib.

    Note that **kwargs are ONLY present now for backward compatibility / warnings. No kwargs
    should be used.

    Parameters
    ----------
    braindata : Dataview (e.g. instance of cortex.Volume, cortex.Vertex,...)
        the data you would like to plot on a flatmap
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include
        'trilinear', 'nearest', 'lanczos'; see functions in cortex.mapper.samplers.py for all options
    height : int
        Height of the image to render. Automatically scales the width for the aspect
        of the subject's flatmap
    depth : float
        Value between 0 and 1 for how deep to sample the surface for the flatmap (0 = gray/white matter
        boundary, 1 = pial surface)
    with_rois, with_labels, with_colorbar, with_borders, with_dropout, with_curvature, etc : bool, optional
        Display the rois, labels, colorbar, annotated flatmap borders, etc
    cutout : str
        Name of flatmap cutout with which to clip the full flatmap. Should be the name
        of a sub-layer of the 'cutouts' layer in <filestore>/<subject>/overlays.svg

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
        (R, G, B, A) specification for the fill of each ROI region
    shadow : int, optional
        Standard deviation of the gaussian shadow. Set to 0 if you want no shadow
    labelsize : str, optional
        Font size for the label, e.g. "16pt"
    labelcolor : tuple of float, optional
        (R, G, B, A) specification for the label color
    curvature_brightness : float, optional
        Mean* brightness of background. 0 = black, 1 = white, intermediate values are corresponding
        grayscale values. If None, Defaults to config file value. (*this does not precisely specify
        the mean; the actual mean luminance of the curvature depends on the value for
        `curvature_contrast`. It's easiest to think about it as the mean brightness, though.)
    curvature_contrast : float, optional
        Contrast of curvature. 1 = maximal contrast (black/white), 0 = no contrast (solid color for
        curvature equal to `curvature_brightness`).
    cvmax : float, optional [DEPRECATED! use `curvature_brightness` and `curvature_contrast` instead]
        Maximum value for background curvature colormap. Defaults to config file value.
    cvthr : bool, optional [DEPRECATED! use `curvature_threshold` instead]
        Apply threshold to background curvature
    extra_disp : tuple, optional
        Optional extra display layer from external .svg file. Tuple specifies (filename, layer)
        filename should be a full path. External svg file should be structured exactly as
        overlays.svg for the subject. (Best to just copy overlays.svg somewhere else and add
        layers to it.) Default value is None.
    extra_hatch : tuple, optional
        Optional extra crosshatch-textured layer, given as (DataView, [r, g, b]) tuple.
    colorbar_location : str or tuple, optional
        Location of the colorbar. Default locations are one of
        'left', 'center', 'right' (default 'center').
        Alternatively, a tuple with four floats between 0 and 1 can be passed
        indicating (left, bottom, width, height).
    colorbar_ticks : array-like, optional
        For 1D colormaps indicates the ticks of the colorbar. If None,
        it defaults to equally spaced values between vmin and vmax.
        This parameter is not used for 2D colormaps, and it defaults to the
        vmin, vmax specified in the Volume2D object.
    fig : figure or ax
        figure into which to plot flatmap
    nanmean : bool, optional (default = False)
        If True, NaNs in the data will be ignored when averaging across layers.
    """
    from matplotlib import pyplot as plt

    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.Dataview):
        raise TypeError('Please provide a Dataview (e.g. an instance of cortex.Volume, cortex.Vertex, etc), not a Dataset')
    if fig is None:
        fig_resize = True
        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
    elif isinstance(fig, plt.Figure):
        fig_resize = False
        fig = plt.figure(fig.number)
        ax = fig.add_axes((0, 0, 1, 1))
    elif isinstance(fig, plt.Axes):
        fig_resize = False
        ax = fig
        fig = ax.figure

    # Add data
    data_im, extents = composite.add_data(ax, dataview, pixelwise=pixelwise, thick=thick, sampler=sampler,
                                          height=height, depth=depth, recache=recache, nanmean=nanmean)

    layers = dict(data=data_im)
    # Add curvature
    if with_curvature:
        # backward compatibility
        if any([x in kwargs for x in ['cvmin', 'cvmax', 'cvthr']]):
            import warnings
            warnings.warn(("Use of `cvmin`, `cvmax`, and `cvthr` is deprecated! Please use \n"
                           "`curvature_brightness`, `curvature_contrast`, and `curvature_threshold`\n"
                           "to set appearance of background curvature."))
            legacy_mode = True
            if ('cvmin' in kwargs) and ('cvmax' in kwargs):
                # Assumes that if one is specified, both are; weird case where only one is
                # specified will still break.
                curvature_lims = (kwargs.pop('cvmin'), kwargs.pop('cvmax'))
            else:
                curvature_lims = 0.5
            if 'cvthr' in kwargs:
                curvature_threshold = kwargs.pop('cvthr')
        else:
            curvature_lims = 0.5
            legacy_mode = False
        curv_im = composite.add_curvature(ax, dataview, extents,
                                          brightness=curvature_brightness,
                                          contrast=curvature_contrast,
                                          threshold=curvature_threshold,
                                          curvature_lims=curvature_lims,
                                          legacy_mode=legacy_mode,
                                          recache=recache)
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

        drop_im = composite.add_hatch(ax, hatch_data, extents=extents, height=height,
                                      sampler=sampler, recache=recache)
        layers['dropout'] = drop_im
    # Add extra hatching
    if extra_hatch is not None:
        hatch_data2, hatch_color = extra_hatch
        hatch_im = composite.add_hatch(ax, hatch_data2, extents=extents, height=height,
                                       sampler=sampler, recache=recache)
        layers['hatch'] = hatch_im
    # Add rois
    if with_rois:
        roi_im = composite.add_rois(ax, dataview, extents=extents, height=height, linewidth=linewidth, linecolor=linecolor,
                                    roifill=roifill, shadow=shadow, labelsize=labelsize, labelcolor=labelcolor,
                                    with_labels=with_labels, overlay_file=overlay_file,
                                    roi_list=roi_list)
        layers['rois'] = roi_im
    # Add sulci
    if with_sulci:
        sulc_im = composite.add_sulci(ax, dataview, extents=extents, height=height, linewidth=linewidth, linecolor=linecolor,
                                      shadow=shadow, labelsize=labelsize, labelcolor=labelcolor, with_labels=with_labels,
                                      overlay_file=overlay_file)
        layers['sulci'] = sulc_im
    # Add custom
    if extra_disp is not None:
        svgfile, layer = extra_disp
        custom_im = composite.add_custom(ax, dataview, svgfile, layer, height=height, extents=extents,
                                         linewidth=linewidth, linecolor=linecolor, shadow=shadow, labelsize=labelsize,
                                         labelcolor=labelcolor, with_labels=with_labels)
        layers['custom'] = custom_im
    # Add connector lines btw connected vertices
    if with_connected_vertices:
        vertex_lines = composite.add_connected_vertices(ax, dataview, recache=recache)

    ax.axis('off')
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])

    if fig_resize:
        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))

    # Add (apply) cutout of flatmap
    if cutout is not None:
        extents = composite.add_cutout(ax, cutout, dataview, layers, overlay_file=overlay_file)

    if with_colorbar:
        colorbar_location = _check_colorbar_location(colorbar_location)
        # Allow 2D colorbars:
        if isinstance(dataview, dataset.view2D.Dataview2D):
            colorbar_ticks = np.round([
                    dataview.vmin, dataview.vmax,
                    dataview.vmin2, dataview.vmax2
                ], 2)
            colorbar = composite.add_colorbar_2d(
                ax, dataview.cmap, colorbar_ticks,
                colorbar_location=colorbar_location)
        else:
            colorbar = composite.add_colorbar(
                ax, data_im,
                colorbar_location=colorbar_location,
                colorbar_ticks=colorbar_ticks
            )
        # Reset axis to main figure axis
        plt.sca(ax)

    return fig

def make_png(fname, braindata, recache=False, pixelwise=True, sampler='nearest', height=1024,
             bgcolor=None, dpi=100, **kwargs):
    """Create a PNG of the VertexData or VolumeData on a flatmap.

    Parameters
    ----------
    fname : str
        Filename for where to save the PNG file
    braindata : Dataview (e.g. instance of cortex.Volume, cortex.Vertex, ...)
        the data you would like to plot on a flatmap
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. Useful if you've made changes to the alignment
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
        'trilinear', 'nearest', 'lanczos'; see functions in cortex.mapper.samplers.py for all options

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
        (R, G, B, A) specification for the fill of each ROI region
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

def make_svg(fname, braindata, with_labels=False, with_curvature=True, layers=['rois'],
             height=1024, overlay_file=None, with_dropout=False, **kwargs):
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
    with_curvature : bool
        Whether to include background curvature
    layers : list
        List of layer names to show
    height : int
        Height of PNG in pixels
    overlay_file : str
        Custom ROI overlays file to use
    with_dropout : bool or Dataview
        If True or a cortex.Dataview object, hatches will be overlaid on top of the
        flatmap to indicate areas with dropout. If set to True, the dropout areas will
        be estimated from the intensity of the reference image. If set to a
        cortex.Dataview object, values in the dataset will be considered dropout areas.
        The transparency of the hatches is proportional to the intensity of the values
        in the dropout dataset.
    """
    fp = io.BytesIO()
    from matplotlib.pyplot import imsave

    ## Render PNG file & retrieve image data
    arr, extents = make_flatmap_image(braindata, height=height, **kwargs)
    # Set nans to alpha = 0. to enable transparency when saving as PNG
    mask_nans = np.isnan(arr[..., 3])
    arr[mask_nans, 3] = 0.

    if hasattr(braindata, 'cmap'):
        imsave(fp, arr, cmap=braindata.cmap, vmin=braindata.vmin, vmax=braindata.vmax)
    else:
        imsave(fp, arr)
    fp.seek(0)
    pngdata = binascii.b2a_base64(fp.read())
    image_data = [pngdata]

    if with_curvature:
        # no options. learn to love it.
        from cortex import db
        fpc = io.BytesIO()
        curv_vertices = db.get_surfinfo(braindata.subject)
        curv_arr, _ = make_flatmap_image(curv_vertices, height=height)
        mask = np.isnan(curv_arr)
        curv_arr = np.where(curv_arr > 0, 0.5, 0.25)
        curv_arr[mask] = np.nan

        imsave(fpc, curv_arr, cmap='Greys_r', vmin=0, vmax=1)
        fpc.seek(0)
        image_data = [binascii.b2a_base64(fpc.read()), pngdata]

    # Add dropout -- modified from quickflat.view.make_figure
    if with_dropout:
        dataview = dataset.normalize(braindata)
        # Support old api:
        if isinstance(with_dropout, dataset.Dataview):
            hatch_data = with_dropout
        else:
            hatch_data = utils.get_dropout(dataview.subject, dataview.xfmname)
        sampler = kwargs.get("sampler", "nearest")
        recache = kwargs.get("recache", False)
        hatch_space = 4
        hatch_color = (0, 0, 0)
        hatchim = composite._make_hatch_image(
            hatch_data, height, sampler, recache=recache, hatch_space=hatch_space
        )
        hatchim[:, :, 0] = hatch_color[0]
        hatchim[:, :, 1] = hatch_color[1]
        hatchim[:, :, 2] = hatch_color[2]
        fpc = io.BytesIO()
        imsave(fpc, hatchim)
        fpc.seek(0)
        # Add dropout above data layer
        image_data += [binascii.b2a_base64(fpc.read())]

    ## Create and save SVG file
    roipack = utils.db.get_overlay(braindata.subject, overlay_file)
    roipack.get_svg(fname, layers=layers, labels=with_labels, with_ims=image_data)


def make_gif(output_destination, volumes, frame_duration=1, **figure_kwargs):
    """Make an animated gif from several pycortex volumes

    Parameters
    ----------
    output_destination : str or stream-like
        The destination for the created gif. If a str, saves to a file. If stream-like (file handle
        or io.BytesIO), writes to the stream
    volumes : dict of pycortex Volumes
    duration : float
        The duration of each frame in seconds
    **figure_kwargs
        Passed to `cortex.quickflat.make_figure`

    Returns
    -------
    If output_destination is a file path, return the path. If stream-like, return the stream data.
    """
    import imageio
    from matplotlib import pyplot as plt

    tmpdir = tempfile.TemporaryDirectory()

    images = []
    for i, name in enumerate(volumes):
        fig = plt.figure(figsize=(12, 6), dpi=100)
        _ = make_figure(volumes[name], fig=fig, **figure_kwargs)
        _ = fig.suptitle(name)
        path = os.path.join(tmpdir.name, str(i) + '.png')
        fig.savefig(path)
        images.append(imageio.imread(path))
        _ = plt.close(fig)

    tmpdir.cleanup()

    imageio.mimsave(output_destination, images, format='gif', duration=frame_duration)

    if hasattr(output_destination, 'seek'):
        output_destination.seek(0)


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
