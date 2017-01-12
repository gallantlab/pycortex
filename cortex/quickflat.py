import io
import os
import glob
import binascii
import numpy as np

from . import utils
from . import dataset
from .database import db
from .options import config

def make_figure(braindata, recache=False, pixelwise=True, thick=32, sampler='nearest',
                height=1024, dpi=100, depth=0.5, with_rois=True, with_sulci=False,
                with_labels=True, with_colorbar=True, with_borders=False, 
                with_dropout=False, with_curvature=False, extra_disp=None, 
                linewidth=None, linecolor=None, roifill=None, shadow=None,
                labelsize=None, labelcolor=None, cutout=None, cvmin=None,
                cvmax=None, cvthr=None, fig=None, extra_hatch=None,
                colorbar_ticks=None, colorbar_ticklabels=None,
                colorbar_ticklabelsize=None, colorbar_location=(.4, .07, .2, .04),
                **kwargs):
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
    colorbar_ticks: list of float, optional
        Location of of colorbar ticks
    colorbar_ticklabels: list of str, optional
        Labels to use at each colorbar tick
    colorbar_ticklabelsize: int, optional
        Fontsize of colorbarticklabels
    """
    from matplotlib import colors,cm, pyplot as plt
    from matplotlib.collections import LineCollection

    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.Dataview):
        raise TypeError('Please provide a Dataview, not a Dataset')
    
    if fig is None:
        fig_resize = True
        fig = plt.figure()
    else:
        fig_resize = False
        fig = plt.figure(fig.number)

    im, extents = make(dataview, recache=recache, pixelwise=pixelwise, sampler=sampler,
                       height=height, thick=thick, depth=depth)

    if cutout:
        roi = db.get_overlay(dataview.subject,
                             otype='cutouts',
                             roifill=(0.,0.,0.,0.),
                             linecolor=(0.,0.,0.,0.),
                             linewidth=0.)

        # Set ONLY desired cutout to be white
        roi.rois[cutout].set(roifill=(1.,1.,1.,1.),
                             linewidth=2.,
                             linecolor=(1.,1.,1.,1.))
        roitex = roi.get_texture(height, labels=False)
        roitex.seek(0)
        co = plt.imread(roitex)[:,:,0] # Cutout image
        if not np.any(co):
            raise Exception('No pixels in cutout region %s!'%cutout)

        # STUPID BUT NECESSARY 1-PIXEL CHECK:
        if any([np.abs(aa-bb)>0 and np.abs(aa-bb)<2 for aa,bb in zip(im.shape,co.shape)]):
            from scipy.misc import imresize
            co = imresize(co,im.shape[:2]).astype(np.float32)/255.

        # Alpha
        if im.dtype == np.uint8:
            im = np.cast['float32'](im)/255.
            im[:,:,3]*=co
            h,w,cdim = [float(v) for v in im.shape]
        else:
            im[co==0] = np.nan
            h,w = [float(v) for v in im.shape]

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
        curv,ee = make(db.get_surfinfo(dataview.subject),recache=recache,height=height)
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

    kwargs = dict(aspect='equal', 
        extent=extents, 
        origin='lower')
    
    # Check whether dataview has a cmap instance
    cmapdict = _has_cmap(dataview)
    kwargs.update(cmapdict)

    ax = fig.add_axes((0,0,1,1))
    cimg = ax.imshow(im[iy[1]:iy[0]:-1,ix[0]:ix[1]], **kwargs)
    ax.axis('off')
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])


    if with_colorbar and not isinstance(dataview, dataset.Volume2D):
        cbar_axis = fig.add_axes(colorbar_location)
        cbar = fig.colorbar(cimg, cax=cbar_axis, orientation='horizontal',
                     ticks=colorbar_ticks)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
        if colorbar_ticklabelsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_ticklabelsize)

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
    
    overlays = []
    if with_rois:
        roi = db.get_overlay(dataview.subject,
                             linewidth=linewidth,
                             linecolor=linecolor,
                             roifill=roifill,
                             shadow=shadow,
                             labelsize=labelsize,
                             labelcolor=labelcolor)
        overlays.append(roi)
    if with_sulci:
        sulc = db.get_overlay(dataview.subject,
                              otype='sulci',
                              linewidth=linewidth,
                              linecolor=linecolor,
                              shadow=shadow,
                              labelsize=labelsize,
                              labelcolor=labelcolor)
        overlays.append(sulc)
    if not extra_disp is None:
        svgfile,layer = extra_disp
        if not isinstance(layer,(list,tuple)):
            layer = [layer]
        for extralayer in layer:
            # Allow multiple extra layer overlays
            disp = db.get_overlay(dataview.subject,
                              otype='external',
                              shadow=shadow,
                              labelsize=labelsize,
                              labelcolor=labelcolor,
                              layer=extralayer,
                              svgfile=svgfile)
            overlays.append(disp)
    for oo in overlays:
        roitex = oo.get_texture(height, labels=with_labels, size=labelsize)
        roitex.seek(0)
        oax = fig.add_axes((0,0,1,1))
        roi_im = plt.imread(roitex)
        if cutout: 
            # STUPID BUT NECESSARY 1-PIXEL CHECK:
            if any([np.abs(aa-bb)>0 and np.abs(aa-bb)<2 for aa,bb in zip(im.shape,roi_im.shape)]):
                from scipy.misc import imresize
                co = imresize(co,roi_im.shape[:2]).astype(np.float32)/255.
            roi_im[:,:,3]*=co

        oimg = oax.imshow(roi_im[iy[1]:iy[0]:-1,ix[0]:ix[1]],
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

def make(braindata, height=1024, recache=False, **kwargs):
    mask, extents = get_flatmask(braindata.subject, height=height, recache=recache)
    
    if not hasattr(braindata, "xfmname"):
        pixmap = get_flatcache(braindata.subject,
                               None,
                               height=height,
                               recache=recache,
                               **kwargs)
        data = braindata.vertices
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
        raise ValueError("Cannot flatten movie views")

    if data.dtype == np.uint8:
        img = np.zeros(mask.shape+(4,), dtype=np.uint8)
        img[mask] = pixmap * data.reshape(-1, 4)
        return img.transpose(1,0,2)[::-1], extents
    else:
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan*np.ones(mask.shape))
        mimg = (np.nan*np.ones(badmask.shape))
        mimg[badmask] = (pixmap*data.ravel())[badmask]
        img[mask] = mimg

        return img.T[::-1], extents

def overlay_rois(im, subject, name=None, height=1024, labels=True, **kwargs):
    import shlex
    import subprocess as sp
    from matplotlib.pylab import imsave

    if name is None:
        name = 'png:-'

    key = (subject, labels)
    if key not in rois:
        print("loading %s"%subject)
        rois[key] = utils.get_roipack(subject).get_texture(height, labels=labels)
    cmd = "composite {rois} - {name}".format(rois=rois[key].name, name=name)
    proc = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.PIPE)

    fp = io.StringIO()
    imsave(fp, im, **kwargs)
    fp.seek(0)
    out, err = proc.communicate(fp.read())
    if len(out) > 0:
        fp = io.StringIO()
        fp.write(out)
        fp.seek(0)
        return fp

def show(*args, **kwargs):
    raise DeprecationWarning("Use quickflat.make_figure instead")
    return make_figure(*args, **kwargs)

def make_movie(name, data, subject, xfmname, recache=False, height=1024,
               sampler='nearest', dpi=100, tr=2, interp='linear', fps=30,
               vcodec='libtheora', bitrate="8000k", vmin=None, vmax=None, **kwargs):
    raise NotImplementedError
    import sys
    import shlex
    import shutil
    import tempfile
    import subprocess as sp
    import multiprocessing as mp
    
    from scipy.interpolate import interp1d

    #make the flatmaps
    ims,extents = make(data, subject, xfmname, recache=recache, height=height, sampler=sampler)
    if vmin is None:
        vmin = np.nanmin(ims)
    if vmax is None:
        vmax = np.nanmax(ims)

    #Create the matplotlib figure
    fig = make_figure(ims[0], subject, vmin=vmin, vmax=vmax, **kwargs)
    fig.set_size_inches(np.array([ims.shape[2], ims.shape[1]]) / float(dpi))
    img = fig.axes[0].images[0]

    #set up interpolation
    times = np.arange(0, len(ims)*tr, tr)
    interp = interp1d(times, ims, kind=interp, axis=0, copy=False)
    frames = np.linspace(0, times[-1], (len(times)-1)*tr*fps+1)
    
    try:
        path = tempfile.mkdtemp()
        impath = os.path.join(path, "im%09d.png")

        def overlay(idxts):
            idx, ts = idxts
            img.set_data(interp(ts))
            fig.savefig(impath%idx, transparent=True, dpi=dpi)

        list(map(overlay, enumerate(frames)))

        cmd = "avconv -i {path} -vcodec {vcodec} -r {fps} -b {br} {name}".format(path=impath, vcodec=vcodec, fps=fps, br=bitrate, name=name)
        sp.call(shlex.split(cmd))
    finally:
        shutil.rmtree(path)

def get_flatmask(subject, height=1024, recache=False):
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

def _make_hatch_image(dropout_data, height, sampler, recache=False):
    dmap, ee = make(dropout_data, height=height, sampler=sampler, recache=recache)
    hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    hatchspace = 4
    hatchpat = (hx+hy)%(2*hatchspace) < 2
    hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
    hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
    hatchim[:,:,3] *= np.clip(dmap, 0, 1).astype(float)

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
        elif isinstance(dataview.cmap,colors.Colormap):
            # Allow input of matplotlib colormap class
            cmap = dataview.cmap

        cmapdict.update(cmap=cmap, 
                        vmin=dataview.vmin, 
                        vmax=dataview.vmax)

    return cmapdict



def is_str(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)
