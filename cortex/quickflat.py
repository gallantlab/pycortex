import io
import os
import sys
import time
import glob
import pickle
import binascii
import numpy as np

from . import utils
from . import dataset
from .db import surfs

def make_figure(braindata, recache=False, pixelwise=True, thick=32, sampler='nearest', height=1024, dpi=100, depth=0.5,
                with_rois=True, with_labels=True, with_colorbar=True, with_borders=False, with_dropout=False, 
                linewidth=None, linecolor=None, roifill=None, shadow=None, labelsize=None, labelcolor=None,
                **kwargs):
    """Show a VolumeData or VertexData on a flatmap with matplotlib. Additional kwargs are passed on to
    matplotlib's imshow command.

    Parameters
    ----------
    braindata : DataView
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
        Height of the image to render. Automatically scales the width for the aspect of the subject's flatmap
    with_rois, with_labels, with_colorbar, with_borders, with_dropout : bool, optional
        Display the rois, labels, colorbar, annotated flatmap borders, and cross-hatch dropout?

    Other Parameters
    ----------------
    dpi : int
        DPI of the generated image. Only applies to the scaling of matplotlib elements, specifically the colormap
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
    from matplotlib import cm, pyplot as plt
    from matplotlib.collections import LineCollection

    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.DataView):
        raise TypeError('Please provide a DataView, not a Dataset')
    if dataview.data.movie:
        raise ValueError('Cannot flatten movie volumes')
    
    im, extents = make(dataview.data, recache=recache, pixelwise=pixelwise, sampler=sampler, height=height, thick=thick, depth=depth)
    
    fig = plt.figure()
    ax = fig.add_axes((0,0,1,1))
    cimg = ax.imshow(im[::-1], 
        aspect='equal', 
        extent=extents, 
        cmap=dataview.cmap, 
        vmin=dataview.vmin, 
        vmax=dataview.vmax,
        origin='lower')
    ax.axis('off')
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])

    if with_colorbar:
        cbar = fig.add_axes((.4, .07, .2, .04))
        fig.colorbar(cimg, cax=cbar, orientation='horizontal')

    if with_dropout:
        dax = fig.add_axes((0,0,1,1))
        dmap, extents = make(utils.get_dropout(braindata.subject, braindata.xfmname),
                    height=height, sampler=sampler)
        hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
        hatchspace = 4
        hatchpat = (hx+hy)%(2*hatchspace) < 2
        hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
        hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
        hatchim[:,:,3] *= (dmap>0.5).astype(float)
        dax.imshow(hatchim[::-1], aspect="equal", interpolation="nearest", extent=extents, origin='lower')
    
    if with_borders:
        border = _gen_flat_border(braindata.data.subject, im.shape[0])
        bax = fig.add_axes((0,0,1,1))
        blc = LineCollection(border[0], linewidths=3.0,
                             colors=[['r','b'][mw] for mw in border[1]])
        bax.add_collection(blc)
        #bax.invert_yaxis()
    
    if with_rois:
        roi = surfs.getOverlay(braindata.data.subject, linewidth=linewidth, linecolor=linecolor, roifill=roifill, shadow=shadow, labelsize=labelsize, labelcolor=labelcolor)
        roitex = roi.get_texture(height, labels=with_labels)
        roitex.seek(0)
        oax = fig.add_axes((0,0,1,1))
        oimg = oax.imshow(plt.imread(roitex)[::-1],
            aspect='equal', 
            interpolation='bicubic', 
            extent=extents, 
            zorder=3,
            origin='lower')

    return fig

def make_png(fname, braindata, recache=False, pixelwise=True, sampler='nearest', height=1024,
    bgcolor=None, dpi=100, **kwargs):
    """
    make_png(name, braindata, recache=False, pixelwise=True, thick=32, sampler='nearest', height=1024, dpi=100,
                with_rois=True, with_labels=True, with_colorbar=True, with_borders=False, with_dropout=False, 
                linewidth=None, linecolor=None, roifill=None, shadow=None, labelsize=None, labelcolor=None,
                **kwargs)

    Create a PNG of the VertexData or VolumeData on a flatmap.

    Parameters
    ----------
    fname : str
        Filename for where to save the PNG file
    braindata : DataView
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
        Height of the image to render. Automatically scales the width for the aspect of the subject's flatmap
    with_rois, with_labels, with_colorbar, with_borders, with_dropout : bool, optional
        Display the rois, labels, colorbar, annotated flatmap borders, and cross-hatch dropout?

    Other Parameters
    ----------------
    dpi : int
        DPI of the generated image. Only applies to the scaling of matplotlib elements, specifically the colormap
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
    fig = make_figure(braindata, recache=recache, pixelwise=pixelwise, sampler=sampler, height=height, **kwargs)
    imsize = fig.get_axes()[0].get_images()[0].get_size()
    fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
    if bgcolor is None:
        fig.savefig(fname, transparent=True, dpi=dpi)
    else:
        fig.savefig(fname, facecolor=bgcolor, transparent=False, dpi=dpi)
    plt.close()

def make_svg(fname, braindata, recache=False, pixelwise=True, sampler='nearest', height=1024, thick=32, depth=0.5, **kwargs):
    dataview = dataset.normalize(braindata)
    if not isinstance(dataview, dataset.DataView):
        raise TypeError('Please provide a DataView, not a Dataset')
    if dataview.data.movie:
        raise ValueError('Cannot flatten movie volumes')
    ## Create quickflat image array
    im, extents = make(dataview.data, recache=recache, pixelwise=pixelwise, sampler=sampler, height=height, thick=thick, depth=depth)
    ## Convert to PNG
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()
    from matplotlib.pylab import imsave
    imsave(fp, im, cmap=dataview.cmap, vmin=dataview.vmin, vmax=dataview.vmax, **kwargs)
    fp.seek(0)
    pngdata = binascii.b2a_base64(fp.read())
    ## Create and save SVG file
    roipack = utils.get_roipack(dataview.data.subject)
    roipack.get_svg(fname, labels=True, with_ims=[pngdata])

def make(braindata, height=1024, **kwargs):
    if not isinstance(braindata, dataset.BrainData):
        raise TypeError('Invalid type for quickflat')
    if braindata.movie:
        raise ValueError('Cannot flatten multiple volumes')

    npz = surfs.getAnat(braindata.subject, "flatmask", height=height)
    mask = npz['mask'].T
    extents = npz['extents']
    npz.close()
    
    if isinstance(braindata, dataset.VertexData):
        pixmap = get_flatcache(braindata.subject, None, height=height, **kwargs)
        data = braindata.vertices
    else:
        pixmap = get_flatcache(braindata.subject, braindata.xfmname, height=height, **kwargs)
        data = braindata.volume

    if braindata.raw:
        img = np.zeros(mask.shape+(4,), dtype=np.uint8)
        img[mask] = pixmap * data.reshape(-1, 4)
        return img.transpose(1,0,2)[::-1], extents
    else:
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan*np.ones(mask.shape)).astype(braindata.data.dtype)
        mimg = (np.nan*np.ones(badmask.shape)).astype(braindata.data.dtype)
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

def make_movie(name, data, subject, xfmname, recache=False, height=1024, sampler='nearest', dpi=100, tr=2, interp='linear', fps=30, vcodec='libtheora', bitrate="8000k", vmin=None, vmax=None, **kwargs):
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


def _gen_flat_border(subject, height=1024):
    from . import polyutils
    flatpts, flatpolys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    flatpolyset = set(map(tuple, flatpolys))
    
    fidpts, fidpolys = surfs.getSurf(subject, "fiducial", merge=True, nudge=True)
    fidpolyset = set(map(tuple, fidpolys))
    fidonlypolys = fidpolyset - flatpolyset
    fidonlypolyverts = np.unique(np.array(list(fidonlypolys)).ravel())
    
    fidonlyverts = np.setdiff1d(fidpolys.ravel(), flatpolys.ravel())
    
    import networkx as nx
    def iter_surfedges(tris):
        for a,b,c in tris:
            yield a,b
            yield b,c
            yield a,c

    def make_surface_graph(tris):
        graph = nx.Graph()
        graph.add_edges_from(iter_surfedges(tris))
        return graph

    bounds = [p for p in polyutils.trace_poly(polyutils.boundary_edges(flatpolys))]
    allbounds = np.hstack(bounds)
    
    g = make_surface_graph(fidonlypolys)
    fog = g.subgraph(fidonlyverts)
    badverts = np.array([v for v,d in fog.degree().iteritems() if d<2])
    g.remove_nodes_from(badverts)
    fog.remove_nodes_from(badverts)
    mwallset = set.union(*(set(g[v]) for v in fog.nodes())) & set(allbounds)
    #cutset = (set(g.nodes()) - mwallset) & set(allbounds)

    mwallbounds = [np.in1d(b, mwallset) for b in bounds]
    changes = [np.nonzero(np.diff(b.astype(float))!=0)[0]+1 for b in mwallbounds]
    
    #splitbounds = [np.split(b, c) for b,c in zip(bounds, changes)]
    splitbounds = []
    for b,c in zip(bounds, changes):
        sb = []
        rb = [b[-1]] + b
        rc = [1] + (c + 1).tolist() + [len(b)]
        for ii in range(len(rc)-1):
            sb.append(rb[rc[ii]-1 : rc[ii+1]])
        splitbounds.append(sb)
    
    ismwall = [[s.mean()>0.5 for s in np.split(mwb, c)] for mwb,c in zip(mwallbounds, changes)]
    
    aspect = (height / (flatpts.max(0) - flatpts.min(0))[1])
    lpts = (flatpts - flatpts.min(0)) * aspect
    rpts = (flatpts - flatpts.min(0)) * aspect
    
    #im = Image.new('RGBA', (int(aspect * (flatpts.max(0) - flatpts.min(0))[0]), height))
    #draw = ImageDraw.Draw(im)

    ismwalls = []
    lines = []
    
    for bnds, mw, pts in zip(splitbounds, ismwall, [lpts, rpts]):
        for pbnd, pmw in zip(bnds, mw):
            #color = {True:(0,0,255,255), False:(255,0,0,255)}[pmw]
            #draw.line(pts[pbnd,:2].ravel().tolist(), fill=color, width=2)
            ismwalls.append(pmw)
            lines.append(pts[pbnd,:2])
    
    #return np.array(im)[::-1]/255.0
    return lines, ismwalls

def _make_vertex_cache(subject, height=1024):
    from scipy import sparse
    from scipy.spatial import cKDTree
    flat, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)
    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)

    npz = surfs.getAnat(subject, "flatmask", height=height, recache=True)
    mask = npz['mask'].T
    npz.close()
    assert mask.shape[0] == width and mask.shape[1] == height

    kdt = cKDTree(flat[valid,:2])
    dist, vert = kdt.query(grid.T[mask.ravel()])
    dataij = (np.ones((len(vert),)), np.array([np.arange(len(vert)), valid[vert]]))
    return sparse.csr_matrix(dataij, shape=(mask.sum(), len(flat)))

def _make_pixel_cache(subject, xfmname, height=1024, thick=32, depth=0.5, sampler='nearest'):
    from scipy import sparse
    from scipy.spatial import cKDTree, Delaunay
    flat, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)
    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)
    
    npz = surfs.getAnat(subject, "flatmask", height=height)
    mask = npz['mask'].T
    npz.close()
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
    xfm = surfs.getXfm(subject, xfmname, xfmtype='coord')
    sampclass = getattr(samplers, sampler)

    ## Transform fiducial vertex locations to pixel locations using barycentric xfm
    try:
        pia, polys = surfs.getSurf(subject, "pia", merge=True, nudge=False)
        wm, polys = surfs.getSurf(subject, "wm", merge=True, nudge=False)
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
            mapper = mapper + sparse.csr_matrix((data / thick, (vidx[i], j)), shape=mapper.shape)
            return mapper

        for t in np.linspace(0, 1, thick+2)[1:-1]:
            i, j, data = sampclass(piacoords[valid]*t + wmcoords[valid]*(1-t), xfm.shape)
            mapper = mapper + sparse.csr_matrix((data / thick, (vidx[i], j)), shape=mapper.shape)
        return mapper

    except IOError:
        fid, polys = surfs.getSurf(subject, "fiducial", merge=True)
        fidcoords = xfm((fid[valid][dl.vertices][simps] * ll[np.newaxis].T).sum(1))

        valid = reduce(np.logical_and, [reduce(np.logical_and, (0 <= fidcoords).T),
            fidcoords[:,0] < xfm.shape[2],
            fidcoords[:,1] < xfm.shape[1],
            fidcoords[:,2] < xfm.shape[0]])
        vidx = np.nonzero(valid)[0]

        i, j, data = sampclass(fidcoords[valid], xfm.shape)
        csrshape = mask.sum(), np.prod(xfm.shape)
        return sparse.csr_matrix((data, (vidx[i], j)), shape=csrshape)

def get_flatcache(subject, xfmname, pixelwise=True, thick=32, sampler='nearest', recache=False, height=1024, depth=0.5):
    cachedir = surfs.getFiles(subject)['cachedir']
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
