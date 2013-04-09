import io
import os
import sys
import time
import glob
import pickle
import binascii
import numpy as np

from . import utils
from .db import surfs

def _gen_flat_mask(subject, height=1024):
    from . import polyutils
    import Image
    import ImageDraw
    pts, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    bounds = [p for p in polyutils.trace_poly(polyutils.boundary_edges(polys))]
    left, right = bounds[0], bounds[1]
    aspect = (height / (pts.max(0) - pts.min(0))[1])
    lpts = (pts[left] - pts.min(0)) * aspect
    rpts = (pts[right] - pts.min(0)) * aspect

    im = Image.new('L', (int(aspect * (pts.max(0) - pts.min(0))[0]), height))
    draw = ImageDraw.Draw(im)
    draw.polygon(lpts[:,:2].ravel().tolist(), fill=255)
    draw.polygon(rpts[:,:2].ravel().tolist(), fill=255)
    return np.array(im) > 0

def _make_flat_cache(subject, xfmname, height=1024):
    from scipy.spatial import cKDTree

    flat, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)

    mask = _gen_flat_mask(subject, height=height).T
    assert mask.shape[0] == width and mask.shape[1] == height

    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)
    kdt = cKDTree(flat[valid,:2])
    dist, idx = kdt.query(grid.T[mask.ravel()])

    return valid[idx], mask

def _gen_flat_border(subject, height=1024):
    from . import polyutils
    #import Image
    #import ImageDraw
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

cache = dict()
def get_cache(subject, xfmname, recache=False, height=1024):
    key = (subject, xfmname, height)
    if not recache and key in cache:
        return cache[key]

    cacheform = surfs.getFiles(subject)['flatcache']
    cachefile = cacheform.format(xfmname=xfmname, height=height, date="*")
    #pull a list of candidate cache files
    files = glob.glob(cachefile)
    if len(files) < 1 or recache:
        #if recaching, delete all existing files
        for f in files:
            os.unlink(f)
        print("Generating a flatmap cache")
        #pull points and transform from database
        verts, mask = _make_flat_cache(subject, xfmname, height=height)
        #save them into the proper file
        date = time.strftime("%Y%m%d")
        cachename = cacheform.format(xfmname=xfmname, height=height, date=date)
        pickle.dump((verts, mask), open(cachename, "w"), 2)
    else:
        verts, mask = pickle.load(open(files[0]))

    cache[key] = verts, mask
    return verts, mask

def make(data, subject, xfmname, recache=False, height=1024, projection='nearest', **kwargs):
    mapper = utils.get_mapper(subject, xfmname, type=projection, **kwargs)
    verts, mask = get_cache(subject, xfmname, recache=recache, height=height)

    mdata = np.hstack(mapper(data))
    if mdata.dtype == np.uint8:
        mdata = mdata.swapaxes(-1, -2)
        if mdata.ndim == 2:
            mdata = mdata[np.newaxis]
        shape = (mdata.shape[0],) + mask.shape + (mdata.shape[-1],)
    elif mdata.ndim == 1:
        mdata = mdata[np.newaxis]
        shape = (mdata.shape[0],) + mask.shape

    img = (np.nan*np.ones(shape)).astype(mdata.dtype)
    img[:, mask] = mdata[:,verts]
    return img.swapaxes(1, 2)[:,::-1].squeeze()

rois = dict() ## lame
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

def make_figure(data, subject, xfmname, recache=False, height=1024, projection='nearest',
                with_rois=True, labels=True, colorbar=True, dpi=100,
                with_borders=False, with_dropout=False, **kwargs):
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    im = make(data, subject, xfmname, recache=recache, height=height, projection=projection)
    
    fig = plt.figure()
    ax = fig.add_axes((0,0,1,1))
    cimg = ax.imshow(im, aspect='equal', origin="upper", **kwargs)
    ax.axis('off')

    if colorbar:
        cbar = fig.add_axes((.4, .07, .2, .04))
        fig.colorbar(cimg, cax=cbar, orientation='horizontal')

    if with_dropout:
        dax = fig.add_axes((0,0,1,1))
        dmap = make(np.hstack(utils.get_dropout(subject, xfmname)), subject, xfmname,
                    height=height, projection=projection)
        hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
        hatchspace = 4
        hatchpat = (hx+hy)%(2*hatchspace) < 2
        hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
        hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
        hatchim[:,:,3] *= (dmap>0.5).astype(float)
        dax.imshow(hatchim[::-1], aspect="equal", interpolation="nearest", origin="upper")
    
    if with_borders:
        key = (subject, "borderlines")
        if key not in rois:
            border = _gen_flat_border(subject, im.shape[0])
            rois[key] = border

        bax = fig.add_axes((0,0,1,1))
        blc = LineCollection(rois[key][0], linewidths=3.0,
                             colors=[['r','b'][mw] for mw in rois[key][1]])
        bax.add_collection(blc)
        bax.invert_yaxis()
    
    if with_rois:
        key = (subject, labels)
        if key not in rois:
            roi = utils.get_roipack(subject)
            rois[key] = roi.get_texture(im.shape[0], labels=labels)
        rois[key].seek(0)
        oax = fig.add_axes((0,0,1,1))
        oimg = oax.imshow(plt.imread(rois[key]),
                          aspect='equal', interpolation='bicubic', origin="upper", zorder=3)

    return fig

def make_png(name, data, subject, xfmname, recache=False, height=1024, projection='nearest',
             bgcolor=None, dpi=100, **kwargs):
    from matplotlib import pyplot as plt
    fig = make_figure(data, subject, xfmname, recache, height, projection, **kwargs)
    imsize = fig.get_axes()[0].get_images()[0].get_size()
    fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
    if bgcolor is None:
        fig.savefig(name, transparent=True, dpi=dpi)
    else:
        fig.savefig(name, facecolor=bgcolor, transparent=False, dpi=dpi)
    plt.close()

def show(*args, **kwargs):
    raise DeprecationWarning("Use quickflat.make_figure instead")
    return make_figure(*args, **kwargs)

def make_svg(name, data, subject, xfmname, recache=False, height=1024, projection="nearest", **kwargs):
    ## Create quickflat image array
    im = make(data, subject, xfmname, recache=recache, height=height, projection=projection)
    ## Convert to PNG
    try:
        import cStringIO
        fp = cStringIO.StringIO()
    except:
        fp = io.StringIO()
    from matplotlib.pylab import imsave
    imsave(fp, im, **kwargs)
    fp.seek(0)
    pngdata = binascii.b2a_base64(fp.read())
    ## Create and save SVG file
    roipack = utils.get_roipack(subject)
    roipack.get_svg(name, labels=True, with_ims=[pngdata])

def make_movie(name, data, subject, xfmname, recache=False, height=1024, projection='nearest', dpi=100, tr=2, interp='linear', fps=30, vcodec='libtheora', bitrate="8000k", vmin=None, vmax=None, **kwargs):
    import sys
    import shlex
    import shutil
    import tempfile
    import subprocess as sp
    import multiprocessing as mp
    
    from scipy.interpolate import interp1d

    #make the flatmaps
    ims = make(data, subject, xfmname, recache=recache, height=height, projection=projection)
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
