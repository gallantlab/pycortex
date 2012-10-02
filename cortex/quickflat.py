import os
import time
import glob
import Image
import cPickle
import cStringIO
import binascii
import numpy as np

import db
import utils

def _gen_flat_mask(subject, height=1024):
    import polyutils
    import Image
    import ImageDraw
    pts, polys, norm = db.surfs.getVTK(subject, "flat", merge=True, nudge=True)
    left, right = polyutils.trace_both(pts, polys)

    pts -= pts.min(0)
    pts *= height / pts.max(0)[1]

    im = Image.new('L', (int(pts.max(0)[0]), height))
    draw = ImageDraw.Draw(im)
    draw.polygon(pts[left, :2].ravel().tolist(), fill=255)
    draw.polygon(pts[right, :2].ravel().tolist(), fill=255)
    return np.array(im) > 0

def _make_flat_cache(subject, xfmname, height=1024):
    from scipy.spatial import cKDTree
    coords = np.vstack(db.surfs.getCoords(subject, xfmname))
    flat, polys, norm = db.surfs.getVTK(subject, "flat", merge=True, nudge=True)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)

    mask = _gen_flat_mask(subject, height=height).T
    assert mask.shape[0] == width and mask.shape[1] == height
    flatpos = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)
    kdt = cKDTree(flat[:,:2])
    dist, idx = kdt.query(flatpos.T[mask.ravel()])

    return coords[idx], (width, height), mask


def make(data, subject, xfmname, recache=False, height=1024, **kwargs):
    cacheform = db.surfs.getFiles(subject)['flatcache']
    cachefile = cacheform.format(xfmname=xfmname, height=height, date="*")
    #pull a list of candidate cache files
    files = glob.glob(cachefile)
    if len(files) < 1 or recache:
        #if recaching, delete all existing files
        for f in files:
            os.unlink(f)
        print "Generating a flatmap cache"
        #pull points and transform from database
        coords, size, mask = _make_flat_cache(subject, xfmname, height=height)
        #save them into the proper file
        date = time.strftime("%Y%m%d")
        cachename = cacheform.format(xfmname=xfmname, height=height, date=date)
        cPickle.dump((coords, size, mask), open(cachefile, "w"), 2)
    else:
        coords, size, mask = cPickle.load(open(files[0]))

    idx = np.ravel_multi_index(coords.T, data.T.shape, mode='clip')
    img = np.nan*np.ones(size, dtype=data.dtype)
    img[mask] = data.T.ravel()[idx]
    return img.reshape(size).T[::-1]

def overlay_rois(im, subject, name=None, height=1024, labels=True, **kwargs):
    from matplotlib.pylab import imsave
    fp = cStringIO.StringIO()
    imsave(fp, im, **kwargs)
    fp.seek(0)
    img = binascii.b2a_base64(fp.read())
    rois = utils.get_roipack(subject)
    return rois.get_texture(height, background=img, name=name, labels=labels)

def make_png(data, subject, xfmname, name=None, with_rois=True, recache=False, height=1024, **kwargs):
    import Image
    im = make(data, subject, xfmname, recache=recache, height=height)

    if with_rois:
        return overlay_rois(im, subject, name=name, height=height, **kwargs)

    if name is None:
        fp = cStringIO.StringIO()
        imsave(fp, im, **kwargs)
        fp.seek(0)
        return fp

    imsave(name, im, **kwargs)


def show(data, subject, xfmname, recache=False, height=1024, with_rois=True, **kwargs):
    from matplotlib.pylab import imshow, imread, axis
    im = make(data, subject, xfmname, recache=recache, height=height)
    if with_rois:
        im = imread(overlay_rois(im, subject, height=height, **kwargs))
    ax = imshow(im, **kwargs)
    ax.axes.set_aspect('equal')
    axis('off')
    return ax