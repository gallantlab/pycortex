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
    pts = pts.copy()[:,:2]
    pts -= pts.min(0)
    pts *= height / pts.max(0)[1]
    im = Image.new('L', pts.max(0), 0)
    draw = ImageDraw.Draw(im)

    left, right = polyutils.trace_both(pts, polys)
    draw.polygon(pts[left], outline=None, fill=255)
    draw.polygon(pts[right], outline=None, fill=255)
    
    del draw
    return np.array(im) > 0

def _make_flat_cache(subject, xfmname, height=1024):
    from scipy.interpolate import griddata
    coords = np.vstack(db.surfs.getCoords(subject, xfmname))
    flat, polys, norm = db.surfs.getVTK(subject, "flat", merge=True, nudge=True)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = aspect * 1024

    flatpos = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)
    pcoords = griddata(flat[:,:2], coords, flatpos.T, method="nearest")
    return pcoords, (width, height)


def make(data, subject, xfmname, recache=False, height=1024):
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
        coords, size = _make_flat_cache(subject, xfmname, height=height)
        mask = _gen_flat_mask(subject, height=height).T
        #save them into the proper file
        date = time.strftime("%Y%m%d")
        cachename = cacheform.format(xfmname=xfmname, height=height, date=date)
        cPickle.dump((coords, size, mask), open(cachefile, "w"), 2)
    else:
        coords, size, mask = cPickle.load(open(files[0]))

    ravelpos = coords[:,0]*data.shape[1]*data.shape[0]
    ravelpos += coords[:,1]*data.shape[0] + coords[:,2]
    validpos = ravelpos[mask.ravel()].astype(int)
    img = np.nan*np.ones_like(ravelpos)
    img[mask.ravel()] = data.T.ravel()[validpos]
    return img.reshape(size).T[::-1], mask

def make_png(data, subject, xfmname, name=None, recache=False, height=1024, with_rois=False, **kwargs):
    import Image
    from matplotlib.pylab import imsave, imread

    if with_rois:
        pngdat, mask = makepng(data, subject, xfmname, recache=recache, height=height)
        rois = utils.get_roipack(subject)
        img = rois.get_texture(height, background=pngdat)
        im = imread(img)
        im[~mask.T[::-1], -1] = 0
        im = Image.fromarray((im*255).astype(np.uint8))
        im.save(name)

    else:
        im, mask = make(data, subject, xfmname, recache=recache, height=height)

        if name is None:
            fname = cStringIO.StringIO()
            imsave(fname, im, **kwargs)
            fname.seek(0)
            return binascii.b2a_base64(fname.read()), mask

        imsave(name, im, **kwargs)

def show(data, subject, xfmname, recache=False, height=1024, **kwargs):
    from matplotlib.pylab import imshow
    im, mask = make(data, subject, xfmname, recache=recache, height=height)
    imshow(im, **kwargs)