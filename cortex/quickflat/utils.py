"""Makes flattened views of volumetric data on the cortical surface.
"""
import os
import string
import warnings
from functools import reduce

import numpy as np

from .. import dataset, utils
from ..database import db
from ..options import config


def make_flatmap_image(braindata, height=1024, recache=False, nanmean=False, **kwargs):
    """Generate flatmap image from volumetric brain data

    This 

    Parameters
    ----------
    braindata : one of: {cortex.Volume, cortex.Vertex, cortex.Dataview)
        Object containing containing data to be plotted, subject (surface identifier), 
        and transform.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. Useful if you've made changes to the alignment.
    nanmean : bool, optional (default = False)
        If True, NaNs in the data will be ignored when averaging across layers.
    kwargs : idk
        idk

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
        raise ValueError("Input data was not the correct dimensionality - please provide 3D Volume or 2D Vertex data")

    if data.dtype != np.uint8:
        # Convert data to float to avoid image artifacts
        data = data.astype(float)
    if data.dtype == np.uint8:
        img = np.zeros(mask.shape+(4,), dtype=np.uint8)
        img[mask] = pixmap * data.reshape(-1, 4)
        img = img.transpose(1,0,2)[::-1]
        # Make img a c-contiguous array or pil will complain when saving it
        if not img.flags["C_CONTIGUOUS"]:
            img = img.copy(order="C")
        return img, extents
    else:
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan*np.ones(mask.shape)).astype(data.dtype)
        mimg = (np.nan*np.ones(badmask.shape)).astype(data.dtype)

        # Pixmap is a (pixels x voxels) sparse non-negative weight matrix where
        # each row sums to 1, so pixmap.dot(vec) gives the mean of vec across
        # cortical thickness.

        # To ignore NaNs (or masked data) in the weighted mean, we normalize by
        # the sum of the non-ignored weights: sum(weights * non-ignored values)
        # / sum(weights on non-ignored values)
        ignored = None

        if not nanmean:  # NaN are not ignored: mean([1., 2., NaN]) = NaN
            averaged_data = pixmap.dot(data.ravel())
            if isinstance(data, np.ma.MaskedArray):  
                ignored = data.ravel().mask  # masked voxels are ignored
    
        else:  # NaN are ignored: mean([1., 2., NaN]) = 1.5
            averaged_data = pixmap.dot(np.nan_to_num(data.ravel()))
            ignored = np.isnan(data.ravel())
            if isinstance(data, np.ma.MaskedArray):
                ignored = ignored.filled()  # masked voxels are also ignored

        if ignored is not None:
            weights_not_ignored = pixmap.dot((~ignored).astype(data.dtype))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                averaged_data = averaged_data / weights_not_ignored

        mimg[badmask] = averaged_data[badmask].astype(mimg.dtype)
        img[mask] = mimg
        img = img.T[::-1]
        # Make img a c-contiguous array or pil will complain when saving it
        if not img.flags["C_CONTIGUOUS"]:
            img = img.copy(order="C")

        return img, extents

def get_flatmask(subject, height=1024, recache=False):
    """
    Parameters
    ----------
    subject : str
        Name of subject in pycortex store
    height : int
        Height in pixels to generate the image
    recache : bool
        Recache the intermediate files? Can resolve some issues but is slower.
    """
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
    """
    
    Parameters
    ----------
    subject : str
        Subject name in pycortex db
    xfmname : str
        Name of transform for subject
    pixelwise : bool
    
    thick : int
    
    sampler : 
    
    recache : bool
        Recache intermediate files? Doing so is slower but can resolve some errors.
    height : int
        Height in pixels of image to generated
    depth : float
    
    Returns
    -------
    """
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

def _return_pixel_pairs(vert_pair_list, x_dict, y_dict):
    """Janky and probably unnecessary"""
    pix_list = []
    vert_pairs_valid = []
    for (vert1, vert2) in vert_pair_list:
        if vert1 in x_dict and vert2 in x_dict:
            pix1 = np.array((x_dict[vert1], y_dict[vert1]))
            pix2 = np.array((x_dict[vert2], y_dict[vert2]))
            pix_list.append(np.array([pix1, pix2]))
            vert_pairs_valid.append((vert1, vert2))
        else:
            #These are vertex pairs not represented in the flatmap. I have found them to belong to the middle brain are that is deleted while creating the flat map.
            pass 
    return np.array(pix_list), np.array(vert_pairs_valid)

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
        labelcolor='label-fill',  # we start label kwargs with "label-"
        labelsize='label-font-size',
        linealpha='stroke-opacity',
        roifill='fill',
        fillcolor='fill',
        fillalpha='fill-opacity',
        dashes='stroke-dasharray'
        #dash_capstyle # ADD ME?
        #dash_joinstyle # ADD ME?
        )  
    svg_style_value_mapping = dict(
        linewidth=lambda x: x,
        lw=lambda x: x,
        linecolor=lambda x: _color2hex(x), 
        lc=lambda x: _color2hex(x), 
        labelcolor=lambda x: _color2hex(x), 
        labelsize=lambda x: x,
        linealpha=lambda x: x,
        roifill=lambda x: _color2hex(x),
        fillcolor=lambda x: _color2hex(x),
        fillalpha=lambda x: x,
        dashes=lambda x: '{}, {}'.format(*x),
        #dash_capstyle # ADD ME?
        #dash_joinstyle # ADD ME?
        )
    out = dict((svg_style_key_mapping[k], svg_style_value_mapping[k](v)) 
               for k,v in kwargs.items() if v is not None)
    return out

def _parse_defaults(section):
    defaults = dict(config.items(section))
    for k in defaults.keys():
        # Convert numbers to floating point numbers
        if defaults[k][0] in string.digits + '.':
            if ',' in defaults[k]:
                defaults[k] = [float(x) for x in defaults[k].split(',')]
            else:
                defaults[k] = float(defaults[k])
        # Convert 'None' to None
        if defaults[k] == 'None':
            defaults[k] = None
        # Special case formatting
        if k=='stroke' or k=='fill':
            defaults[k] = _color2hex(defaults[k])
        elif k=='stroke-dasharray' and isinstance(defaults[k], (list,tuple)):
            defaults[k] = '{}, {}'.format(*defaults[k])
    return defaults

def _get_fig_and_ax(fig):
    """Get figure and current ax. Input can be either a figure or an ax."""
    import matplotlib.pyplot as plt
    if isinstance(fig, plt.Axes):
        ax = fig
        fig = ax.figure
    elif isinstance(fig, plt.Figure):
        ax = fig.gca()
    else:
        raise ValueError("fig should be a matplotlib Figure or Axes instance.")

    return fig, ax

def _get_images(fig):
    """Get all images in a given matplotlib axis"""
    from matplotlib.image import AxesImage
    _, ax = _get_fig_and_ax(fig)
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

    Returns
    -------
    hatchim : RGBA array
        flatmap image with hatches over hatch_data
    """
    dmap, _ = make_flatmap_image(
        hatch_data, height=height, sampler=sampler, recache=recache, nanmean=True
    )
    mask_nans = np.isnan(dmap)
    hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))

    hatchpat = (hx+hy)%(2*hatch_space) < 2
    # Leila code that breaks:
    #hatch_size = [0, 4, 4]
    #hatchpat = (hx + hy + hatch_size[0])%(hatch_size[1] * hatch_space) < hatch_size[2]

    hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
    hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
    hatchim[:, :, 3] *= np.clip(dmap, 0, 1).astype(float)
    # Set nans to alpha = 0. for transparency
    hatchim[mask_nans, 3] = 0.

    return hatchim

def _make_flatmask(subject, height=1024):
    from PIL import Image, ImageDraw

    from .. import polyutils
    pts, polys = db.get_surf(subject, "flat", merge=True, nudge=True)
    left, right = polyutils.trace_poly(polyutils.boundary_edges(polys))

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

    # Get barycentric coordinates
    dl = Delaunay(flat[valid,:2])
    simps = dl.find_simplex(grid.T[mask.ravel()])
    missing = simps == -1
    tfms = dl.transform[simps]
    l1, l2 = (tfms[:,:2].transpose(1,2,0) * (grid.T[mask.ravel()] - tfms[:,2]).T).sum(1)
    l3 = 1 - l1 - l2

    ll = np.vstack([l1, l2, l3])
    ll[:,missing] = 0

    from ..mapper import samplers
    xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
    sampclass = getattr(samplers, sampler)

    # Transform fiducial vertex locations to pixel locations using barycentric xfm
    try:
        pia, polys = db.get_surf(subject, "pia", merge=True, nudge=False)
        wm, polys = db.get_surf(subject, "wm", merge=True, nudge=False)
        piacoords = xfm((pia[valid][dl.simplices][simps] * ll[np.newaxis].T).sum(1))
        wmcoords = xfm((wm[valid][dl.simplices][simps] * ll[np.newaxis].T).sum(1))

        valid_p = np.array([np.all((0 <= piacoords), axis=1),
                            piacoords[:,0] < xfm.shape[2],
                            piacoords[:,1] < xfm.shape[1],
                            piacoords[:,2] < xfm.shape[0]])
        valid_p = np.all(valid_p, axis=0)

        valid_w = np.array([np.all((0 <= wmcoords), axis=1),
                            wmcoords[:,0] < xfm.shape[2],
                            wmcoords[:,1] < xfm.shape[1],
                            wmcoords[:,2] < xfm.shape[0]])
        valid_w = np.all(valid_w, axis=0)

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
        fidcoords = xfm((fid[valid][dl.simplices][simps] * ll[np.newaxis].T).sum(1))

        valid = reduce(np.logical_and,
                       [reduce(np.logical_and, (0 <= fidcoords).T),
                        fidcoords[:, 0] < xfm.shape[2],
                        fidcoords[:, 1] < xfm.shape[1],
                        fidcoords[:, 2] < xfm.shape[0]])

        vidx = np.nonzero(valid)[0]

        i, j, data = sampclass(fidcoords[valid], xfm.shape)
        csrshape = mask.sum(), np.prod(xfm.shape)
        return sparse.csr_matrix((data, (vidx[i], j)), shape=csrshape)
