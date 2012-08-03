import os
import time
import json
import glob
import numpy as np
from scipy.interpolate import interp1d

from webgl_view import show as webshow
from quickflat import show as quickflat

import db

cwd = os.path.split(os.path.abspath(__file__))[0]

def _tcoords(subject):
    left, right = db.surfs.getVTK(subject, "flat", hemisphere="both", nudge=True)
    fpts = np.vstack([left[0], right[0]])
    fmin = fpts.min(0)
    fpts -= fmin
    fmax = fpts.max(0)
    
    allpts = []
    for pts, polys, norms in [left, right]:
        pts -= fmin
        pts /= fmax
        allpts.append(pts[:,:2])
    return allpts

def _get_surf_interp(subject, types=('inflated',)):
    types = ("fiducial",) + types
    pts = []
    for t in types:
        ptpolys = db.surfs.getVTK(subject, t, nudge=True)
        pts.append([p[0] for p in ptpolys])

    left, right = db.surfs.getVTK(subject, "flat", nudge=False)
    pts.append([left[0], right[0]])
    flatpolys = [p[1] for p in [left, right]]

    fidleft, fidright = db.surfs.getVTK(subject, "fiducial", nudge=True)
    fidpolys = [p[1] for p in [fidleft, fidright]]

    flatmin = 0
    for p in pts[-1]:
        flatpts = np.zeros_like(p)
        flatpts[:,[1,2]] = p[:,:2]
        #flatpts[:,0] = lt.min(0)[1]
        p[:] = flatpts
        flatmin += p[:,1].min()
    #We have to flip the left hemisphere to make it expand correctly
    pts[-1][0][:,1] = -pts[-1][0][:,1]
    #We also have to put them equally far back for pivot to line up correctly
    flatmin /= 2.
    for p in pts[-1]:
        p[:,1] -= p[:,1].min()
        p[:,1] += flatmin

    interp = [interp1d(np.linspace(0,1,len(p)), p, axis=0) for p in zip(*pts)]
    ## Store the name of each "stop" in the interpolator
    for i in interp:
        i.stops = list(types)+["flat"]

    return interp, flatpolys, fidpolys

def get_mixer_args(subject, xfmname, types=('inflated',)):
    coords = db.surfs.getCoords(subject, xfmname)
    interp, flatpolys, fidpolys = _get_surf_interp(subject, types)
    
    overlay = db.surfs.getFiles(subject)['rois']
    if not os.path.exists(overlay):
        #Can't find the roi overlay, create a new one!
        ptpolys = db.surfs.getVTK(subject, "flat", hemisphere="both")
        pts = np.vstack(ptpolys[0][0][:,:2], ptpolys[0][1][:,:2])
        size = pts.max(0) - pts.min(0)
        aspect = size[0] / size[-1]
        with open(overlay, "w") as xml:
            xmlbase = open(os.path.join(cwd, "svgbase.xml")).read()
            xml.write(xmlbase.format(width=aspect * 1024, height=1024))

    return dict(points=interp, flatpolys=flatpolys, fidpolys=fidpolys, coords=coords,
                tcoords=_tcoords(subject), nstops=len(types)+2, svgfile=overlay)

def show(data, subject, xfm, types=('inflated',)):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    kwargs = get_mixer_args(subject, xfm, types)

    if hasattr(data, "get_affine"):
        #this is a nibabel file -- it has the nifti headers intact!
        if isinstance(xfm, str):
            kwargs['coords'] = db.surfs.getCoords(subject, xfm, hemisphere=hemisphere, magnet=data.get_affine())
        data = data.get_data()
    elif isinstance(xfm, np.ndarray):
        ones = np.ones(len(interp[0](0)))
        coords = [np.dot(xfm, np.hstack([i(0), ones]).T)[:3].T for i in interp ]
        kwargs['coords'] = [ c.round().astype(np.uint32) for c in coords ]

    kwargs['data'] = data

    import mixer
    m = mixer.Mixer(**kwargs)
    m.edit_traits()
    return m

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display epi data on various surfaces, \
        allowing you to interpolate between the surfaces")
    parser.add_argument("epi", type=str)
    parser.add_argument("--transform", "-T", type=str)
    parser.add_argument("--surfaces", nargs="*")
