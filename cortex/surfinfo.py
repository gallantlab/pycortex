"""
Contains functions for computing various surface properties. Mostly wrappers
for functions in `polyutils.Surface` and `polyutils.Distortion`.
"""

import os
import shlex
import shutil
import tempfile
import subprocess as sp

import numpy as np

from . import utils
from . import polyutils
from .database import db
from .xfm import Transform

def curvature(outfile, subject, smooth=20):
    """
    Compute smoothed mean curvature of the fiducial surface for the given 
    subject and save it to `outfile`.

    Parameters
    ----------
    outfile : str
        Path where the curvature map will be saved as an npz file.
    subject : str
        Subject in the pycortex database for whom curvature will be computed.
    smooth : float, optional
        Amount of smoothing to apply to the curvature map. Default 20.
    """
    curvs = []
    for pts, polys in db.get_surf(subject, "fiducial"):
        surf = polyutils.Surface(pts, polys)
        curv = surf.smooth(surf.mean_curvature(), smooth)
        curvs.append(curv)
    np.savez(outfile, left=curvs[0], right=curvs[1])

def distortion(outfile, subject, dist_type='areal', smooth=20):
    """
    Compute distortion of flatmap relative to fiducial surface and save it
    at `outfile`. Several different types of distortion are available:
    
    'areal': computes the areal distortion for each triangle in the flatmap, defined as the
    log ratio of the area in the fiducial mesh to the area in the flat mesh. Returns
    a per-vertex value that is the average of the neighboring triangles.
    See: http://brainvis.wustl.edu/wiki/index.php/Caret:Operations/Morphing
    
    'metric': computes the linear distortion for each vertex in the flatmap, defined as
    the mean squared difference between distances in the fiducial map and distances in
    the flatmap, for each pair of neighboring vertices. See Fishl, Sereno, and Dale, 1999.

    Parameters
    ----------
    outfile : str
        Path where the distortion map will be saved as an npz file.
    subject : str
        Subject in the pycortex database for whom distortion will be computed.
    dist_type : ['areal', 'metric'], optional
        Type of distortion to compute. Default 'areal'.
    smooth : float, optional
        Amount of smoothing to apply to the distortion map before returning.
        Default 20.
    """
    distortions = []
    for hem in ["lh", "rh"]:
        fidvert, fidtri = db.get_surf(subject, "fiducial", hem)
        flatvert, flattri = db.get_surf(subject, "flat", hem)
        surf = polyutils.Surface(fidvert, fidtri)

        dist = getattr(polyutils.Distortion(flatvert, fidvert, flattri), dist_type)
        smdist = surf.smooth(dist, smooth)
        distortions.append(smdist)

    np.savez(outfile, left=distortions[0], right=distortions[1])

def thickness(outfile, subject):
    """
    Compute cortical thickness as the distance between corresponding pial and 
    white matter vertices for the given subject. Note that this is slightly
    different than the method used by Freesurfer, and will yield ever-so-slightly
    different results.

    Parameters
    ----------
    outfile : str
        Path where the thickness map will be saved.
    subject : str
        Subject in the pycortex database for whom cortical thickness will be 
        computed.
    """
    pl, pr = db.get_surf(subject, "pia")
    wl, wr = db.get_surf(subject, "wm")
    left = np.sqrt(((pl[0] - wl[0])**2).sum(1))
    right = np.sqrt(((pr[0] - wr[0])**2).sum(1))
    np.savez(outfile, left=left, right=right)

def tissots_indicatrix(outfile, sub, radius=10, spacing=50):
    """
    Compute a Tissot's indicatrix for the given subject and save the result to
    a file. This involves randomly filling in discs of fixed geodesic radius
    on the fiducial surface.

    See https://en.wikipedia.org/wiki/Tissot's_indicatrix for more info.

    Parameters
    ----------
    outfile : str
        Path where the indicatrix map will be saved.
    sub : str
        Subject in the pycortex database for whom the indicatrix will be 
        computed.
    radius : float, optional
        The geodesic radius of each disc in mm. Default 10.
    spacing : float, optional
        The minimum distance between disc centers in mm. Default 50.
    """
    tissots = []
    allcenters = []
    for hem in ["lh", "rh"]:
        fidpts, fidpolys = db.get_surf(sub, "fiducial", hem)
        #G = make_surface_graph(fidtri)
        surf = polyutils.Surface(fidpts, fidpolys)
        nvert = fidpts.shape[0]
        tissot_array = np.zeros((nvert,))

        centers = [np.random.randint(nvert)]
        cdists = [surf.geodesic_distance(centers)]
        while True:
            ## Find possible vertices
            mcdist = np.vstack(cdists).min(0)
            possverts = np.nonzero(mcdist > spacing)[0]
            #possverts = np.nonzero(surf.geodesic_distance(centers) > spacing)[0]
            if not len(possverts):
                break
            ## Pick random vertex
            centervert = possverts[np.random.randint(len(possverts))]
            centers.append(centervert)
            print("Adding vertex %d.." % centervert)
            dists = surf.geodesic_distance([centervert])
            cdists.append(dists)

            ## Find appropriate set of vertices
            selverts = dists < radius
            tissot_array[selverts] = 1

        tissots.append(tissot_array)
        allcenters.append(np.array(centers))
    
    # make an array of objects to allow different lengths for each hemisphere
    allcenters = np.array(allcenters, dtype="object")
    np.savez(outfile, left=tissots[0], right=tissots[1], centers=allcenters)

def flat_border(outfile, subject):
    flatpts, flatpolys = db.get_surf(subject, "flat", merge=True, nudge=True)
    flatpolyset = set([tuple(x) for x in flatpolys])
    
    fidpts, fidpolys = db.get_surf(subject, "fiducial", merge=True, nudge=True)
    fidpolyset = set([tuple(x) for x in fidpolys])
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
    badverts = np.array([v for v,d in fog.degree().items() if d<2])
    g.remove_nodes_from(badverts)
    fog.remove_nodes_from(badverts)
    mwallset = set.union(*(set(g[v]) for v in fog.nodes())) & set(allbounds)
    #cutset = (set(g.nodes()) - mwallset) & set(allbounds)

    mwallbounds = [np.isin(b, mwallset) for b in bounds]
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
    
    np.savez(outfile, lines=lines, ismwalls=ismwalls)

def _relax_hemisphere(args):
    """Relax one hemisphere onto its flatmap. Top level so it can be pickled.

    Subjects with no flat surface get zeros rather than an error.
    """
    subject, hemi, poisson_ratio = args
    wm, polys = db.get_surf(subject, "wm", hemi)
    pia, _ = db.get_surf(subject, "pia", hemi)
    try:
        flat, flatpolys = db.get_surf(subject, "flat", hemi)
    except IOError:
        return np.zeros_like(wm)

    slab = polyutils.FlatSlab(flat, wm, pia, flatpolys,
                              poisson_ratio=poisson_ratio)
    offsets = slab.relaxed
    info = slab.info
    print("%s %s: %d elements, energy %.4g -> %.4g in %d iterations, "
          "volume %+.2f%%" % (subject, hemi, info['n_tets'],
                              info['energy_initial'], info['energy_final'],
                              info['iterations'],
                              100 * (info['volume_relaxed']
                                     / info['volume_folded'] - 1)))
    return offsets

def bumpy_flatmap(outfile, subject, poisson_ratio=0.45, parallel=True):
    """
    Relax the cortical slab onto the flatmap and save the resulting pial offsets.

    The white matter surface is pinned to the flatmap and the pial surface is
    allowed to settle as an elastic solid, so the flatmap gets relief that
    reflects cortical thickness and folding: gyri, which flattening compresses,
    end up thicker, and sulci, which it stretches, thinner. See
    `cortex.polyutils.FlatSlab` for the model and for why Poisson's ratio is the
    only material parameter that matters.

    This is the expensive one -- it is a nonlinear optimisation over every pial
    vertex, so it takes minutes rather than seconds. It is generated
    automatically when a flatmap is imported (see `cortex.freesurfer.import_flat`)
    so that the viewer does not have to wait for it.

    Subjects with no flat surface get an array of zeros rather than an error.

    Parameters
    ----------
    outfile : str
        Path where the offsets will be saved as an npz file.
    subject : str
        Subject in the pycortex database for whom the flatmap will be relaxed.
    poisson_ratio : float, optional
        How strictly the tissue preserves volume, in [0, 0.5). Default 0.45.
    parallel : bool, optional
        Relax the two hemispheres in separate processes. Default True.

    Notes
    -----
    The arrays are stored under ``bump_left`` and ``bump_right`` rather than
    ``left`` and ``right`` on purpose: `cortex.database.Database.get_surfinfo`
    turns a file with ``left`` and ``right`` keys into a `Vertex` by
    concatenating them, which assumes one value per vertex and would quietly
    mangle these three-component offsets. With these names it hands back the npz
    itself.
    """
    args = [(subject, hemi, poisson_ratio) for hemi in ["lh", "rh"]]
    if parallel:
        # The hemispheres are completely independent, so this halves the wall
        # clock. Each one peaks at not quite a gigabyte; pass parallel=False on a
        # machine where running two at once would be tight.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as pool:
            offsets = list(pool.map(_relax_hemisphere, args))
    else:
        offsets = [_relax_hemisphere(a) for a in args]

    # Compressed, and single precision: these are millimetre offsets that end
    # up in a float32 vertex attribute anyway, so the extra digits are only
    # taking up room in the filestore.
    np.savez_compressed(outfile,
                        bump_left=offsets[0].astype(np.float32),
                        bump_right=offsets[1].astype(np.float32))

def equivolume_areas(outfile, subject, smooth=1.0):
    """
    Compute smoothed vertex areas on the white matter and pial surfaces.

    These are what the webgl viewer's equivolume depth sampling needs in order to
    turn a requested volume fraction through the cortical sheet into a position
    between the two surfaces. They used to be recomputed in javascript on every
    viewer load with a uniform "umbrella" smoothing; computing them here instead
    means they are cached, and lets them be smoothed with the cotangent-weighted
    operator in `cortex.polyutils.Surface.smooth`, which respects the varying
    size of the triangles rather than treating every neighbour equally.

    Parameters
    ----------
    outfile : str
        Path where the areas will be saved as an npz file.
    subject : str
        Subject in the pycortex database for whom the areas will be computed.
    smooth : float, optional
        Amount of smoothing to apply. Default 1.0. Pass 0 for the raw
        barycentric vertex areas.

    Notes
    -----
    Stored under ``wm_left`` / ``wm_right`` / ``pia_left`` / ``pia_right``; see
    the note in `bumpy_flatmap` for why ``left`` and ``right`` are avoided.

    Changing the smoothing operator does move the depths the viewer samples at,
    slightly: against the five umbrella iterations the javascript used, the
    depth equivolume sampling picks for a requested fraction of 0.5 shifts by
    about 0.01 of the cortical thickness at the median on S1. The tail is larger,
    but it is concentrated where the two areas are nearly equal and the depth is
    poorly determined either way.
    """
    areas = dict()
    for hemi, side in zip(["lh", "rh"], ["left", "right"]):
        for name in ["wm", "pia"]:
            pts, polys = db.get_surf(subject, name, hemi)
            surf = polyutils.Surface(pts, polys)
            # The lumped mass matrix of the Laplace-Beltrami operator is exactly
            # the barycentric vertex area, i.e. a third of each incident face.
            _, vertex_area, _, _ = surf.laplace_operator
            areas["%s_%s" % (name, side)] = surf.smooth(vertex_area, smooth)

    np.savez(outfile, **areas)
