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

def curvature(outfile, subject, smooth=20, **kwargs):
    curvs = []
    for pts, polys in db.get_surf(subject, "fiducial"):
        surf = polyutils.Surface(pts, polys)
        curv = surf.smooth(surf.mean_curvature(), smooth)
        curvs.append(curv)
    np.savez(outfile, left=curvs[0], right=curvs[1])

def distortion(outfile, subject, type='areal', smooth=20):
    """Computes distortion of flatmap relative to fiducial surface. Several different
    types of distortion are available:
    
    'areal': computes the areal distortion for each triangle in the flatmap, defined as the
    log ratio of the area in the fiducial mesh to the area in the flat mesh. Returns
    a per-vertex value that is the average of the neighboring triangles.
    See: http://brainvis.wustl.edu/wiki/index.php/Caret:Operations/Morphing
    
    'metric': computes the linear distortion for each vertex in the flatmap, defined as
    the mean squared difference between distances in the fiducial map and distances in
    the flatmap, for each pair of neighboring vertices. See Fishl, Sereno, and Dale, 1999.
    """
    distortions = []
    for hem in ["lh", "rh"]:
        fidvert, fidtri = db.get_surf(subject, "fiducial", hem)
        flatvert, flattri = db.get_surf(subject, "flat", hem)
        surf = polyutils.Surface(fidvert, fidtri)

        dist = getattr(polyutils.Distortion(flatvert, fidvert, flattri), type)
        smdist = surf.smooth(dist, smooth)
        distortions.append(smdist)

    np.savez(outfile, left=distortions[0], right=distortions[1])

def thickness(outfile, subject):
    pl, pr = db.get_surf(subject, "pia")
    wl, wr = db.get_surf(subject, "wm")
    left = np.sqrt(((pl[0] - wl[0])**2).sum(1))
    right = np.sqrt(((pr[0] - wr[0])**2).sum(1))
    np.savez(outfile, left=left, right=right)

def tissots_indicatrix(outfile, sub, radius=10, spacing=50, maxfails=100): 
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

    np.savez(outfile, left=tissots[0], right=tissots[1], centers=allcenters)

def flat_border(outfile, subject):
    flatpts, flatpolys = db.get_surf(subject, "flat", merge=True, nudge=True)
    flatpolyset = set(map(tuple, flatpolys))
    
    fidpts, fidpolys = db.get_surf(subject, "fiducial", merge=True, nudge=True)
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
    
    np.savez(outfile, lines=lines, ismwalls=ismwalls)
