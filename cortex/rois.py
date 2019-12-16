from functools import reduce
import io
import os
import tempfile
import binascii
import numpy as np

import networkx as nx

from . import db
from .svgoverlay import get_overlay, _make_layer, _find_layer, parser
from lxml import etree
from .dataset import Vertex
from .polyutils import Surface, boundary_edges
from .utils import add_roi
from . import quickflat

class ROIpack(object):
    def __init__(self, subject, roifile):
        self.subject = subject
        self.roifile = roifile
        self.rois = {}
        self.load_roifile()

    def load_roifile(self):
        """Load ROI definitions from self.roifile.
        """
        # Check if file exists
        if not os.path.exists(self.roifile):
            print("ROI file %s doesn't exist.." % self.roifile)
            return

        # Create basic Vertex to avoid expensive initialization..
        empty = Vertex(None, self.subject)
        
        # Load ROIs from file
        if self.roifile.endswith("npz"):
            roidata = np.load(self.roifile)
            for roi in roidata.keys():
                self.rois[roi] = empty.copy(roidata[roi])
            roidata.close()
            
        elif self.roifile.endswith("svg"):
            pts, polys = surfs.getSurf(self.subject, "flat", merge=True, nudge=True)
            npts = len(pts)
            svgroipack = get_overlay(self.subject, self.roifile, pts, polys)
            for name in svgroipack.names:
                roimask = np.zeros((npts,))
                roimask[svgroipack.get_roi(name)] = 1
                self.rois[name] = empty.copy(roimask)
            
        elif self.roifile.endswith("hf5"):
            raise NotImplementedError
        
        else:
            raise ValueError("Don't understand ROI filetype: %s" % self.roifile)

    def to_npz(self, filename):
        """Saves npz file containing ROI masks.
        """
        roidata = dict([(name,vd.data) for name,vd in self.rois.items()])
        np.savez(filename, **roidata)

    def to_svg(self, open_inkscape=False, filename=None):
        """Generate SVG file from vertex ROI masks.
        """
        # Generate temp filename if not provided
        if filename is None:
            filename = tempfile.mktemp(suffix=".svg", prefix=self.subject+"-rois-")
        
        mpts, mpolys = db.get_surf(self.subject, "flat", merge=True, nudge=True)
        svgmpts = mpts[:,:2].copy()
        svgmpts -= svgmpts.min(0)
        svgmpts *= 1024 / svgmpts.max(0)[1]
        svgmpts[:,1] = 1024 - svgmpts[:,1]
        
        npts = len(mpts)
        svgroipack = get_overlay(self.subject, filename, mpts, mpolys)

        # Add default layers
        # Add curvature
        from matplotlib import cm
        #curv = Vertex(np.hstack(get_curvature(self.subject)), self.subject)
        #curv = db.get_surfinfo(self.subject, 'curvature')
        #fp = io.BytesIO()
        #curvim = quickflat.make_png(fp, curv, height=1024, with_rois=False, with_labels=False,
                                    #with_colorbar=False, cmap=cm.gray,recache=True)
        #fp.seek(0)
        #svgroipack.add_roi("curvature", binascii.b2a_base64(fp.read()), add_path=False)

        # Add thickness
        

        # Add ROI boundaries
        svg = etree.parse(svgroipack.svgfile, parser=parser)

        # Find boundary vertices for each ROI
        lsurf, rsurf = [Surface(*pp) for pp in db.get_surf(self.subject, "fiducial")]
        flsurf, frsurf = [Surface(*pp) for pp in db.get_surf(self.subject, "flat")]
        valids = [set(np.unique(flsurf.polys)), set(np.unique(frsurf.polys))]

        # Construct polygon adjacency graph for each surface
        polygraphs = [poly_graph(lsurf), poly_graph(rsurf)]
        for roi in self.rois.keys():
            print("Adding %s.." % roi)
            masks = self.rois[roi].left, self.rois[roi].right
            mmpts = svgmpts[:len(masks[0])], svgmpts[len(masks[0]):]
            roilayer = _make_layer(_find_layer(_find_layer(svg, "rois"),"shapes"), roi)
            for valid, pgraph, surf, mask, mmp in zip(valids, polygraphs,
                                                      [lsurf, rsurf], masks, mmpts):
                if mask.sum() == 0:
                    continue
                
                # Find bounds
                inbound, exbound = get_boundary(surf, np.nonzero(mask)[0])
                
                # Find polys
                allbpolys = np.unique(surf.connected[inbound+exbound].indices)
                selbpolys = surf.polys[allbpolys]
                inpolys = np.in1d(selbpolys, inbound).reshape(selbpolys.shape)
                expolys = np.in1d(selbpolys, exbound).reshape(selbpolys.shape)
                badpolys = np.logical_or(inpolys.all(1), expolys.all(1))
                boundpolys = np.logical_and(np.logical_or(inpolys, expolys).all(1), ~badpolys)

                # Walk around boundary
                boundpolyinds = set(allbpolys[np.nonzero(boundpolys)[0]])

                bgraph = nx.Graph()
                pos = dict()
                for pa in boundpolyinds:
                    for pb in set(pgraph[pa]) & boundpolyinds:
                        edge = pgraph[pa][pb]["verts"]
                        validverts = list(valid & edge)
                        if len(validverts) > 0:
                            pos[edge] = mmp[validverts].mean(0)
                            bgraph.add_edge(*edge)

                cc = nx.cycles.cycle_basis(bgraph)
                if len(cc) == 0:
                    continue
                if len(cc) > 1:
                    # Need to deal with this later: map/reduce calls not python3 compatible
                    edges = reduce(set.symmetric_difference,
                                   [set(map(lambda l:tuple(sorted(l)), zip(c, c[1:]+[c[0]]))) for c in cc])
                    eg = nx.from_edgelist(edges)
                    cycles = nx.cycles.cycle_basis(eg)
                    #longest = np.argmax(map(len, cycles))
                    longest = np.argmax([len(x) for x in  cycles]) # python3 compatible
                    path_order = cycles[longest]
                else:
                    path_order = cc[0]
                path_points = [tuple(pos[frozenset(p)]) for p in zip(path_order[:-1],
                                                                     path_order[1:])]

                # Store poly
                path = "M %f %f L" % tuple(np.nan_to_num(path_points[0]))
                path += ", ".join(["%f %f"%tuple(np.nan_to_num(p)) for p in path_points[1:]])
                path += "Z "

                # Insert into SVG
                svgpath = etree.SubElement(roilayer, "path")
                svgpath.attrib["style"] = "fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opactiy:1"
                svgpath.attrib["d"] = path
                #svgpath.attrib["sodipodi:nodetypes"] = "c" * len(pts)
        
        with open(svgroipack.svgfile, "wb") as xml:
            xml.write(etree.tostring(svg, pretty_print=True))


def poly_graph(surf):
    """NetworkX undirected graph representing polygons of a Surface.
    """
    import networkx as nx
    from collections import defaultdict
    edges = defaultdict(list)
    for ii,(a,b,c) in enumerate(surf.polys):
        edges[frozenset([a,b])].append(ii)
        edges[frozenset([a,c])].append(ii)
        edges[frozenset([b,c])].append(ii)

    #nedges = len(edges)
    #ii,jj = np.vstack(edges.values()).T
    #polymat = sparse.coo_matrix((np.ones((nedges,)), (ii, jj)), shape=[len(self.polys)]*2)
    polygraph = nx.Graph()
    polygraph.add_edges_from(((p[0], p[1], dict(verts=k)) for k,p in edges.items()))
    return polygraph

def get_boundary(surf, vertices, remove_danglers=False):
    """Return interior and exterior boundary sets for `vertices`.
    If `remove_danglers` is True vertices in the internal boundary with
    only one neighbor in the internal boundary will be moved to the external
    boundary.
    """
    if not len(vertices):
        return [], []

    import networkx as nx

    # Use networkx to get external boundary
    external_boundary = set(nx.node_boundary(surf.graph, vertices))

    # Find adjacent vertices to get inner boundary
    internal_boundary = set.union(*[set(surf.graph[v].keys())
                                    for v in external_boundary]).intersection(set(vertices))

    if remove_danglers:
        ingraph = surf.graph.subgraph(internal_boundary)
        danglers = [n for n,d in ingraph.degree().items() if d==1]
        while danglers:
            internal_boundary -= set(danglers)
            external_boundary |= set(danglers)

            ingraph = surf.graph.subgraph(internal_boundary)
            danglers = [n for n,d in ingraph.degree().items() if d<2]

    return list(internal_boundary), list(external_boundary)

