
import numpy as np


class Distortion(object):
    """Used to compute distortion metrics between fiducial and another (e.g. flat)
    surface.

    Parameters
    ----------
    flat : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in flatmap space.
    ref : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in fiducial (reference) space.
    polys : 2D ndarray, shape (total_polys, 3)
        Triangle vertex indices in both `flat` and `ref`.
    """
    def __init__(self, flat, ref, polys):
        self.flat = flat
        self.ref = ref
        self.polys = polys

    @property
    def areal(self):
        """Compute areal distortion of the flatmap.

        Areal distortion is calculated at each triangle as the log2 ratio of
        the triangle area in the flatmap to the area in the reference surface.
        Distortion values are then resampled onto the vertices.

        Thus a value of 0 indicates the areas are equal (no distortion), a 
        value of +1 indicates that the area in the flatmap is 2x the area
        in the reference surface (expansion), and a value of -1 indicates
        that the area in the flatmap is 1/2x the area in the reference
        surface (compression).

        See: http://brainvis.wustl.edu/wiki/index.php/Caret:Operations/Morphing

        Returns
        -------
        vertratios : 1D ndarray, shape (total_verts,)
            Areal distortion at each vertex.
        """
        def area(pts, polys):
            ppts = pts[polys]
            cross = np.cross(ppts[:,1] - ppts[:,0], ppts[:,2] - ppts[:,0])
            return np.sqrt((cross**2).sum(-1))

        refarea = area(self.ref, self.polys)
        flatarea = area(self.flat, self.polys)
        tridists = np.log2(flatarea/refarea)
        
        vertratios = np.zeros((len(self.ref),))
        vertratios[self.polys[:,0]] += tridists
        vertratios[self.polys[:,1]] += tridists
        vertratios[self.polys[:,2]] += tridists
        vertratios /= np.bincount(self.polys.ravel(), minlength=len(self.ref))
        vertratios = np.nan_to_num(vertratios)
        vertratios[vertratios==0] = 1
        return vertratios

    @property
    def metric(self):
        """Compute metric distortion of the flatmap.

        Metric distortion is calculated as the difference in squared distance
        from each vertex to its neighbors between the flatmap and the reference.

        Positive values of metric distortion mean that vertices are farther from
        their neighbors in the flatmap than in the reference surface (expansion),
        etc.

        See: Fishl, Sereno, and Dale, 1999.

        Returns
        -------
        vertdists : 1D ndarray, shape (total_verts,)
            Metric distortion at each vertex.
        """
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

        G = make_surface_graph(self.polys)
        selverts = np.unique(self.polys.ravel())
        ref_dists = [np.sqrt(((self.ref[np.array([x for x in G.neighbors(ii)])] - 
                               self.ref[ii])**2).sum(1)) for ii in selverts]
        flat_dists = [np.sqrt(((self.flat[np.array([x for x in G.neighbors(ii)])] - 
                                self.flat[ii])**2).sum(1)) for ii in selverts]
        msdists = np.array([(f-r).mean() for r,f in zip(ref_dists, flat_dists)])
        alldists = np.zeros((len(self.ref),))
        alldists[selverts] = msdists
        return alldists
