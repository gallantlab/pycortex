from collections import OrderedDict
import numpy as np
from scipy.spatial import distance, Delaunay

class Surface(object):
    def __init__(self, pts, polys):
        self.pts = pts
        self.polys = polys
        self._connected = None

    @property
    def connected(self):
        if self._connected is None:
            self._connected = [set([]) for _ in range(len(self.pts))]
            for i, poly in enumerate(self.polys):
                for p in poly:
                    self._connected[p].add(i)

            self._connected = [list(i) for i in self._connected]
        return self._connected

    @property
    def normals(self):
        fnorms = np.zeros((len(self.polys),3))
        for i, face in enumerate(self.polys):
            x, y, z = self.pts[face]
            fnorms[i] = np.cross(y-x, z-x)

        vnorms = np.zeros((len(self.pts),3))
        for i in range(len(self.pts)):
            vnorms[i] = fnorms[self.connected[i]].mean(0)

        return vnorms

    def extract_chunk(self, nfaces=100, seed=None, auxpts=None):
        '''Extract a chunk of the surface using breadth first search, for testing purposes'''
        node = seed
        if seed is None:
            node = np.random.randint(len(self.pts))

        ptmap = dict()
        queue = [node]
        faces = set()
        visited = set([node])
        while len(faces) < nfaces and len(queue) > 0:
            node = queue.pop(0)
            for face in self.connected[node]:
                if face not in faces:
                    faces.add(face)
                    for pt in self.polys[face]:
                        if pt not in visited:
                            visited.add(pt)
                            queue.append(pt)

        pts, aux, polys = [], [], []
        for face in faces:
            for pt in self.polys[face]:
                if pt not in ptmap:
                    ptmap[pt] = len(pts)
                    pts.append(self.pts[pt])
                    if auxpts is not None:
                        aux.append(auxpts[pt])
            polys.append([ptmap[p] for p in self.polys[face]])

        if auxpts is not None:
            return np.array(pts), np.array(aux), np.array(polys)

        return np.array(pts), np.array(polys)

    def smooth(self, values, neighborhood=3, smooth=8):
        if len(values) != len(self.pts):
            raise ValueError('Each point must have a single value')
            
        def getpts(pt, n):
            if pt in self.connected:
                for p in self.connected[pt]:
                    if n == 0:
                        yield p
                    else:
                        for q in getpts(p, n-1):
                            yield q
    
        output = np.zeros(len(scalars))
        for i, val in enumerate(scalars):
            neighbors = list(set(getpts(i, neighborhood)))
            if len(neighbors) > 0:
                g = np.exp(-((scalars[neighbors] - val)**2) / (2*smooth**2))
                output[i] = (g * scalars[neighbors]).mean()
            
        return output

    def polyhedra(self, wm):
        '''Iterates through the polyhedra that make up the closest volume to a certain vertex'''
        for p, faces in enumerate(self.connected):
            pts, polys = _ptset(), _quadset()
            if len(faces) > 0:
                poly = np.roll(self.polys[faces[0]], -np.nonzero(self.polys[faces[0]] == p)[0][0])
                assert pts[wm[p]] == 0
                assert pts[self.pts[p]] == 1
                pts[wm[poly[[0, 1]]].mean(0)]
                pts[self.pts[poly[[0, 1]]].mean(0)]

                for face in faces:
                    poly = np.roll(self.polys[face], -np.nonzero(self.polys[face] == p)[0][0])
                    a = pts[wm[poly].mean(0)]
                    b = pts[self.pts[poly].mean(0)]
                    c = pts[wm[poly[[0, 2]]].mean(0)]
                    d = pts[self.pts[poly[[0, 2]]].mean(0)]
                    e = pts[wm[poly[[0, 1]]].mean(0)]
                    f = pts[self.pts[poly[[0, 1]]].mean(0)]

                    polys((0, c, a, e))
                    polys((1, f, b, d))
                    polys((1, d, c, 0))
                    polys((1, 0, e, f))
                    polys((f, e, a, b))
                    polys((d, b, a, c))

            yield pts.points, np.array(list(polys.triangles))

    def polyconvex(self, wm):
        try:
            import progressbar as pb
            progress = pb.ProgressBar(maxval=len(self.connected))
            progress.start()
        except ImportError:
            pass

        for p, faces in enumerate(self.connected):
            polys = self.polys[faces]
            x, y = np.nonzero(polys == p)
            x = np.tile(x, [3, 1]).T
            y = np.vstack([y, (y+1)%3, (y+2)%3]).T
            polys = polys[x, y]
            mid = self.pts[polys].mean(1)
            left = self.pts[polys[:,[0,2]]].mean(1)
            right = self.pts[polys[:,[0,1]]].mean(1)
            wmid = wm[polys].mean(1)
            wleft = wm[polys[:,[0,2]]].mean(1)
            wright = wm[polys[:,[0,1]]].mean(1)
            top = np.vstack([mid, left, right])
            bot = np.vstack([wmid, wleft, wright])
            #remove duplicates
            top = top[(distance.cdist(top, top) == 0).sum(0) == 1]
            bot = bot[(distance.cdist(bot, bot) == 0).sum(0) == 1]
            try:
                progress.update(p+1)
            except NameError:
                pass
            yield np.vstack([top, bot, self.pts[p], wm[p]])
        try:
            progress.finish()
        except NameError:
            pass


class _ptset(object):
    def __init__(self):
        self.idx = OrderedDict()
    def __getitem__(self, idx):
        idx = tuple(idx)
        if idx not in self.idx:
            self.idx[idx] = len(self.idx)
        return self.idx[idx]

    @property
    def points(self):
        return np.array(list(self.idx.keys()))

class _quadset(object):
    def __init__(self):
        self.polys = dict()

    def __call__(self, quad):
        idx = tuple(sorted(quad))
        if idx in self.polys:
            del self.polys[idx]
        else:
            self.polys[idx] = quad

    @property
    def triangles(self):
        for quad in list(self.polys.values()):
            yield quad[:3]
            yield [quad[0], quad[2], quad[3]]

class Distortion(object):
    def __init__(self, flat, ref, polys):
        self.flat = flat
        self.ref = ref
        self.polys = polys

    @property
    def areal(self):
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
        vertratios /= np.bincount(self.polys.ravel())
        vertratios = np.nan_to_num(vertratios)
        vertratios[vertratios==0] = 1
        return vertratios

    @property
    def metric(self):
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
        ref_dists = [np.sqrt(((self.ref[G.neighbors(ii)] - self.ref[ii])**2).sum(1))
                     for ii in selverts]
        flat_dists = [np.sqrt(((self.flat[G.neighbors(ii)] - self.flat[ii])**2).sum(1))
                      for ii in selverts]
        msdists = np.array([(fl-fi).mean() for fi,fl in zip(ref_dists, flat_dists)])
        alldists = np.zeros((len(self.ref),))
        alldists[selverts] = msdists
        return alldists

def tetra_vol(pts):
    '''Volume of a tetrahedron'''
    tetra = pts[1:] - pts[0]
    return np.abs(np.dot(tetra[0], np.cross(tetra[1], tetra[2]))) / 6

def brick_vol(pts):
    '''Volume of a triangular prism'''
    return tetra_vol(pts[[0, 1, 2, 4]]) + tetra_vol(pts[[0, 2, 3, 4]]) + tetra_vol(pts[[2, 3, 4, 5]])

def face_volume(pts1, pts2, polys):
    '''Volume of each face in a polyhedron sheet'''
    vols = np.zeros((len(polys),))
    for i, face in enumerate(polys):
        vols[i] = brick_vol(np.append(pts1[face], pts2[face], axis=0))
        if i % 1000 == 0:
            print(i)
    return vols

def decimate(pts, polys):
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    dec = tvtk.DecimatePro(input=pd)
    dec.set(preserve_topology=True, splitting=False, boundary_vertex_deletion=False, target_reduction=1.0)
    dec.update()
    return dec.output.points.to_array(), dec.output.polys.to_array().reshape(-1, 4)[:,1:]

def curvature(pts, polys):
    '''Computes mean curvature using VTK'''
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    curv = tvtk.Curvatures(input=pd, curvature_type="mean")
    curv.update()
    return curv.output.point_data.scalars.to_array()

def inside_convex_poly(pts):
    """Returns a function that checks if inputs are inside the convex hull of polyhedron defined by pts

    Alternative method to check is to get faces of the convex hull, then check if each normal is pointed away from each point.
    As it turns out, this is vastly slower than using qhull's find_simplex, even though the simplex is not needed.
    """
    tri = Delaunay(pts)
    return lambda x: tri.find_simplex(x) != -1

def make_cube(center=(.5, .5, .5), size=1):
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=float)
    pts -= (.5, .5, .5)
    polys = np.array([(0, 2, 3), (0, 3, 1), (0, 1, 4), (1, 5, 4),
                      (1, 3, 5), (3, 7, 5), (2, 7, 3), (2, 6, 7),
                      (0, 6, 2), (0, 4, 6), (4, 7, 6), (4, 5, 7)], dtype=np.uint32)
    return pts * size + center, polys

def boundary_edges(polys):
    '''Returns the edges that are on the boundary of a mesh, as defined by belonging to only 1 face'''
    edges = dict()
    for i, poly in enumerate(np.sort(polys)):
        for a, b in [(0,1), (1,2), (0, 2)]:
            key = poly[a], poly[b]
            if key not in edges:
                edges[key] = []
            edges[key].append(i)

    epts = []
    for edge, faces in edges.items():
        if len(faces) == 1:
            epts.append(edge)

    return np.array(epts)

def trace_poly(edges):
    '''Given a disjoint set of edges, yield complete linked polygons'''
    idx = dict((i, set([])) for i in np.unique(edges))
    for i, (x, y) in enumerate(edges):
        idx[x].add(i)
        idx[y].add(i)

    eset = set(range(len(edges)))
    while len(eset) > 0:
        eidx = eset.pop()
        poly = list(edges[eidx])
        stack = set([eidx])
        while poly[-1] != poly[0] or len(poly) == 1:
            next = list(idx[poly[-1]] - stack)[0]
            eset.remove(next)
            stack.add(next)
            if edges[next][0] == poly[-1]:
                poly.append(edges[next][1])
            elif edges[next][1] == poly[-1]:
                poly.append(edges[next][0])
            else:
                raise Exception

        yield poly

def rasterize(poly, shape=(256, 256)):
    #ImageDraw sucks at its job, so we'll use imagemagick to do rasterization
    import subprocess as sp
    import cStringIO
    import shlex
    import Image

    polygon = " ".join(["%0.3f,%0.3f"%tuple(p[::-1]) for p in np.array(poly)-(.5, .5)])
    cmd = 'convert -size %dx%d xc:black -fill white -stroke none -draw "polygon %s" PNG32:-'%(shape[0], shape[1], polygon)
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    png = cStringIO.StringIO(proc.communicate()[0])
    im = Image.open(png)

    # For PNG8:
    # mode, palette = im.palette.getdata()
    # lut = np.fromstring(palette, dtype=np.uint8).reshape(-1, 3)
    # if (lut == 255).any():
    #     white = np.nonzero((lut == 255).all(1))[0][0]
    #     return np.array(im) == white
    # return np.zeros(shape, dtype=bool)
    return (np.array(im)[:,:,0] > 128).T

def voxelize(pts, polys, shape=(256, 256, 256), center=(128, 128, 128), mp=True):
    from tvtk.api import tvtk
    import Image
    import ImageDraw
    
    pd = tvtk.PolyData(points=pts + center + (0, 0, 0), polys=polys)
    plane = tvtk.Planes(normals=[(0,0,1)], points=[(0,0,.5)])
    clip = tvtk.ClipPolyData(clip_function=plane, input=pd)
    feats = tvtk.FeatureEdges(
        manifold_edges=False, 
        non_manifold_edges=False, 
        feature_edges=False,
        boundary_edges=True,
        input=clip.output)

    def func(i):
        plane.points = [(0,0,i+.5)]
        feats.update()
        vox = np.zeros(shape[:2][::-1], np.uint8)
        if feats.output.number_of_lines > 0:
            epts = feats.output.points.to_array()
            edges = feats.output.lines.to_array().reshape(-1, 3)[:,1:]
            for poly in trace_poly(edges):
                vox += rasterize(epts[poly][:,:2], shape=shape[:2][::-1])
        return vox

    if mp:
        from . import mp
        layers = mp.map(func, range(shape[2]))
    else:
        layers = map(func, range(shape[2]))

    return np.array(layers).T
