from collections import OrderedDict
import numpy as np
from scipy.spatial import distance, Delaunay
from matplotlib.path import Path
from matplotlib import patches

class Surface(object):
    def __init__(self, pts, polys):
        self.pts = pts
        self.polys = polys
        self.members = [[] for _ in range(len(pts))]
        for i, poly in enumerate(polys):
            for p in poly:
                self.members[p].append(i)

    @property
    def normals(self):
        fnorms = np.zeros((len(self.polys),3))
        for i, face in enumerate(self.polys):
            x, y, z = self.pts[face]
            fnorms[i] = np.cross(y-x, z-x)

        vnorms = np.zeros((len(self.pts),3))
        for i in range(len(self.pts)):
            vnorms[i] = fnorms[self.members[i]].mean(0)

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
            for face in self.members[node]:
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

    def polyhedra(self, wm):
        '''Iterates through the polyhedra that make up the closest volume to a certain vertex'''
        for p, faces in enumerate(self.members):
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
        for p, faces in enumerate(self.members):
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
            yield np.vstack([top, bot, self.pts[p], wm[p]])

    def polyparts(self, wm, idx):
        #polypart = np.array([[0, 1, 2], [0, 3, 1], [3, 4, 1], [4, 5, 1], [5, 2, 1], [0, 2, 5], [0, 5, 3], [3, 5, 4]])
        faces = self.members[idx]
        if len(faces) > 0:
            p0 = self.pts[idx]
            w0 = wm[idx]
            for face in faces:
                poly = np.roll(self.polys[face], -np.nonzero(self.polys[face] == idx)[0][0])
                p1 = self.pts[poly[[0,1]]].mean(0)
                p2 = self.pts[poly].mean(0)
                p3 = self.pts[poly[[0,2]]].mean(0)
                w1 = wm[poly[[0,1]]].mean(0)
                w2 = wm[poly].mean(0)
                w3 = wm[poly[[0,2]]].mean(0)

                tri1 = np.vstack([p0, p1, p2, w0, w1, w2])
                tri2 = np.vstack([p0, p2, p3, w0, w2, w3])
                yield tri1, Delaunay(tri1).convex_hull
                yield tri2, Delaunay(tri2).convex_hull


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

def _tetra_vol(pts):
    tetra = pts[1:] - pts[0]
    return np.abs(np.dot(tetra[0], np.cross(tetra[1], tetra[2]))) / 6

def _brick_vol(pts):
    return _tetra_vol(pts[[0, 1, 2, 4]]) + _tetra_vol(pts[[0, 2, 3, 4]]) + _tetra_vol(pts[[2, 3, 4, 5]])

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

def face_volume(pts1, pts2, polys):
    vols = np.zeros((len(polys),))
    for i, face in enumerate(polys):
        vols[i] = _brick_vol(np.append(pts1[face], pts2[face], axis=0))
        if i % 1000 == 0:
            print(i)
    return vols

def get_connected(polys):
    data = [set([]) for _ in range(polys.max()+1)]
    for i, poly in enumerate(polys):
        for p in poly:
            data[p].add(i)

    return data

def check_cycle(i, polys, ptpoly):
    pts = polys[list(ptpoly[i])]
    cycles = pts[np.nonzero(pts-i)]
    return cycles

def remove_pairs(arr):
    return [p for p in np.unique(arr) if sum(p == arr) != 2]

def trace_edge(seed, pts, polys, ptpoly, flip=False):
    edge = [seed]
    while True:
        cycle = remove_pairs(check_cycle(edge[-1], polys, ptpoly))
        if cycle[0 if not flip else 1] not in edge:
            edge.append(cycle[0 if not flip else 1])
        elif cycle[1 if not flip else 0] not in edge:
            edge.append(cycle[1 if not flip else 0])
        else:
            #both in edges, we've gone around!
            break;
    return edge

def trace_both(pts, polys):
    ptpoly = get_connected(polys)
    left = trace_edge(pts.argmin(0)[0], pts, polys, ptpoly)
    right = trace_edge(pts.argmax(0)[0], pts, polys, ptpoly, flip=True)
    return left, right

def get_dist(pts):
    return distance.squareform(distance.pdist(pts))

def get_closest_nonadj(dist, adjthres=10):
    closest = []
    for i, d in enumerate(dist):
        sort = d.argsort()[:adjthres] - i
        sort = np.array([sort, abs(sort - len(dist))]).min(0)
        find = sort[sort > adjthres]
        if len(find) > 0:
            closest.append((i, find[0]+i))

    return np.array([x for x in closest if (x[1], x[0]) not in closest])

def make_rot(close, pts):
    pair = pts[close, :2]
    refpt = pair[0] - pair[1]
    d1 = close[1] - close[0]
    d2 = len(pts) - close[1] + close[0]
    if d2 < d1:
        refpt = pair[1] - pair[0]
    a = np.arctan2(refpt[1], refpt[0])
    m = np.array([[np.cos(-a), -np.sin(-a)],
                  [np.sin(-a),  np.cos(-a)]])
    return m, d2 < d1

def get_height(closest, pts):
    data = []
    for close in closest:
        m, flip = make_rot(close, pts)
        npt = (pts-pts[close[1]])
        tpt = np.dot(m, npt.T[:2]).T
        if flip:
            data.append(np.vstack([tpt[max(close):], tpt[:min(close)]]).max(0)[1])
        else:
            data.append(tpt[min(close):max(close)].max(0)[1])
    return np.array(data)
    #return np.array([np.dot(make_rot(pts[close]), (pts-pts[close[1]]).T[:2]).max(1)[1] for close in closest])

def _edge_perp(pts, i, close, width=11):
    idx = (np.arange(-(width/2), width/2+1)+i) % len(pts)
    x = np.vstack([pts[idx,0], np.ones((width,))]).T
    m, b = np.linalg.lstsq(x, pts[idx, 1])[0]
    return m

def _line_perp(pts, i, close):
    x, y = np.diff(pts[close,:2], axis=0)[0]
    return y / x

def get_edge_perp(pts, i, close, perp_func=_line_perp):
    m = perp_func(pts, i, close)
    a = np.arctan2(-1./m, 1)
    
    func = lambda h: (np.cos(a)*h + pts[i,0], np.sin(a)*h + pts[i,1])
    p1 = pts[(i+1)%len(pts)] - pts[i]
    p2 = func(1) - pts[i, :2]
    if np.cross(p1[:2], p2) < 0:
        return func
    else:
        return lambda h: func(-h)

def make_path(closest, pts, factor=1, offset_func=_line_perp):
    height = get_height(closest, pts)
    #height /= height.max()
    pm = pts.mean(0)[:2]
    cpts = pts[:,:2] - pm
    verts, codes = [], []
    #return [cpts[close]*h*factor + pm for h, close in zip(height, closest)]
    ccode = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    fcode = [Path.MOVETO, Path.LINETO]
    for h, close in zip(height, closest):
        if h > 0.05:
            verts.append(pts[close[0],:2])
            verts.append(get_edge_perp(pts, close[0], close)(h*factor))
            verts.append(get_edge_perp(pts, close[1], close)(h*factor))
            verts.append(pts[close[1],:2])
            codes += ccode
        else:
            verts.append(pts[close[0], :2])
            verts.append(pts[close[1], :2])
            codes += fcode

    return verts, codes

def _test_inside(close, pts):
    ref = pts[close[0]]
    return np.cross(pts[close[0]-1] - ref, pts[close[1]] - ref) > 0

def draw_curves(closest, pts, factor=1):
    import matplotlib.pyplot as plt
    path = Path(*make_path(closest, pts, factor=factor))
    patch = patches.PathPatch(path, lw=1, facecolor='none')
    ax = plt.gca()
    ax.add_patch(patch)
    ax.plot(pts.T[0], pts.T[1], 'x-')

def draw_lines(closest, pts):
    import matplotlib.pyplot as plt
    inside, outside = [], []
    for close in closest:
        if _test_inside(close, pts[...,:2]):
            inside.append(pts[close, :2])
        else:
            outside.append(pts[close, :2])
    
    codepair = [Path.MOVETO, Path.LINETO]
    inside = Path(np.vstack(inside), codepair*len(inside))
    outside = Path(np.vstack(outside), codepair*len(outside))
    ipatch = patches.PathPatch(inside, lw=1, color='red', facecolor='none')
    opatch = patches.PathPatch(outside, lw=1, color='blue', facecolor='none')

    ax = plt.gca()
    ax.add_patch(ipatch)
    #ax.add_patch(opatch)
    ax.plot(pts.T[0], pts.T[1], 'x-')
    '''
    pts = np.vstack(pairs[...,:2])
    codes = [Path.MOVETO, Path.LINETO]*(len(pts)/2)
    path = Path(pts, codes)
    patch = patches.PathPatch(path, lw=2, facecolor='none')
    ax = plt.gca()
    ax.add_patch(patch)
    '''

def decimate(pts, polys):
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    dec = tvtk.DecimatePro(input=pd)
    dec.set(preserve_topology=True, splitting=False, boundary_vertex_deletion=False, target_reduction=1.0)
    dec.update()
    return dec.output.points.to_array(), dec.output.polys.to_array().reshape(-1, 4)[:,1:]

def boundary_edges(polys):
    edges = dict()
    for i, poly in enumerate(polys):
        p = np.sort(poly)
        for a, b in [(0,1), (1,2), (0, 2)]:
            key = p[a], p[b]
            if key not in edges:
                edges[key] = []
            edges[key].append(i)

    verts = set()
    for (v1, v2), faces in edges.items():
        if len(faces) == 1:
            verts.add(v1)
            verts.add(v2)

    return np.array(list(verts))

def curvature(pts, polys):
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    curv = tvtk.Curvatures(input=pd, curvature_type="mean")
    curv.update()
    return curv.output.point_data.scalars.to_array()

def polysmooth(scalars, polys, smooth=8, neighborhood=3):
    faces = dict()
    for poly in polys:
        for pt in poly:
            if pt not in faces:
                faces[pt] = set()
            faces[pt] |= set(poly)

    def getpts(pt, n):
        if pt in faces:
            for p in faces[pt]:
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

def inside_convex_poly(pts):
    tri = Delaunay(pts)
    return lambda x: tri.find_simplex(x) != -1
    #delaunay triangulation + find_simplex is WAY faster than this method
    # phull = pts[hull]
    # faces = phull.mean(1)
    # norms = np.cross(phull[:,1] - phull[:,0], phull[:,2] - phull[:,0])
    # flipped = (norms * (faces - pts.mean(0))).sum(1) < 0
    # norms[flipped] = -norms[flipped]

    # def func(samples):
    #     svec = faces[np.newaxis] - samples[:,np.newaxis]
    #     #dot product
    #     return ((svec * norms).sum(2) > 0).all(1)

    # return func

def edgefaces(polys, n=1):
    '''Get the edges which belong to n faces. Typically used for searching
    for non-manifold edges or boundary edges'''
    edges = dict()
    def add(x, y):
        key = tuple(sorted([x, y]))
        if key not in edges:
            edges[key] = 0
        edges[key] += 1

    for i, (a, b, c) in enumerate(polys):
        add(a, b)
        add(b, c)
        add(a, c)

    return [k for k, v in edges.items() if v == n]

def make_cube(center=(.5, .5, .5), size=1):
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=float)
    pts -= (.5, .5, .5)
    polys = np.array([(0, 2, 3), (0, 3, 1), (0, 1, 4), (1, 5, 4),
                      (1, 3, 5), (3, 7, 5), (2, 7, 3), (2, 6, 7),
                      (0, 6, 2), (0, 4, 6), (4, 7, 6), (4, 5, 7)], dtype=np.uint32)
    return pts * size + center, polys

if __name__ == "__main__":
    import pickle
    from .db import surfs
    pts, polys = surfs.getSurf("JG", "flat", merge=True, nudge=True)
    fpts, fpolys = surfs.getSurf("JG", "fiducial", merge=True, nudge=False)
    #pts, polys, fpts = cPickle.load(open("/tmp/ptspolys.pkl"))
    left, right = trace_both(pts, polys)
    dist = get_dist(fpts[left])
    rdist = get_dist(fpts[right])
    closest = get_closest_nonadj(dist)
    rclosest = get_closest_nonadj(dist)

    pairs = pts[left][closest]