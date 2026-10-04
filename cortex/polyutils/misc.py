
import io

from scipy import sparse
from scipy.spatial import Delaunay
import numpy as np
import functools


def _memo(fn):
    """Helper decorator memoizes the given zero-argument function.
    Really helpful for memoizing properties so they don't have to be recomputed
    dozens of times.
    """
    @functools.wraps(fn)
    def memofn(self, *args, **kwargs):
        if id(fn) not in self._cache:
            self._cache[id(fn)] = fn(self)
        return self._cache[id(fn)]

    return memofn

def tetra_vol(pts):
    '''Volume of a tetrahedron'''
    tetra = pts[1:] - pts[0]
    return np.abs(np.dot(tetra[0], np.cross(tetra[1], tetra[2]))) / 6

def brick_vol(pts):
    '''Volume of a triangular prism'''
    return tetra_vol(pts[[0, 1, 2, 4]]) + tetra_vol(pts[[0, 2, 3, 4]]) + tetra_vol(pts[[2, 3, 4, 5]])

def sort_polys(polys):
    amin = polys.argmin(1)
    xind = np.arange(len(polys))
    return np.array([polys[xind, amin], polys[xind, (amin+1)%3], polys[xind, (amin+2)%3]]).T

def face_area(pts):
    '''Area of triangles

    Parameters
    ----------
    pts : array_like
        n x 3 x 3 array with n triangles, 3 pts, and (x,y,z) coordinates
    '''
    return 0.5 * np.sqrt((np.cross(pts[:,1]-pts[:,0], pts[:,2]-pts[:,0])**2).sum(1))

def face_volume(pts1, pts2, polys):
    '''Volume of each face in a polyhedron sheet

    The same three-tetrahedron decomposition as `brick_vol`, over every face at
    once. Nodes 0-2 are the `pts1` triangle and 3-5 the `pts2` one.
    '''
    polys = np.asarray(polys)
    corners = np.concatenate([np.asarray(pts1)[polys],
                              np.asarray(pts2)[polys]], axis=1)
    vols = np.zeros(len(polys))
    for a, b, c, d in ((0, 1, 2, 4), (0, 2, 3, 4), (2, 3, 4, 5)):
        edges = np.stack([corners[:, b] - corners[:, a],
                          corners[:, c] - corners[:, a],
                          corners[:, d] - corners[:, a]], axis=1)
        vols += np.abs(np.linalg.det(edges)) / 6
    return vols

def decimate(pts, polys):
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    try:
        dec = tvtk.DecimatePro()
        dec.set_input_data(pd)
    except Exception:
        dec = tvtk.DecimatePro(input=pd)  # VTK version < 6

    dec.set(preserve_topology=True, splitting=False, boundary_vertex_deletion=False, target_reduction=1.0)
    dec.update()
    dpts = dec.output.points.to_array()
    dpolys = dec.output.polys.to_array().reshape(-1, 4)[:,1:]
    return dpts, dpolys

def decimate_faces(pts, polys, cell, split=None):
    """Faces for drawing a surface coarsely, over a subset of its vertices.

    The vertices that fall in one box of a grid of `cell` mm over `pts`, and
    that are joined to one another across the mesh, stand for one vertex of
    the coarse surface: the one of them nearest the middle of the group. A box
    holding two pieces of surface that run past each other, the two banks of a
    sulcus for one, has them as separate groups, since they are joined to each
    other only around the fold and so outside the box.

    The faces come back written in the numbering of `pts`, which is what lets
    them be drawn from the arrays an existing surface is already held in: only
    the triangles change, and every vertex of a triangle is a vertex of the
    surface they were cut from.

    Parameters
    ----------
    pts : array (n, 3)
        The vertices of the surface the faces are cut from.
    polys : array (m, 3)
        Its triangles.
    cell : float
        The width of a box of the grid, in the units of `pts`. A surface is
        left as it is when this is 0 or less.
    split : array (n,), optional
        A label per vertex that a group is never allowed to cross, such as
        which side of a cut a vertex is on. Vertices of different labels are
        put in different groups however close together they are.

    Returns
    -------
    verts : array (k,)
        The vertices of `pts` the coarse surface is drawn from.
    faces : array (j, 3)
        Its triangles, in the numbering of `pts`.
    """
    pts = np.asarray(pts, dtype=float)
    polys = np.asarray(polys)
    if cell is None or cell <= 0:
        return np.arange(len(pts)), polys.astype(np.uint32)

    grid = np.floor((pts - pts.min(0)) / cell).astype(np.int64)
    span = grid.max(0) + 1
    if np.prod(span, dtype=float) < 2 ** 62:
        #one number per box, which sorts far faster than the rows do
        boxes = np.unique((grid * [1, span[0], span[0] * span[1]]).sum(1),
                          return_inverse=True)[1]
    else:
        boxes = np.unique(grid, axis=0, return_inverse=True)[1]

    #a box holds as many groups as it has pieces of surface in it
    edges = np.vstack([polys[:, [0, 1]], polys[:, [1, 2]], polys[:, [0, 2]]])
    together = boxes[edges[:, 0]] == boxes[edges[:, 1]]
    if split is not None:
        split = np.asarray(split)
        together &= split[edges[:, 0]] == split[edges[:, 1]]
    joined = edges[together]
    graph = sparse.coo_matrix(
        (np.ones(len(joined), dtype=np.int8), (joined[:, 0], joined[:, 1])),
        shape=(len(pts), len(pts)))
    ngroups, groups = sparse.csgraph.connected_components(graph, directed=False)

    #the vertex nearest the middle of a group stands for it: writing the
    #groups' vertices in from the farthest leaves the nearest one written last
    middles = np.zeros((ngroups, 3))
    for axis in range(3):
        middles[:, axis] = np.bincount(groups, weights=pts[:, axis], minlength=ngroups)
    middles /= np.bincount(groups, minlength=ngroups)[:, None]
    order = np.argsort(-((pts - middles[groups]) ** 2).sum(1))
    verts = np.zeros(ngroups, dtype=np.int64)
    verts[groups[order]] = order

    faces = verts[groups[polys]]
    #a face whose vertices are not three different groups has no area left,
    #and several faces of the surface can come down to the same triangle
    kept = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) &
            (faces[:, 0] != faces[:, 2]))
    faces = faces[kept]
    first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]
    return np.sort(verts), faces[np.sort(first)].astype(np.uint32)


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
    """Returns the two largest connected components, out of a set of boundary
    edges (as returned by `boundary_edges`)
    """
    conn = dict((e, []) for e in np.unique(np.array(edges).ravel()))
    for a, b in edges:
        conn[a].append(b)
        conn[b].append(a)
    
    components = []
    while len(conn) > 0:
        vert, nverts = next(iter(conn.items()))
        poly = [vert]
        while (len(poly) == 1 or poly[0] != poly[-1]) and len(conn[poly[-1]]) > 0:
            nvert = conn[poly[-1]][0]
            conn[nvert].remove(poly[-1])
            conn[poly[-1]].remove(nvert)
            if len(conn[nvert]) == 0:
                del conn[nvert]
            if len(conn[poly[-1]]) == 0:
                del conn[poly[-1]]
            
            poly.append(nvert)

        components.append(poly)
    
    # If the flat surfaces have more than 2 components due to cut leftovers,
    # we filter them by keeping only the two largest components.
    # Note that they are not necessarily ordered as (left, right).
    lengths = [len(comp) for comp in components]
    order = np.argsort(lengths)
    hemisphere_0, hemisphere_1 = components[order[-1]], components[order[-2]]
    return hemisphere_0, hemisphere_1


def rasterize(poly, shape=(256, 256)):
    #ImageDraw sucks at its job, so we'll use imagemagick to do rasterization
    import subprocess as sp
    import shlex
    from PIL import Image
    
    polygon = " ".join(["%0.3f,%0.3f"%tuple(p[::-1]) for p in np.array(poly)-(.5, .5)])
    cmd = 'convert -size %dx%d xc:black -fill white -stroke none -draw "polygon %s" PNG32:-'%(shape[0], shape[1], polygon)
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    png = io.BytesIO(proc.communicate()[0])
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
    
    pd = tvtk.PolyData(points=pts + center + (0, 0, 0), polys=polys)
    plane = tvtk.Planes(normals=[(0,0,1)], points=[(0,0,0)])
    clip = tvtk.ClipPolyData(clip_function=plane, input=pd)
    feats = tvtk.FeatureEdges(
        manifold_edges=False, 
        non_manifold_edges=False, 
        feature_edges=False,
        boundary_edges=True,
        input=clip.output)

    def func(i):
        plane.points = [(0,0,i)]
        feats.update()
        vox = np.zeros(shape[:2][::-1], np.uint8)
        if feats.output.number_of_lines > 0:
            epts = feats.output.points.to_array()
            edges = feats.output.lines.to_array().reshape(-1, 3)[:,1:]
            for poly in trace_poly(edges):
                vox += rasterize(epts[poly][:,:2]+[.5, .5], shape=shape[:2][::-1])
        return vox % 2

    if mp:
        from . import mp
        layers = mp.map(func, range(shape[2]))
    else:
        #layers = map(func, range(shape[2]))
        layers = [func(x) for x in range(shape[2])] # python3 compatible

    return np.array(layers).T

def measure_volume(pts, polys):
    from tvtk.api import tvtk
    pd = tvtk.PolyData(points=pts, polys=polys)
    mp = tvtk.MassProperties(input=pd)
    return mp.volume

def marching_cubes(volume, smooth=True, decimate=True, **kwargs):
    from tvtk.api import tvtk
    imgdata = tvtk.ImageData(dimensions=volume.shape)
    imgdata.point_data.scalars = volume.flatten('F')

    contours = tvtk.ContourFilter(input=imgdata, number_of_contours=1)
    contours.set_value(0, 1)

    if smooth:
        smoothargs = dict(number_of_iterations=40, feature_angle = 90, pass_band=.05)
        smoothargs.update(kwargs)
        contours = tvtk.WindowedSincPolyDataFilter(input=contours.output, **smoothargs)
    if decimate:
        contours = tvtk.QuadricDecimation(input=contours.output, target_reduction=.75)
    
    contours.update()
    pts = contours.output.points.to_array()
    polys = contours.output.polys.to_array().reshape(-1, 4)[:,1:]
    return pts, polys
