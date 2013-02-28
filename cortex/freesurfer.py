import os
import struct
import tempfile
import shlex
import subprocess as sp

import numpy as np

from . import db
from . import vtkutils_new as vtk

def import_subj(subject, sname=None):
    import nibabel
    surfs = os.path.join(db.filestore, "surfaces", "{subj}_{name}_{hemi}.vtk")
    anats = os.path.join(db.filestore, "anatomicals", "{subj}_{name}.{type}")
    fspath = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'mri')

    if sname is None:
        sname = subject

    #import anatomicals
    for fsname, name in dict(T1="raw", wm="whitematter").items():
        path = os.path.join(fspath, "{fsname}.mgz").format(fsname=fsname)
        out = anats.format(subj=sname, name=name, type='nii.gz')
        cmd = "mri_convert {path} {out}".format(path=path, out=out)
        sp.call(shlex.split(cmd))

    #Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter
    trans = nibabel.load(out).get_affine()[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    curvs = dict(lh=[], rh=[])
    #import surfaces
    for fsname, name in dict(smoothwm="wm", pial="pia", inflated="inflated").items():
        for hemi in ("lh", "rh"):
            pts, polys, curv = get_surf(subject, hemi, fsname)
            fname = surfs.format(subj=sname, name=name, hemi=hemi)
            vtk.write(fname, pts + surfmove, polys)
            if fsname == 'smoothwm':
                curvs[hemi] = curv
    import ipdb
    ipdb.set_trace()
    np.savez(anats.format(subj=sname, name="curvature", type='nii.npz'), left=curvs['lh'], right=curvs['rh'])

def parse_surf(filename):
    with open(filename, 'rb') as fp:
        #skip magic
        fp.seek(3)
        comment = fp.readline()
        fp.readline()
        print(comment)
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.fromstring(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.fromstring(fp.read(4*3*faces), dtype='I4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def write_surf(filename, pts, polys, comment=''):
    with open(filename, 'wb') as fp:
        fp.write('\xff\xff\xfe')
        fp.write(comment+'\n\n')
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap().tostring())
        fp.write(polys.astype(np.uint32).byteswap().tostring())
        fp.write('\n')

def parse_curv(filename):
    with open(filename, 'rb') as fp:
        fp.seek(15)
        return np.fromstring(fp.read(), dtype='>f4').byteswap().newbyteorder()

def parse_patch(filename):
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data

def get_surf(subject, hemi, type, patch=None, curv='wm'):
    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)

    if type == "patch":
        assert patch is not None
        surf_file = os.path.join(path, "surf", hemi+'.smoothwm')
    else:
        surf_file = os.path.join(path, "surf", hemi+'.'+type)
    
    if curv == "wm":
        curv_file = os.path.join(path, "surf", hemi+'.curv')
    else:
        curv_file = os.path.join(path, 'surf', hemi+'.curv.'+curv)
    
    pts, polys = parse_surf(surf_file)

    if patch is not None:
        patch_file = os.path.join(path, "surf", hemi+'.'+patch+'.patch.3d')
        try:
            patch = parse_patch(patch_file)
        except IOError:
            patch_file = os.path.join(path, "surf", patch)
            patch = parse_patch(patch_file)
        verts = patch[patch['vert'] > 0]['vert'] - 1
        edges = -patch[patch['vert'] < 0]['vert'] - 1

        idx = np.zeros((len(pts),), dtype=bool)
        idx[verts] = True
        idx[edges] = True
        valid = idx[polys.ravel()].reshape(-1, 3).all(1)
        polys = polys[valid]
        idx = np.zeros((len(pts),))
        idx[verts] = 1
        idx[edges] = -1

    if type == "patch":
        for i, x in enumerate(['x', 'y', 'z']):
            pts[verts, i] = patch[patch['vert'] > 0][x]
            pts[edges, i] = patch[patch['vert'] < 0][x]
        return pts, polys, idx

    return pts, polys, parse_curv(curv_file)

def show_surf(subject, hemi, type, patch=None):
    from mayavi import mlab
    from tvtk.api import tvtk

    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)
    pts, polys, curv = get_surf(subject, hemi, type, patch)
    
    fig = mlab.figure()
    src = mlab.pipeline.triangular_mesh_source(pts[:,0], pts[:,1], pts[:,2], polys, scalars=curv, figure=fig)
    norms = mlab.pipeline.poly_data_normals(src, figure=fig)
    norms.filter.splitting = False
    surf = mlab.pipeline.surface(norms, figure=fig)
    surf.parent.scalar_lut_manager.set(lut_mode='RdBu', data_range=[-1,1], use_default_range=False)

    cursors = mlab.pipeline.scalar_scatter([0], [0], [0])
    glyphs = mlab.pipeline.glyph(cursors, figure=fig)
    glyphs.glyph.glyph_source.glyph_source = glyphs.glyph.glyph_source.glyph_dict['axes']

    fig.scene.background = (0,0,0)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

    def picker_callback(picker):
        if picker.actor in surf.actor.actors:
            npts = np.append(cursors.data.points.to_array(), [pts[picker.point_id]], axis=0)
            cursors.data.points = npts
            print picker.point_id
            x, y, z = pts[picker.point_id]
            with open(os.path.join(path, 'tmp', 'edit.dat'), 'w') as fp:
                fp.write('%f %f %f\n'%(x, y, z))

    picker = fig.on_mouse_pick(picker_callback)
    picker.tolerance = 0.01

    return surf

def write_dot(fname, pts, polys, name="test"):
    import networkx as nx
    def iter_surfedges(tris):
        for a,b,c in tris:
            yield a,b
            yield b,c
            yield a,c
    graph = nx.Graph()
    graph.add_edges_from(iter_surfedges(polys))
    lengths = []
    with open(fname, "w") as fp:
        fp.write("graph %s {\n"%name)
        fp.write('node [shape=point,label=""];\n')
        for a, b in graph.edges_iter():
            l = np.sqrt(((pts[a] - pts[b])**2).sum(-1))
            lengths.append(l)
            fp.write("%s -- %s [len=%f];\n"%(a, b, l))
        fp.write("maxiter=1000000;\n");
        fp.write("}")

def read_dot(fname, pts):
    import re
    parse = re.compile(r'\s(\d+)\s\[label="", pos="([\d\.]+),([\d\.]+)".*];')
    data = np.zeros((len(pts), 2))
    with open(fname) as fp:
        fp.readline()
        fp.readline()
        fp.readline()
        fp.readline()
        el = fp.readline().split(' ')
        while el[1] != '--':
            x, y = el[2][5:-2].split(',')
            data[int(el[0][1:])] = float(x), float(y)
            el = fp.readline().split(' ')
    return data

def write_decimated(path, pts, polys):
    from .polyutils import decimate, boundary_edges
    dpts, dpolys = decimate(pts, polys)
    write_surf(path+'.smoothwm', dpts, dpolys)
    edges = boundary_edges(dpolys)
    data = np.zeros((len(dpts),), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    data['vert'] = np.arange(len(dpts))+1
    data['vert'][edges] *= -1
    data['x'] = dpts[:,0]
    data['y'] = dpts[:,1]
    data['z'] = dpts[:,2]
    with open(path+'.full.patch.3d', 'wb') as fp:
        fp.write(struct.pack('>i', -1))
        fp.write(struct.pack('>i', len(dpts)))
        fp.write(data.tostring())

from scipy.spatial import cKDTree
class SpringLayout(object):
    def __init__(self, pts, polys, dpts=None, pins=None, stepsize=.1):
        import networkx as nx
        self.pts = pts
        self.polys = polys
        self.stepsize = stepsize
        if pins is None:
            pins = np.zeros((len(pts),), dtype=bool)
        self.pins = pins
        self.neighbors = [set() for _ in range(len(pts))]
        self.graph = nx.Graph()
        for i, j, k in polys:
            self.neighbors[i].add(j)
            self.neighbors[i].add(k)
            self.neighbors[j].add(i)
            self.neighbors[j].add(k)
            self.neighbors[k].add(i)
            self.neighbors[k].add(j)
            self.graph.add_edges_from([(i,j), (i,k), (j,k)])

        for i in range(len(self.neighbors)):
            self.neighbors[i] = list(self.neighbors[i])

        if dpts is None:
            dpts = pts

        self.dists = []
        for i, neighbors in enumerate(self.neighbors):
            self.dists.append(np.sqrt(((dpts[neighbors] - dpts[i])**2).sum(-1)))

        self.kdt = cKDTree(self.pts)
        self._next = self.pts.copy()

    def _spring(self, i):
        diff = self.pts[self.neighbors[i]] - self.pts[i]
        dist = np.sqrt((diff**2).sum(1))
        mag = self.stepsize * (self.dists[i] - dist)
        return (mag * diff.T).T.mean(0)

    def _estatic(self, idx):
        dist, neighbors = self.kdt.query(self.pts[idx], k=20)
        valid = dist > 0
        mag = self.stepsize * (1 / dist)
        diff = self.pts[neighbors] - self.pts[idx]
        return (mag[valid] * diff[valid].T).T.mean(0)

    def step(self):
        for i in range(len(self.pts)):
            self._next[i] -= self._spring(i) #+ self._estatic(i)
        self.pts[:] = self._next
        #self.kdt = cKDTree(self.pts)

    def run(self, n=1000, show=False):
        if show:
            import matplotlib.pyplot as plt
            import networkx as nx
            nx.draw(self.graph, pos=self.pts)
        for _ in range(n):
            self.step()
            if show:
                plt.clf()
                nx.draw(self.graph, pos=self.pts)
                plt.show()
