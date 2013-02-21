import os
import struct
import tempfile
import shlex
import subprocess as sp

import numpy as np

import vtkutils_new as vtk

def parse_surf(filename):
    with open(filename) as fp:
        #skip magic
        fp.seek(3)
        comment = ' '
        while comment[-1] != '\n':
            comment += fp.read(1)
        comment += fp.read(1)
        print comment[1:-2]
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.fromstring(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.fromstring(fp.read(4*3*faces), dtype='I4').byteswap()
        print fp.read()
        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def parse_curv(filename):
    with open(filename) as fp:
        fp.seek(15)
        return np.fromstring(fp.read(), dtype='>f4').byteswap()

def parse_patch(filename):
    with open(filename) as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data

def get_surf(subject, hemi, type, patch=None, curv='wm'):
    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)

    if type == "patch":
        assert patch is not None
        type = "smoothwm"

    surf_file = os.path.join(path, "surf", hemi+'.'+type)
    
    if curv == "wm":
        curv_file = os.path.join(path, "surf", hemi+'.curv')
    else:
        curv_file = os.path.join(path, 'surf', hemi+'.curv.'+curv)
    
    pts, polys = parse_surf(surf_file)

    if patch is not None:
        patch_file = os.path.join(path, "surf", hemi+'.'+patch+'.patch.3d')
        patch = parse_patch(patch_file)
        verts = patch[patch['vert'] > 0]['vert'] - 1
        edges = -patch[patch['vert'] < 0]['vert'] - 1

        idx = np.zeros((len(pts),), dtype=bool)
        idx[verts] = True
        idx[edges] = True
        valid = idx[polys.ravel()].reshape(-1, 3).all(1)
        polys = polys[valid]

    if type == "patch":
        return patch[['x', 'y', 'z']], polys

    return pts, polys, parse_curv(curv_file)

def show_surf(subject, hemi, type, patch=None):
    from mayavi import mlab
    from tvtk.api import tvtk

    pts, polys = get_surf(subject, hemi, type, patch)
    
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
