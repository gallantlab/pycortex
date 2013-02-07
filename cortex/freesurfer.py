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

def show_surf(subject, hemi, type):
    from mayavi import mlab
    from tvtk.api import tvtk

    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)
    surf_file = os.path.join(path, "surf", hemi+'.'+type)
    curv_file = os.path.join(path, "surf", hemi+'.curv')
    
    pts, polys = parse_surf(surf_file)
    curv = parse_curv(curv_file)
    
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