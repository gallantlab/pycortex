import struct
import numpy as np
from matplotlib import cm, colors

import bpy.ops
from bpy import context as C
from bpy import data as D

def make_object(pts, polys, name="mesh"):
    mesh = D.meshes.new(name)
    mesh.from_pydata(pts.tolist(), [], polys.tolist())
    obj = D.objects.new(name, mesh)
    C.scene.objects.link(obj)
    return obj, mesh

class Hemi(object):
    def __init__(self, pts, polys, curv=None, name='hemi'):
        self.mesh, self.obj = make_object(pts, polys, name=name)
        self._loopidx = np.zeros((len(self.mesh.loops),), dtype=np.uint32)
        self.mesh.loops.foreach_get('vertex_index', self._loopidx)
        self.obj.scale = .1, .1, .1
        C.scene.objects.active = self.obj
        #Add basis shape
        bpy.ops.object.shape_key_add()
        self.addVColor(curv, name='curvature',vmin=-1, vmax=1)
    
    def addVColor(self, color, name='color', cmap=cm.RdBu_r, vmin=None, vmax=None):
        if color.ndim == 1:
            if vmin is None:
                vmin = color.min()
            if vmax is None:
                vmax = color.max()
            color = cmap((color - vmin) / (vmax - vmin))[:,:3]

        vcolor = self.mesh.vertex_colors.new(name)
        for i, j in enumerate(self._loopidx):
            vcolor.data[i].color = list(color[j])

    def addShape(self, shape, name=None):
        C.scene.objects.active = self.obj
        self.obj.select = True
        bpy.ops.object.shape_key_add()
        key = D.shape_keys[-1].key_blocks[-1]
        if name is not None:
            key.name = name

        for i in range(len(key.data)):
            key.data[i].co = shape[i]
        return key

def show(data, subject, xfmname, types=('inflated',)):
    from .db import surfs
    surfs.getVTK(data, "fiducial")

def fs_cut(subject, hemi):
    from .freesurfer import get_surf
    wpts, polys, curv = get_surf(subject, hemi, )
    ipts, polys, _ = surfs.getVTK(subjfs, 'inflated', hemi)
    npz = np.load(surfs.getAnat(subjfs, 'curvature'))
    curv = npz[dict(lh='left', rh='right')[hemi]]
    
    hemi = Hemi(wpts, polys, curv=curv)
    hemi.addShape(ipts, name='inflated')
    return hemi

def write_patch(outfile, hemi):
    if isinstance(hemi, Hemi):
        mesh = hemi.mesh
    else:
        mesh = hemi

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all()
    C.tool_settings.mesh_select_mode = False, True, False
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.object.mode_set(mode='OBJECT')

    mwall_edge = set()
    for edge in mesh.edges:
        if edge.select:
            mwall_edge.add(edge.vertices[0])
            mwall_edge.add(edge.vertices[1])

    bpy.ops.object.mode_set(mode='EDIT') 
    C.tool_settings.mesh_select_mode = True, False, False
    bpy.ops.mesh.select_all()
    bpy.ops.object.mode_set(mode='OBJECT')
    seam = set()
    for edge in mesh.edges:
        if edge.use_seam:
            seam.add(edge.vertices[0])
            seam.add(edge.vertices[1])
            edge.select = True

    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_more()
    bpy.ops.object.mode_set(mode='OBJECT')
    smore = set()
    for i, vert in enumerate(mesh.vertices):
        if vert.select:
            smore.add(i)

    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all()
    bpy.ops.object.mode_set(mode='OBJECT')

    fverts = set()
    for face in mesh.polygons:
        fverts.add(face.vertices[0])
        fverts.add(face.vertices[1])
        fverts.add(face.vertices[2])

    edges = mwall_edge | (smore - seam)
    verts = fverts - seam

    with open(outfile, 'wb') as fp:
        fp.write(struct.pack('>2i', -1, len(verts)))
        for v in verts:
            pt = D.shape_keys['Key'].key_blocks['inflated'].data[v].co
            if v in edges:
                fp.write(struct.pack('>i3f', -v-1, *pt))
            else:
                fp.write(struct.pack('>i3f', v+1, *pt))