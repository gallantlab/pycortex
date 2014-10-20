import os
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

def add_vcolor(color, mesh=None, name='color', cmap=cm.RdBu, vmin=None, vmax=None):
    if mesh is None:
        mesh = C.scene.objects.active.data
    elif isinstance(mesh, str):
        mesh = D.meshes[mesh]

    bpy.ops.object.mode_set(mode='OBJECT')
    if color.ndim == 1:
        if vmin is None:
            vmin = color.min()
        if vmax is None:
            vmax = color.max()
        color = cmap((color - vmin) / (vmax - vmin))[:,:3]

    loopidx = np.zeros((len(mesh.loops),), dtype=np.uint32)
    mesh.loops.foreach_get('vertex_index', loopidx)

    vcolor = mesh.vertex_colors.new(name)
    for i, j in enumerate(loopidx):
        vcolor.data[i].color = list(color[j])
    return vcolor

def add_shapekey(shape, name=None):
    bpy.ops.object.shape_key_add()
    key = D.shape_keys[-1].key_blocks[-1]
    if name is not None:
        key.name = name

    for i in range(len(key.data)):
        key.data[i].co = shape[i]
    return key

def cut_data(volumedata, name="retinotopy", projection="nearest", cmap=cm.RdBu, vmin=None, vmax=None, mesh="hemi"):
    if isinstance(mesh, str):
        mesh = D.meshes[mesh]

    mapped = volumedata.map(projection)
    if mapped.llen == len(mesh.vertices):
        print("left hemisphere")
        vcolor = mapped.left
    else:
        print ("right hemisphere")
        vcolor = mapped.right

    return add_vcolor(vcolor, mesh=mesh, name=name, cmap=cmap, vmin=vmin, vmax=vmax)

def fs_cut(subject, hemi):
    from .freesurfer import get_surf
    wpts, polys, curv = get_surf(subject, hemi, 'smoothwm')
    ipts, _, _ = get_surf(subject, hemi, 'inflated')

    obj, mesh = make_object(wpts, polys, name='hemi')
    obj.scale = .1, .1, .1
    C.scene.objects.active = obj
    bpy.ops.object.shape_key_add()
    add_vcolor(curv, mesh, vmin=-.6, vmax=.6, name='curvature')
    add_shapekey(ipts, name='inflated')
    obj.use_shape_key_edit_mode = True
    return mesh

def write_patch(subject, hemi, name, mesh='hemi'):
    from .freesurfer import get_paths, write_patch
    if isinstance(mesh, str):
        mesh = D.meshes[mesh]

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
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
    bpy.ops.mesh.select_all(action='DESELECT')
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
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    fverts = set()
    for face in mesh.polygons:
        fverts.add(face.vertices[0])
        fverts.add(face.vertices[1])
        fverts.add(face.vertices[2])

    edges = mwall_edge | (smore - seam)
    verts = fverts - seam
    pts = [(v, D.shape_keys['Key'].key_blocks['inflated'].data[v].co) for v in verts]
    write_patch(get_paths(subject, hemi).format(name=name), pts, edges)