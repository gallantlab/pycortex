"""This module is intended to be imported directly by blender.
It provides utility functions for adding meshes and saving them to communicate with the rest of pycortex
"""
import struct
import xdrlib
import tempfile

import bpy.ops
from bpy import context as C
from bpy import data as D

def _repack(linear, n=3):
    """This ridiculous function returns chunks of n from a linear list.
    For example, _repack([1, 2, 3, 4, 5, 6], n=3) -> [[1, 2, 3], [4, 5, 6]]
    Good for unravelling ravelled data
    """
    return list(zip(*[iter(linear)] * n))

def clear_all():
    """Remove all objects from the active scene in blender"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def init_subject(wpts, ipts, polys, curv):
    """Initialize a subject """
    print('Started init_subject in blender!')
    obj, mesh = make_object(_repack(wpts), _repack(polys), name='hemi')
    obj.scale = .1, .1, .1
    if bpy.app.version < (2, 80, 0):
        # Backward compatibility
        obj.select = True
        C.scene.objects.active = obj
    else:
        obj.select_set(True)
        C.view_layer.objects.active = obj

    bpy.ops.object.shape_key_add()
    add_vcolor(curv, mesh, name='curvature')
    show_vertex_colors()
    add_shapekey(_repack(ipts), name='inflated')
    obj.use_shape_key_edit_mode = True

def make_object(pts, polys, name="mesh"):
    mesh = D.meshes.new(name)
    mesh.from_pydata(pts, [], polys)
    obj = D.objects.new(name, mesh)
    if bpy.app.version > (2, 80, 0):
        C.scene.collection.objects.link(obj)
    else:
        C.scene.objects.link(obj)
    return obj, mesh

def get_ptpoly(name):
    verts = D.meshes[name].vertices
    faces = D.meshes[name].polygons
    pts = np.empty((len(verts),3))
    polys = np.empty((len(faces),3), dtype=np.uint32)
    verts.foreach_get('co', pts.ravel())
    faces.foreach_get('vertices', polys.ravel())
    return pts, polys


def show_vertex_colors():
    """Vertex colors are fussy to find and display via the interface. So display them automatically"""
    if bpy.app.version > (2, 80, 0):
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'


def add_vcolor(hemis, mesh=None, name='color'):
    if mesh is None:
        if bpy.app.version < (2, 80, 0):
            mesh = C.scene.objects.active.data
        else:
            mesh = C.view_layer.objects.active.data
    elif isinstance(mesh, str):
        mesh = D.meshes[mesh]

    bpy.ops.object.mode_set(mode='OBJECT')

    color = hemis
    if len(hemis) == 2:
        color = hemis[0]
        if len(mesh.vertices) == len(hemis[1]):
            color = hemis[1]
    vcolor = mesh.vertex_colors.new(name=name)
    if hasattr(mesh, "loops"):
        loopidx = [0]*len(mesh.loops)
        mesh.loops.foreach_get('vertex_index', loopidx)
        if not isinstance(color[0], (list, tuple)):
            for i, j in enumerate(loopidx):
                if bpy.app.version < (2, 80, 0):
                    vcolor.data[i].color = [color[j]]*3
                else:
                    vcolor.data[i].color = [color[j]]*3 + [1]
        else:
            for i, j in enumerate(loopidx):
                if bpy.app.version < (2, 80, 0): 
                    vcolor.data[i].color = color[j]
                else:
                    vcolor.data[i].color = list(color[j]) + [1]
    else:
        # older blender version, need to iterate faces instead
        print("older blender found...")
        if not isinstance(color[0], (list, tuple)):
            for i in range(len(mesh.faces)):
                v = mesh.faces[i].vertices
                vcolor.data[i].color1 = [color[v[0]]] * 3
                vcolor.data[i].color2 = [color[v[1]]] * 3
                vcolor.data[i].color3 = [color[v[2]]] * 3
        else:
            for i in len(vcolor):
                v = mesh.faces[i].vertices
                vcolor.data[i].color1 = color[v[0]]
                vcolor.data[i].color2 = color[v[1]]
                vcolor.data[i].color3 = color[v[2]]
    print("Successfully added vcolor '%s'"%name)
    return vcolor

def add_shapekey(shape, name=None):
    bpy.ops.object.shape_key_add()
    key = D.shape_keys[-1].key_blocks[-1]
    if name is not None:
        key.name = name

    for i in range(len(key.data)):
        key.data[i].co = shape[i]
    return key

def write_patch(filename, pts, edges=None):
    """Writes a patch file that is readable by freesurfer.
    
    Parameters
    ----------
    filename : name for patch to write. Should be of the form 
        <subject>.flatten.3d
    pts : array-like
        points in the mesh
    edges : array-like
        edges in the mesh

    """
    if edges is None:
        edges = set()

    with open(filename, 'wb') as fp:
        fp.write(struct.pack('>2i', -1, len(pts)))
        for i, pt in pts:
            if i in edges:
                fp.write(struct.pack('>i3f', -i-1, *pt))
            else:
                fp.write(struct.pack('>i3f', i+1, *pt))

def _get_pts_edges(mesh):
    """Function called within blender to get non-cut vertices & edges

    Operates on a mesh object within an open instance of blender. 

    Parameters
    ----------
    mesh : str
        name of mesh to cut
    """
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
    # Leave cuts (+ area around them) selected.
    # Uncomment the next lines to revert to previous behavior
    # (deselecting everything)
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='DESELECT')
    # bpy.ops.object.mode_set(mode='OBJECT')

    fverts = set()
    if hasattr(mesh, "polygons"):
        faces = mesh.polygons
    else:
        faces = mesh.faces
    for face in faces:
        fverts.add(face.vertices[0])
        fverts.add(face.vertices[1])
        fverts.add(face.vertices[2])

    print("exported %d faces"%len(fverts))
    edges = mwall_edge | (smore - seam)
    verts = fverts - seam
    pts = [(v, D.shape_keys['Key'].key_blocks['inflated'].data[v].co) for v in verts]
    return verts, pts, edges

def save_patch(fname, mesh='hemi'):
    """Saves patch to file that can be read by freesurfer"""
    verts, pts, edges = _get_pts_edges(mesh)
    write_patch(fname, pts, edges)

def read_xdr(filename):
    with open(filename, "rb") as fp:
        u = xdrlib.Unpacker(fp.read())
        pts = u.unpack_array(p.unpack_double)
        polys = u.unpack_array(p.unpack_uint)
        return pts, polys

def write_xdr(filename, pts, polys):
    with open(filename, "wb") as fp:
        p = xdrlib.Packer()
        p.pack_array(pts, p.pack_double)
        p.pack_array(polys, p.pack_uint)
        fp.write(p.get_buffer())
