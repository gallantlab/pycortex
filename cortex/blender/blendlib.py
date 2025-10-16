"""
This module is intended to be imported directly by blender.
It provides utility functions for adding meshes and saving them to communicate with the rest of pycortex.

Read more about Blender Python API here: https://docs.blender.org/api/current/index.html.
"""
import struct
from mda_xdrlib import xdrlib
import tempfile
import time
import math

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

def _write_patch(filename, pts, edges=None):
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
    print("Wrote freesurfer patch to %s"%filename)

def _circularize_uv_coords(pts, u_min, u_max, v_min, v_max):
    """Transform UV coordinates into a circular shape while preserving relative positions.
    
    Parameters
    ----------
    pts : dict
        Dictionary mapping vertex indices to (u, v, z) coordinates
    u_min, u_max, v_min, v_max : float
        Original bounds of the UV coordinates
        
    Returns
    -------
    dict
        Dictionary mapping vertex indices to new (u, v, z) coordinates
    """
    # Convert to normalized coordinates in [-1, 1] range
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2
    u_scale = (u_max - u_min) / 2
    v_scale = (v_max - v_min) / 2
    
    new_pts = {}
    for idx, (u, v, z) in pts.items():
        # Normalize coordinates
        u_norm = (u - u_center) / u_scale
        v_norm = (v - v_center) / v_scale
        
        # Convert to polar coordinates
        r = math.sqrt(u_norm**2 + v_norm**2)
        theta = math.atan2(v_norm, u_norm)
        
        # Normalize radius to create perfect circle
        # Use square root to preserve area/density
        r = math.sqrt(r)
        
        # Convert back to Cartesian coordinates
        u_new = r * math.cos(theta)
        v_new = r * math.sin(theta)
        
        # Scale back to original range
        u_new = u_new * u_scale + u_center
        v_new = v_new * v_scale + v_center
        
        new_pts[idx] = (u_new, v_new, z)
    
    return new_pts

def _get_geometry(mesh, hemi, flatten, method=None):
    """Function called within blender to get non-cut vertices & edges

    Operates on a mesh object within an open instance of blender. 

    Parameters
    ----------
    mesh : str
        name of mesh to cut
    hemi : str
        hemisphere name (lh or rh)
    flatten : bool
        if True, returns flattened coordinates using UV unwrap
    method : str
        method to use for UV unwrap. One of 'CONFORMAL', 'ANGLE_BASED', 'MINIMUM_STRETCH'.

    Returns
    -------
    verts : set
        set of vertex indices
    pts : list
        list of (vertex_index, flattened_coordinates) tuples
    edges : set
        set of edge vertex indices
    """
    if isinstance(mesh, str):
        mesh = D.meshes[mesh]

    # Collect edge vertex indices
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    C.tool_settings.mesh_select_mode = False, True, False
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.object.mode_set(mode='OBJECT')

    edge_vertex_idxs = set()  # Medial wall in standard case
    for edge in mesh.edges:
        if edge.select:
            edge_vertex_idxs.add(edge.vertices[0])
            edge_vertex_idxs.add(edge.vertices[1])

    # Collect seam vertex indices & select seams
    bpy.ops.object.mode_set(mode='EDIT') 
    C.tool_settings.mesh_select_mode = True, False, False
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    seam_vertex_idxs = set()
    for edge in mesh.edges:
        if edge.use_seam:
            seam_vertex_idxs.add(edge.vertices[0])
            seam_vertex_idxs.add(edge.vertices[1])
            edge.select = True

    # Expand seam selection & collect expanded vertex indices
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_more()
    bpy.ops.object.mode_set(mode='OBJECT')
    expanded_seam_vertex_idxs = set()
    for i, vert in enumerate(mesh.vertices):
        if vert.select:
            expanded_seam_vertex_idxs.add(i)

    # Leave cuts (+ area around them) selected.
    # Uncomment the next lines to revert to previous behavior
    # (deselecting everything)
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='DESELECT')
    # bpy.ops.object.mode_set(mode='OBJECT')

    face_vertices = set()
    for face in getattr(mesh, "polygons", getattr(mesh, "faces", None)):
        face_vertices.add(face.vertices[0])
        face_vertices.add(face.vertices[1])
        face_vertices.add(face.vertices[2])

    verts = face_vertices - seam_vertex_idxs
    pts = [(v, D.shape_keys['Key'].key_blocks['inflated'].data[v].co) for v in verts]
    edges = edge_vertex_idxs | (expanded_seam_vertex_idxs - seam_vertex_idxs)

    if flatten:
        # Scales
        u_coords, v_coords = [u for _, (u, _, _) in pts], [v for _, (_, v, _) in pts]
        u_min, u_max, v_min, v_max = min(u_coords), max(u_coords), min(v_coords), max(v_coords)
        print("u_min: %f, u_max: %f, v_min: %f, v_max: %f"%(u_min, u_max, v_min, v_max))

        if not mesh.uv_layers:
            mesh.uv_layers.new(name="FlattenUV")
        
        print("UV unwrapping mesh with method %s (may take a few minutes)..."%method)
        start = time.time()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method=method, margin=0.001)
        bpy.ops.object.mode_set(mode='OBJECT')
        end = time.time()
        print("UV unwrapping mesh took %.1f seconds" % (end - start))
        
        print("Collecting coordinates for %d verts..."%len(verts))
        start = time.time()
        pts = {}
        for loop in mesh.loops:
            if loop.vertex_index in verts:
                coords2d = mesh.uv_layers.active.data[loop.index].uv
                u_scaled = coords2d[0] * (u_max - u_min) + u_min
                v_scaled = coords2d[1] * (v_max - v_min) + v_min
                
                if hemi == "rh":
                    # Rotate 180 degrees clockwise
                    u_center = (u_max + u_min) / 2
                    v_center = (v_max + v_min) / 2
                    u_rotated = 2 * u_center - u_scaled
                    v_rotated = 2 * v_center - v_scaled
                    pts[loop.vertex_index] = (u_rotated, v_rotated, 0.0)
                else:
                    pts[loop.vertex_index] = (u_scaled, v_scaled, 0.0)
        
        # Circularize the UV coordinates
        print("Circularizing UV coordinates...")
        pts = _circularize_uv_coords(pts, u_min, u_max, v_min, v_max)
        
        pts = sorted(pts.items(), key=lambda x: x[0])
        end = time.time()
        print("Collecting coordinates took %.1f seconds" % (end - start))

    print("Collected geometry. verts: %d, pts: %d, edges: %d"%(len(verts), len(pts), len(edges)))
    return verts, pts, edges


def save_patch(fname, mesh="hemi"):
    """Deprecated: please use write_volume_patch instead"""
    return write_volume_patch(fname, "lh", mesh)


def write_volume_patch(fname, hemi, mesh="hemi"):
    """Write mesh patch in freesurfer format"""
    _, pts, edges = _get_geometry(mesh, hemi, flatten=False)
    _write_patch(fname, pts, edges)


def write_flat_patch(fname, hemi, mesh="hemi", method="MINIMUM_STRETCH"):
    """Write flat patch in freesurfer format
    
    Parameters
    ----------
    fname : str
        Output filename
    hemi : str
        hemisphere name (lh or rh)
    mesh : str
        Name of the mesh to flatten
    method : str
        UV unwrapping method to use
    """
    _, pts, edges = _get_geometry(mesh, hemi, flatten=True, method=method)
    _write_patch(fname, pts, edges)


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
