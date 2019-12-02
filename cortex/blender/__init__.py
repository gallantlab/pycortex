import os
import six
import shlex
import xdrlib
import tempfile
import subprocess as sp

import numpy as np

from .. import options
from .. import freesurfer
from .. import dataset
from .. import utils 

default_blender = options.config.get('dependency_paths', 'blender')

_base_imports = """import sys
sys.path.insert(0, '{path}')
import xdrlib
import blendlib
import bpy.ops
from bpy import context as C
from bpy import data as D
""".format(path=os.path.split(os.path.abspath(__file__))[0])

def _call_blender(filename, code, blender_path=default_blender):
    """Call blender, while running the given code. If the filename doesn't exist, save a new file in that location.
    New files will be initially cleared by deleting all objects.
    """
    with tempfile.NamedTemporaryFile() as tf:
        print("In new named temp file: %s"%tf.name)
        startcode = _base_imports
        endcode = "\nbpy.ops.wm.save_mainfile(filepath='{fname}')".format(fname=filename)
        cmd = "{blender_path} -b {fname} -P {tfname}".format(blender_path=blender_path, fname=filename, tfname=tf.name)
        if not os.path.exists(filename):
            startcode += "blendlib.clear_all()\n"
            cmd = "{blender_path} -b -P {tfname}".format(blender_path=blender_path, tfname=tf.name)

        tf.write((startcode+code+endcode).encode())
        tf.flush()
        sp.check_call([w.encode() for w in shlex.split(cmd)],)

def add_cutdata(fname, braindata, name="retinotopy", projection="nearest", mesh="hemi"):
    """Add data as vertex colors to blender mesh
    
    Useful to add localizer data for help in placing flatmap cuts

    Parameters
    ----------
    fname : string
        .blend file name
    braindata : dataview or dataset object
        pycortex data to be shown on the mesh
    name : string
        Name for vertex color object (should indicate what the data is). If a dataset is 
        provided instead of a dataview, this parameter is ignored and the keys for the 
        dataset are used as names for the vertex color objects.
    projection : string
        one of {'nearest', 'trilinear', ...} (name for a pycortex mapper)
    mesh : string
        ...
    """
    if isinstance(braindata, dataset.Dataset):
        for view_name, data in braindata.views.items():
            add_cutdata(fname, data, name=view_name, projection=projection, mesh=mesh)
        return
    from matplotlib import cm
    braindata = dataset.normalize(braindata)
    mapped = braindata.map(projection)
    left = mapped.left
    right = mapped.right

    cmap = utils.get_cmap(braindata.cmap)
    vmin = braindata.vmin
    vmax = braindata.vmax
    lcolor = cmap((left - vmin) / (vmax - vmin))[:,:3]
    rcolor = cmap((right - vmin) / (vmax - vmin))[:,:3]

    p = xdrlib.Packer()
    if six.PY3:
        mesh = mesh.encode()
        name = name.encode()

    p.pack_string(mesh)
    p.pack_string(name)
    p.pack_array(lcolor.ravel(), p.pack_double)
    p.pack_array(rcolor.ravel(), p.pack_double)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            mesh = u.unpack_string().decode('utf-8')
            name = u.unpack_string().decode('utf-8')
            left = u.unpack_array(u.unpack_double)
            right = u.unpack_array(u.unpack_double)
            lcolor = blendlib._repack(left)
            rcolor = blendlib._repack(right)
            print(len(lcolor), len(rcolor))
            blendlib.add_vcolor((lcolor, rcolor), mesh, name)
        """.format(tfname=tf.name)
        _call_blender(fname, code)

    return 


def gii_cut(fname, subject, hemi):
    '''
    Add gifti surface to blender
    '''
    from ..database import db
    hemis = dict(lh='left',
                 rh='right')
    
    wpts, polys = db.get_surf(subject, 'wm', hemi)
    ipts, _ = db.get_surf(subject, 'very_inflated', hemi)
    curvature = db.getSurfInfo(subject, 'curvature')
    rcurv = curvature.__getattribute__(hemis[hemi])

    p = xdrlib.Packer()
    p.pack_array(wpts.ravel(), p.pack_double)
    p.pack_array(ipts.ravel(), p.pack_double)
    p.pack_array(polys.ravel(), p.pack_uint)
    p.pack_array(rcurv.ravel(), p.pack_double)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            wpts = u.unpack_array(u.unpack_double)
            ipts = u.unpack_array(u.unpack_double)
            polys = u.unpack_array(u.unpack_uint)
            curv = u.unpack_array(u.unpack_double)
            blendlib.init_subject(wpts, ipts, polys, curv)
        """.format(tfname=tf.name)
        _call_blender(fname, code)


def fs_cut(fname, subject, hemi, freesurfer_subject_dir=None):
    """Cut freesurfer surface using blender interface

    Parameters
    ----------
    fname : str
        file path for new .blend file (must end in ".blend")

    if `freesurfer_subject_dir` is None, it defaults to SUBJECTS_DIR environment variable
    """
    wpts, polys, curv = freesurfer.get_surf(subject, hemi, 'smoothwm', freesurfer_subject_dir=freesurfer_subject_dir)
    ipts, _, _ = freesurfer.get_surf(subject, hemi, 'inflated', freesurfer_subject_dir=freesurfer_subject_dir)
    rcurv = np.clip(((-curv + .6) / 1.2), 0, 1)
    p = xdrlib.Packer()
    p.pack_array(wpts.ravel(), p.pack_double)
    p.pack_array(ipts.ravel(), p.pack_double)
    p.pack_array(polys.ravel(), p.pack_uint)
    p.pack_array(rcurv.ravel(), p.pack_double)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            wpts = u.unpack_array(u.unpack_double)
            ipts = u.unpack_array(u.unpack_double)
            polys = u.unpack_array(u.unpack_uint)
            curv = u.unpack_array(u.unpack_double)
            blendlib.init_subject(wpts, ipts, polys, curv)
        """.format(tfname=tf.name)
        _call_blender(fname, code)

def write_patch(bname, pname, mesh="hemi"):
    """Write out the mesh 'mesh' in the blender file 'bname' into patch file 'pname'
    This is a necessary step for flattening the surface in freesurfer

    Parameters
    ----------
    bname : str
        blender file name that contains the mesh
    pname : str
        name of patch file to be saved
    mesh : str
        name of mesh in blender file
    """
    p = xdrlib.Packer()
    p.pack_string(pname.encode())
    p.pack_string(mesh.encode())
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            pname = u.unpack_string().decode('utf-8')
            mesh = u.unpack_string().decode('utf-8')
            blendlib.save_patch(pname, mesh)
        """.format(tfname=tf.name)
        _call_blender(bname, code)

