import os
import shlex
import xdrlib
import tempfile
import subprocess as sp

import numpy as np

from .. import freesurfer
from .. import dataset

_base_imports = """import sys
sys.path.insert(0, '{path}')
import xdrlib
import blendlib
import bpy.ops
from bpy import context as C
from bpy import data as D
""".format(path=os.path.split(os.path.abspath(__file__))[0])

def _call_blender(filename, code):
    """Call blender, while running the given code. If the filename doesn't exist, save a new file in that location.
    New files will be initially cleared by deleting all objects.
    """
    with tempfile.NamedTemporaryFile() as tf:
        startcode=_base_imports
        endcode = "\nbpy.ops.wm.save_mainfile(filepath='{fname}')".format(fname=filename)
        cmd = "blender -b {fname} -P {tfname}".format(fname=filename, tfname=tf.name)
        if not os.path.exists(filename):
            startcode += "blendlib.clear_all()\n"
            endcode = "\nbpy.ops.wm.save_mainfile(filepath='{fname}')".format(fname=filename)
            cmd = "blender -b -P {tfname}".format(tfname=tf.name)

        tf.write(startcode+code+endcode)
        tf.flush()
        sp.call(shlex.split(cmd))

def add_cutdata(fname, dataview, name="retinotopy", projection="nearest", mesh="hemi"):
    from matplotlib import cm
    dataview = dataset.normalize(dataview)
    mapped = dataview.data.map(projection)
    vcolor = mapped.data

    cmap = cm.get_cmap(dataview.cmap)
    vmin = dataview.vmin
    vmax = dataview.vmax
    vcolor = cmap((vcolor - vmin) / (vmax - vmin))[:,:3]

    p = xdrlib.Packer()
    p.pack_string(mesh)
    p.pack_string(name)
    p.pack_array(vcolor.ravel(), p.pack_double)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            mesh = u.unpack_string().decode('utf-8')
            name = u.unpack_string().decode('utf-8')
            color = u.unpack_array(u.unpack_double)
            blendlib.add_vcolor(blendlib._repack(color), mesh, name)
        """.format(tfname=tf.name)
        _call_blender(fname, code)

    return 

def fs_cut(fname, subject, hemi):
    wpts, polys, curv = freesurfer.get_surf(subject, hemi, 'smoothwm')
    ipts, _, _ = freesurfer.get_surf(subject, hemi, 'inflated')
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
    """
    p = xdrlib.Packer()
    p.pack_string(pname)
    p.pack_string(mesh)
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