import os
import re
import shlex
import shutil
from mda_xdrlib import xdrlib
import tempfile
import subprocess as sp
import site

import numpy as np

from .. import options
from .. import freesurfer
from .. import dataset
from .. import utils 

default_blender = options.config.get('dependency_paths', 'blender')

_base_imports = """import sys
for site_dir in {site_dirs}:
    print("Adding python site directory to sys.path:", site_dir)
    sys.path.insert(0, site_dir)

from mda_xdrlib import xdrlib
import blendlib
import bpy.ops
from bpy import context as C
from bpy import data as D
""".format(site_dirs=[
    os.path.split(os.path.abspath(__file__))[0],
    *site.getsitepackages(),
])


def _wrap_code(code, filename):
    """
    Wrap code for running in blender

    Parameters
    ----------
    code : str
        code to run in blender
    filename : str
        file path for blender file (must end in ".blend")
    """
    wrapped_code = _base_imports
    if not os.path.exists(filename):
        wrapped_code += "blendlib.clear_all()\n"
    wrapped_code += code
    wrapped_code += "\nbpy.ops.wm.save_mainfile(filepath='{fname}')".format(fname=filename)
    return wrapped_code


def _call_blender(filename, code=None, background=True, blender_path=default_blender):
    """
    Call blender, while running the given code. If the filename doesn't exist, save a new file in that location.
    New files will be initially cleared by deleting all objects.

    Parameters
    ----------
    filename : str
        file path for blender file (must end in ".blend")
    code : str, optional
        code to run in blender. If None, blender will be opened without running any code.
    background : bool, optional
        If True, blender will be opened in background mode.
    blender_path : str, optional
        Path to blender executable. If None, defaults to the path specified in pycortexconfig file.
    """
    with tempfile.NamedTemporaryFile() as tf:
        print("In new named temp file: %s"%tf.name)

        # Backup
        if os.path.exists(filename):
            _legacy_blender_backup(filename, blender_path=blender_path)

        # Construct command
        cmd = blender_path
        if background:
            cmd += " -b"
        if os.path.exists(filename):
            cmd += " " + filename
        if code is not None:
            wrapped_code = _wrap_code(code, filename)
            tf.write(wrapped_code.encode())
            tf.flush()
            cmd += " -P {tfname}".format(tfname=tf.name)

        print(f"Calling blender:\n    {cmd}")
        sp.check_call([w.encode() for w in shlex.split(cmd)],)


def _check_executable_blender_version(blender_path=default_blender):
    """Get blender version number"""
    blender_version = sp.check_output([blender_path, '--version']).decode()
    blender_version = blender_version.split('\n')[0]
    print("Detected %s"%blender_version)
    blender_version_number = re.findall('(?<=Blender\s)[0-9]*[,.][0-9]*', blender_version)[0]
    blender_version_number = tuple([int(x) for x in blender_version_number.split('.')])
    # For ver 2.79.x, minor version is not always returned, so standardize:
    blender_major_version_number = blender_version_number[:2]
    return blender_major_version_number


def _check_file_blender_version(fpath):
    """Check which version of blender saved a particular file"""
    import struct
    with open(fpath, mode='rb') as fid:
        fid.seek(7)
        bitness, endianness, major, minor = struct.unpack("sss2s", fid.read(5))
    return (int(major), int(minor))


def _legacy_blender_backup(fname, blender_path=default_blender):
    """Create a copy of a .blend file, because if a blender 2.7x file is 
    opened with blender 2.8+, it usually can't be opened with 2.7x again.
    
    Yes this seems quite bad."""
    executable_28 = _check_executable_blender_version(blender_path=blender_path) >= (2, 80)
    file_27 = _check_file_blender_version(fname) < (2,80)
    if executable_28 and file_27:
        fname_bkup, _ = os.path.splitext(fname)
        fname_bkup += '_b27bkup.blend'
        if os.path.exists(fname_bkup):
            # backup already created
            print("Found extant blender 2.7x backup file, leaving it alone...")
        else:

            msg = ["==============================================",
                   "",
                   "WARNING! If a file is saved with blender 2.8+,",
                   "it cannot be opened with blender 2.7. pycortex is ",
                   "about to open a file created with blender 2.7 in",
                   "a newer version of blender (> 2.8). Would you like",
                   "to create a backup file of the 2.7 version just in",
                   "case? It will be saved as:",
                   f"{fname_bkup}",
                   "",
                   "Y/N:   ",
                   ]
            yn = input('\n'.join(msg))
            if yn.lower()[0] == 'y':
                print("Backing up file...")
                shutil.copy(fname, fname_bkup)


def add_cutdata(fname, braindata, name="retinotopy", projection="nearest", mesh="hemi", blender_path=None):
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
    blender_path = blender_path or default_blender

    if isinstance(braindata, dataset.Dataset):
        for view_name, data in braindata.views.items():
            add_cutdata(fname, data, name=view_name, projection=projection, mesh=mesh)
        return
    braindata = dataset.normalize(braindata)
    if not isinstance(braindata, dataset.braindata.VertexData):
        mapped = braindata.map(projection)
    else:
        mapped = braindata
    left = mapped.left
    right = mapped.right

    cmap = utils.get_cmap(braindata.cmap)
    vmin = braindata.vmin
    vmax = braindata.vmax
    lcolor = cmap((left - vmin) / (vmax - vmin))[:,:3]
    rcolor = cmap((right - vmin) / (vmax - vmin))[:,:3]

    p = xdrlib.Packer()
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
        _call_blender(fname, code, blender_path=blender_path)

    return 


def gii_cut(fname, subject, hemi, blender_path=None):
    '''
    Add gifti surface to blender
    '''
    blender_path = blender_path or default_blender

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
        _call_blender(fname, code, blender_path=blender_path)


def fs_cut_init(fname, subject, hemi, freesurfer_subject_dir=None, blender_path=None):
    """Initialize a blender object from a freesurfer volume.

    Parameters
    ----------
    fname : str
        file path for new .blend file (must end in ".blend")
    subject : str
        subject name
    hemi : str
        hemisphere name (lh or rh)
    freesurfer_subject_dir : str
        path to freesurfer subject directory. If None, it defaults to SUBJECTS_DIR environment variable
    blender_path : str
        path to blender executable. If None, it defaults to the path specified in pycortexconfig file.
    """
    blender_path = blender_path or default_blender
    
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
        _call_blender(fname, code, blender_path=blender_path)


def fs_cut_open(fname, blender_path=None):
    """Open a blender file in blender for the manual cut

    Parameters
    ----------
    fname : str
        file path for blender file (must end in ".blend")
    blender_path : str
        path to blender executable. If None, it defaults to the path specified in pycortexconfig file.
    """
    blender_path = blender_path or default_blender
    
    _call_blender(fname, background=False, blender_path=blender_path)


def write_patch(bname, pname, mesh="hemi", blender_path=default_blender):
    """Deprecated: please use write_volume_patch instead"""
    return write_volume_patch(bname, pname, "hemi", mesh, blender_path)


def write_volume_patch(bname, pname, hemi, mesh="hemi", blender_path=None):
    """Write volume patch in freesurfer format.
    This is a necessary step for flattening the surface in freesurfer

    Parameters
    ----------
    bname : str
        blender file name that contains the mesh
    pname : str
        name of patch file to be saved
    hemi : str
        hemisphere name (lh or rh)
    mesh : str
        name of mesh in blender file
    blender_path : str, optional
        path to blender executable. If None, it defaults to the path specified in pycortexconfig file.
    """
    blender_path = blender_path or default_blender

    p = xdrlib.Packer()
    p.pack_string(pname.encode())
    p.pack_string(hemi.encode())
    p.pack_string(mesh.encode())
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            pname = u.unpack_string().decode('utf-8')
            hemi = u.unpack_string().decode('utf-8')
            mesh = u.unpack_string().decode('utf-8')
            blendlib.write_volume_patch(pname, hemi, mesh)
        """.format(tfname=tf.name)
        _call_blender(bname, code, blender_path=blender_path)
    return True

def write_flat_patch(bname, pname, hemi, mesh="hemi", method="MINIMUM_STRETCH", blender_path=None):
    """Write flat patch in freesurfer format.
    This is a necessary step for flattening the surface in freesurfer

    Parameters
    ----------
    bname : str
        blender file name that contains the mesh
    pname : str
        name of patch file to be saved
    hemi : str
        hemisphere name (lh or rh)
    mesh : str
        name of mesh in blender file
    method : str
        method to use for UV unwrap. One of 'CONFORMAL', 'ANGLE_BASED', 'MINIMUM_STRETCH'.
    blender_path : str, optional
        path to blender executable. If None, it defaults to the path specified in pycortexconfig file.
    """
    blender_path = blender_path or default_blender

    p = xdrlib.Packer()
    p.pack_string(pname.encode())
    p.pack_string(hemi.encode())
    p.pack_string(mesh.encode())
    p.pack_string(method.encode())
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(p.get_buffer())
        tf.flush()
        code = """with open('{tfname}', 'rb') as fp:
            u = xdrlib.Unpacker(fp.read())
            pname = u.unpack_string().decode('utf-8')
            hemi = u.unpack_string().decode('utf-8')
            mesh = u.unpack_string().decode('utf-8')
            method = u.unpack_string().decode('utf-8')
            blendlib.write_flat_patch(pname, hemi, mesh, method)
        """.format(tfname=tf.name)
        _call_blender(bname, code, blender_path=blender_path)
    return True