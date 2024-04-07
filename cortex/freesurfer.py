"""Contains functions for interfacing with freesurfer
"""
from __future__ import print_function

import copy
import os
import shlex
import shutil
import struct
import subprocess as sp
import tempfile
import warnings
from builtins import input
from tempfile import NamedTemporaryFile

import nibabel
import numpy as np
from nibabel import gifti
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

from . import anat, database


def get_paths(fs_subject, hemi, type="patch", freesurfer_subject_dir=None):
    """Retrieve paths for all surfaces for a subject processed by freesurfer

    Parameters
    ----------
    subject : string
        Subject ID for freesurfer subject
    hemi : string ['lh'|'rh']
        Left ('lh') or right ('rh') hemisphere
    type : string ['patch'|'surf'|'curv']
        Which type of files to return
    freesurfer_subject_dir : string | None
        Directory of freesurfer subjects. Defaults to the value for
        the environment variable 'SUBJECTS_DIR' (which should be set
        by freesurfer)
    """
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
    base = os.path.join(freesurfer_subject_dir, fs_subject)
    if type == "patch":
        return os.path.join(base, "surf", hemi+".{name}.patch.3d")
    elif type == "surf":
        return os.path.join(base, "surf", hemi+".{name}")
    elif type == "curv":
        return os.path.join(base, "surf", hemi+".curv{name}")
    elif type == "slim":
        return os.path.join(base, "surf", hemi+".{name}_slim.obj")


def autorecon(fs_subject, type="all", parallel=True, n_cores=None):
    """Run Freesurfer's autorecon-all command for a given freesurfer subject

    Parameters
    ----------
    fs_subject : string
        Freesurfer subject ID (should be a folder in your freesurfer $SUBJECTS_DIR)
    type : string
        Which steps of autorecon-all to perform. {'all', '1','2','3','cp','wm', 'pia'}

    """
    types = {
        'all': 'autorecon-all',
        '1': "autorecon1",
        '2': "autorecon2",
        '3': "autorecon3",
        'cp': "autorecon2-cp",
        'wm': "autorecon2-wm",
        'pia': "autorecon-pial"}

    times = {
        'all': "12 hours",
        '2': "6 hours",
        'cp': "8 hours",
        'wm': "4 hours"
        }
    if str(type) in times:
        resp = input("recon-all will take approximately %s to run! Continue? "%times[str(type)])
        if resp.lower() not in ("yes", "y"):
            return

    cmd = "recon-all -s {subj} -{cmd}".format(subj=fs_subject, cmd=types[str(type)])
    if parallel and type in ('2', 'wm', 'all'):
        # Parallelization only works for autorecon2 or autorecon2-wm
        if n_cores is None:
            import multiprocessing as mp
            n_cores = mp.cpu_count()
        cmd += ' -parallel -openmp {n_cores:d}'.format(n_cores=n_cores)
    print("Calling:\n{cmd}".format(cmd=cmd))
    sp.check_call(shlex.split(cmd))


def flatten(fs_subject, hemi, patch, freesurfer_subject_dir=None, save_every=None):
    """Perform flattening of a brain using freesurfer

    Parameters
    ----------
    fs_subject : str
        Freesurfer subject ID
    hemi : str ['lh' | 'rh']
        hemisphere to flatten
    patch : str
        name for freesurfer patch (used as `name` argument to format output
        of `get_paths()`)
    freesurfer_subject_dir : str
        Freesurfer subjects directory location. None defaults to $SUBJECTS_DIR
    save_every: int
        If not None, this saves a version of the mesh every `save_every` iterations
        of the flattening process. Useful for determining why a flattening fails.

    Returns
    -------

    Notes
    -----
    To look into: link below shows how to give continuous output for a subprocess.
    There maybe indications that a flattening is going badly that we could detect
    in the stdout; perhaps even continuously update a visualization of the generated
    files using segment.show_surface() with the outputs (triggered to update once stdout
    shows that a flattening iteration has completed)
    https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    """
    resp = input('Flattening takes approximately 2 hours! Continue? ')
    if resp.lower() in ('y', 'yes'):
        inpath = get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch)
        outpath = get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch+".flat")
        if save_every is None:
            save_every_str = ''
        else:
            save_every_str = ' -w %d'%save_every
        cmd = "mris_flatten -O fiducial{save_every_str} {inpath} {outpath}".format(inpath=inpath, outpath=outpath, save_every_str=save_every_str)
        print("Calling: ")
        print(cmd)
        sp.check_call(shlex.split(cmd))
        return True
    else:
        print("Not going to flatten...")
        return False


def import_subj(
    freesurfer_subject,
    pycortex_subject=None,
    freesurfer_subject_dir=None,
    whitematter_surf="smoothwm",
):
    """Imports a subject from freesurfer

    This will overwrite (after giving a warning and an option to continue) the 
    pre-existing subject, including all blender cuts, masks, transforms, etc., and
    re-generate surface info files (curvature, sulcal depth, thickness) stored in 
    the surfinfo/ folder for the subject. All cached files for the subject will be 
    deleted. 

    Parameters
    ----------
    freesurfer_subject : str
        Freesurfer subject name
    pycortex_subject : str, optional
        Pycortex subject name. By default it uses the freesurfer subject name. 
        It is advised to stick to that convention, if possible 
        (your life will go more smoothly.)
    freesurfer_subject_dir : str, optional
        Freesurfer subject directory to pull data from. By default uses the directory
        given by the environment variable $SUBJECTS_DIR.
    whitematter_surf : str, optional
        Which whitematter surface to import as 'wm'. By default uses 'smoothwm', but 
        that surface is smoothed and may not be appropriate. 
        A good alternative is 'white'.
    
    Notes
    -----
    This function uses command line functions from freesurfer, so you should make sure
    to have freesurfer sourced before running this function.

    This function will also generate the fiducial surfaces for the subject, which are
    halfway between the white matter and pial surfaces. The surfaces will be stored
    in the freesurfer subject's directory. These fiducial surfaces are used for 
    cutting and flattening.
    """
    # Check if freesurfer is sourced or if subjects dir is passed
    if freesurfer_subject_dir is None:
        if "SUBJECTS_DIR" in os.environ:
            freesurfer_subject_dir = os.environ["SUBJECTS_DIR"]
        else:
            raise ValueError(
                "Please source freesurfer before running this function, "
                "or pass a path to the freesurfer subjects directory in "
                "`freesurfer_subject_dir`"
            )
    fs_mri_path = os.path.join(freesurfer_subject_dir, freesurfer_subject, "mri")
    fs_surf_path = os.path.join(freesurfer_subject_dir, freesurfer_subject, "surf")
    fs_anat_template = os.path.join(fs_mri_path, "{name}.mgz")
    fs_surf_template = os.path.join(fs_surf_path, "{hemi}.{name}")

    # Now deal with pycortex
    if pycortex_subject is None:
        pycortex_subject = freesurfer_subject
    # Create and/or replace extant subject. Throws a warning that this will happen.
    database.db.make_subj(pycortex_subject)

    filestore = os.path.join(database.default_filestore, pycortex_subject)
    anat_template = os.path.join(filestore, "anatomicals", "{name}.nii.gz")
    surf_template = os.path.join(filestore, "surfaces", "{name}_{hemi}.gii")
    surfinfo_template = os.path.join(filestore, "surface-info", "{name}.npz")

    # Dictionary mapping for volumes to be imported over from freesurfer
    volumes_fs2pycortex = {"T1": "raw", "aseg": "aseg", "wm": "raw_wm"}
    # Import volumes
    for fsname, name in volumes_fs2pycortex.items():
        in_volume = fs_anat_template.format(name=fsname)
        out_volume = anat_template.format(name=name)
        cmd = "mri_convert {path} {out}".format(path=in_volume, out=out_volume)
        sp.check_output(shlex.split(cmd))

    # (Re-)Make the fiducial files
    # NOTE: these are IN THE FREESURFER $SUBJECTS_DIR !! which can cause confusion.
    make_fiducial(freesurfer_subject, freesurfer_subject_dir=freesurfer_subject_dir)

    # Dictionary mapping for surfaces to be imported over from freesurfer
    surfaces_fs2pycortex = {
        whitematter_surf: "wm",
        "pial": "pia",
        "inflated": "inflated",
    }
    # Import surfaces
    for fsname, name in surfaces_fs2pycortex.items():
        for hemi in ("lh", "rh"):
            in_surface = fs_surf_template.format(hemi=hemi, name=fsname)
            out_surface = surf_template.format(name=name, hemi=hemi)
            # Use the --to-scanner flag to store the surfaces with the same coordinate
            # system as the volume data, rather than the TKR coordinate system, which
            # has the center set to FOV/2.
            # NOTE: the resulting gifti surfaces will look misaligned with respect to
            # the anatomical volumes when visualized in freeview, because freeview 
            # expects the surfaces to be in TKR coordinates (with center set to FOV/2).
            # But the surfaces stored in the pycortex database are only to be used by
            # pycortex, so that's fine.
            cmd = f"mris_convert --to-scanner {in_surface} {out_surface}"
            sp.check_output(shlex.split(cmd))

    # Dictionary mapping for curvature and extra info to be imported 
    info_fs2pycortex = {
        "sulc": "sulcaldepth",
        "thickness": "thickness",
        "curv": "curvature",
    }
    # Import curvature and extra information
    for fsname, name in info_fs2pycortex.items():
        in_info_lhrh = [
            fs_surf_template.format(hemi=hemi, name=fsname) for hemi in ["lh", "rh"]
        ]
        lh, rh = [parse_curv(in_info) for in_info in in_info_lhrh]
        np.savez(
            surfinfo_template.format(name=name), 
            left=-lh, 
            right=-rh
        )
    # Finally update the database by re-initializing it
    database.db = database.Database()


def import_flat(fs_subject, patch, hemis=['lh', 'rh'], cx_subject=None,
                flat_type='freesurfer', auto_overwrite=False,
                freesurfer_subject_dir=None, clean=True):
    """Imports a flat brain from freesurfer

    NOTE: This will delete the overlays.svg file for this subject, since THE
    FLATMAPS WILL CHANGE, as well as all cached information (e.g. old flatmap 
    boundaries, roi svg intermediate renders, etc). 

    Parameters
    ----------
    fs_subject : str
        Freesurfer subject name
    patch : str
        Name of flat.patch.3d file; e.g., "flattenv01"
    hemis : list
        List of hemispheres to import. Defaults to both hemispheres.
    cx_subject : str
        Pycortex subject name
    freesurfer_subject_dir : str
        directory for freesurfer subjects. None defaults to environment variable
        $SUBJECTS_DIR
    clean : bool
        If True, the flat surface is cleaned to remove the disconnected polys.

    Returns
    -------
    """
    if not auto_overwrite:
        proceed = input(('Warning: This is intended to over-write .gii files storing\n'
                         'flatmap vertex locations for this subject, and will result\n'
                         'in deletion of the overlays.svg file and all cached info\n'
                         'for this subject (because flatmaps will fundamentally change).\n'
                         'Proceed? [y]/n: '))
        if proceed.lower() not in ['y', 'yes', '']:
            print(">>> Elected to quit rather than delete & overwrite files.")
            return

    if cx_subject is None:
        cx_subject = fs_subject
    surfs = os.path.join(database.default_filestore, cx_subject, "surfaces", "flat_{hemi}.gii")

    from . import formats
    for hemi in hemis:
        if flat_type == 'freesurfer':
            pts, polys, _ = get_surf(fs_subject, hemi, "patch", patch+".flat", freesurfer_subject_dir=freesurfer_subject_dir)
            # Reorder axes: X, Y, Z instead of Y, X, Z
            flat = pts[:, [1, 0, 2]]
            # Flip Y axis upside down
            flat[:, 1] = -flat[:, 1]
        elif flat_type == 'slim':
            flat_file = get_paths(fs_subject, hemi, type='slim',
                                  freesurfer_subject_dir=freesurfer_subject_dir)
            flat_file = flat_file.format(name=patch + ".flat")
            flat, polys = formats.read_obj(flat_file)

        if clean:
            polys = _remove_disconnected_polys(polys)
            flat = _move_disconnect_points_to_zero(flat, polys)

        fname = surfs.format(hemi=hemi)
        print("saving to %s"%fname)
        formats.write_gii(fname, pts=flat, polys=polys)

    # clear the cache, per #81
    database.db.clear_cache(cx_subject)
    # Remove overlays.svg file (FLATMAPS HAVE CHANGED)
    overlays_file = database.db.get_paths(cx_subject)['overlays']
    if os.path.exists(overlays_file):
        os.unlink(overlays_file)
    # Regenerate it? 


def _remove_disconnected_polys(polys):
    """Remove polygons that are not in the main connected component.
    
    This function creates a sparse graph based on edges in the input.
    Then it computes the connected components, and returns only the polygons
    that are in the largest component.
    
    This filtering is useful to remove disconnected vertices resulting from a
    poor surface cut.
    """
    n_points = np.max(polys) + 1
    import scipy.sparse as sp

    # create the sparse graph
    row = np.concatenate([
        polys[:, 0], polys[:, 1], polys[:, 0],
        polys[:, 2], polys[:, 1], polys[:, 2]
    ])
    col = np.concatenate([
        polys[:, 1], polys[:, 0], polys[:, 2],
        polys[:, 0], polys[:, 2], polys[:, 1]
    ])
    data = np.ones(len(col), dtype=bool)
    graph = sp.coo_matrix((data, (row, col)), shape=(n_points, n_points),
                          dtype=bool)
    
    # compute connected components
    n_components, labels = sp.csgraph.connected_components(graph)
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_trivial_components = unique_labels[np.where(counts > 1)[0]]
    main_component = unique_labels[np.argmax(counts)]
    extra_components = non_trivial_components[non_trivial_components != main_component]

    # filter all components not in the largest component
    disconnected_pts = np.where(np.isin(labels, extra_components))[0]
    disconnected_polys_mask = np.isin(polys[:, 0], disconnected_pts)
    return polys[~disconnected_polys_mask]


def _move_disconnect_points_to_zero(pts, polys):
    """Change coordinates of points not in polygons to zero.
    
    This cleaning step is useful after _remove_disconnected_polys, to
    avoid using this points in boundaries computations (through pts.max(axis=0)
    here and there).
    """
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.unique(polys)] = True
    pts[~mask] = 0
    return pts


def make_fiducial(fs_subject, freesurfer_subject_dir=None):
    """Make fiducial surface (halfway between white matter and pial surfaces)
    """
    for hemi in ['lh', 'rh']:
        spts, polys, _ = get_surf(fs_subject, hemi, "smoothwm", freesurfer_subject_dir=freesurfer_subject_dir)
        ppts, _, _ = get_surf(fs_subject, hemi, "pial", freesurfer_subject_dir=freesurfer_subject_dir)
        fname = get_paths(fs_subject, hemi, "surf", freesurfer_subject_dir=freesurfer_subject_dir).format(name="fiducial")
        write_surf(fname, (spts + ppts) / 2, polys)


def parse_surf(filename):
    """
    """
    with open(filename, 'rb') as fp:
        #skip magic
        fp.seek(3)
        comment = fp.readline()
        fp.readline()
        print(comment)
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.fromstring(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.fromstring(fp.read(4*3*faces), dtype='i4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)


def write_surf(filename, pts, polys, comment=''):
    """Write freesurfer surface file
    """
    with open(filename, 'wb') as fp:
        fp.write(b'\xff\xff\xfe')
        fp.write((comment+'\n\n').encode())
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap().tostring())
        fp.write(polys.astype(np.uint32).byteswap().tostring())
        fp.write(b'\n')


def write_patch(filename, pts, edges=None):
    """Writes a patch file that is readable by freesurfer.

    Note this function is duplicated here and in blendlib. This function
    writes freesurfer format, so seems natural to place here, but it
    also needs to be called from blender, and the blendlib functions are
    the only ones currently that can easily be called in a running
    blender session.

    Parameters
    ----------
    filename : name for patch to write. Should be of the form
        <subject>.flatten.3d
    pts : array-like
        points in the mesh
    edges : array-like
        edges in the mesh.

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


def parse_curv(filename):
    """
    """
    with open(filename, 'rb') as fp:
        fp.seek(15)
        return np.fromstring(fp.read(), dtype='>f4').byteswap().newbyteorder()


def parse_patch(filename):
    """
    """
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'),
                                               ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data


def get_surf(subject, hemi, type, patch=None, flatten_step=None, freesurfer_subject_dir=None):
    """Read freesurfer surface file
    """
    if type == "patch":
        assert patch is not None
        surf_file = get_paths(subject, hemi, 'surf', freesurfer_subject_dir=freesurfer_subject_dir).format(name='smoothwm')
    else:
        surf_file = get_paths(subject, hemi, 'surf', freesurfer_subject_dir=freesurfer_subject_dir).format(name=type)

    pts, polys = parse_surf(surf_file)

    if patch is not None:
        patch_file = get_paths(subject, hemi, 'patch', freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch)
        if flatten_step is not None:
            patch_file += '%04d'%flatten_step
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

    return pts, polys, get_curv(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir)


def _move_labels(subject, label, hemisphere=('lh','rh'), fs_dir=None, src_subject='fsaverage'):
    """subject is a freesurfer subject"""
    if fs_dir is None:
        fs_dir = os.environ['SUBJECTS_DIR']
    for hemi in hemisphere:
        srclabel = os.path.join(fs_dir, src_subject, 'label',
                                '{hemi}.{label}.label'.format(hemi=hemi, label=label))
        trglabel = os.path.join(fs_dir, subject, 'label',
                                '{hemi}.{label}.label'.format(hemi=hemi, label=label))
        if not os.path.exists(srclabel):
            raise ValueError("Label {} doesn't exist!".format(srclabel))
        fs_sub_dir = os.path.join(fs_dir, subject, 'label')
        if not os.path.exists(fs_sub_dir):
            raise ValueError("Freesurfer subject directory for subject ({}) does not exist!".format(fs_sub_dir))
        cmd = ("mri_label2label --srcsubject {src_subject} --trgsubject {subject} "
               "--srclabel {srclabel} --trglabel {trglabel} "
               "--regmethod surface --hemi {hemi}")
        cmd_f = cmd.format(hemi=hemi, subject=subject, src_subject=src_subject,
                           srclabel=srclabel, trglabel=trglabel)
        print("Calling: ")
        print(cmd_f)
        to_call = shlex.split(cmd_f)
        proc = sp.Popen(to_call,
                           stdin=sp.PIPE,
                           stdout=sp.PIPE,
                           stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        if stderr not in ('', b''):
            raise Exception("Error in freesurfer function call:\n{}".format(stderr))
    print("Labels transferred")


def _parse_labels(label_files, cx_subject):
    """Extract values from freesurfer label file(s) and map to vertices

    Parameters
    ----------
    label_files : str or list
        full paths to label file or files to load
    cx_subject : str
        pycortex subject ID
    """
    if not isinstance(label_files, (list, tuple)):
        label_files = [label_files]
    verts = []
    values = []
    lh_surf, _ = database.db.get_surf(cx_subject, 'fiducial', 'left')
    for fname in label_files:
        with open(fname) as fid:
            lines = fid.readlines()
            lines = [[float(xx.strip()) for xx in x.split(' ') if xx.strip()] for x in lines[2:]]
            vals = np.array(lines)
            if '/lh.' in fname:
                verts.append(vals[:,0])
            elif '/rh.' in fname:
                verts.append(vals[:,0] + lh_surf.shape[0])
            values.append(vals[:,-1])
    verts = np.hstack(verts)
    values = np.hstack(values)
    return verts, values

def get_label(cx_subject, label, fs_subject=None, fs_dir=None, src_subject='fsaverage', hemisphere=('lh', 'rh'), **kwargs):
    """Get data from a label file for fsaverage subject

    Parameters
    ----------
    cx_subject : str
        A pycortex subject ID
    label : str
        Label name
    fs_subject : str
        Freesurfer subject ID, if different from pycortex subject ID
    src_subject : str
        Freesurfer subject ID from which to transfer the label.
    fs_dir : str
        Freesurfer subject directory; None defaults to OS environment variable
    hemisphere : list | tuple

    """
    if fs_dir is None:
        fs_dir = os.environ['SUBJECTS_DIR']
    else:
        os.environ['SUBJECTS_DIR'] = fs_dir
    if fs_subject is None:
        fs_subject = cx_subject
    label_files = [os.path.join(fs_dir, fs_subject, 'label', '{}.{}.label'.format(h, label)) for h in hemisphere]
    if cx_subject not in ['fsaverage', 'MNI']:
        # If label file doesn't exist, try to move it there
        print('looking for {}'.format(label_files))
        if not all([os.path.exists(f) for f in label_files]):
            print("Transforming label file to subject's freesurfer directory...")
            _move_labels(fs_subject, label, hemisphere=hemisphere, fs_dir=fs_dir, src_subject=src_subject)
    verts, values = _parse_labels(label_files, cx_subject)
    idx = verts.astype(int)
    return idx, values


def _mri_surf2surf_command(src_subj, trg_subj, input_file, output_file, hemi):
    # mri_surf2surf --srcsubject <source subject name> --srcsurfval
    # <sourcefile> --trgsubject <target suhject name> --trgsurfval <target
    # file> --hemi <hemifield>

    cmd = ["mri_surf2surf", "--srcsubject", src_subj,
                            "--sval", input_file,
                            "--trgsubject", trg_subj,
                            "--tval", output_file,
                            "--hemi", hemi,
          ]
    return cmd


def _check_datatype(data):
    dtype = data.dtype
    if dtype == np.int64:
        return np.int32
    elif dtype == np.float64:
        return np.float32
    else:
        return dtype


def mri_surf2surf(data, source_subj, target_subj, hemi, subjects_dir=None):
    """Uses freesurfer mri_surf2surf to transfer vertex data between
        two freesurfer subjects
    
    Parameters
    ==========
    data: ndarray, shape=(n_imgs, n_verts)
        data arrays representing vertex data
    
    source_subj: str
        freesurfer subject name of source subject
    
    target_subj: str
        freesurfer subject name of target subject
    
    hemi: str in ("lh", "rh")
        string indicating hemisphere.
    
    Notes
    =====
    Requires path to mri_surf2surf or freesurfer environment to be active.
    """
    datatype = _check_datatype(data)
    data_arrays = [gifti.GiftiDataArray(d, datatype=datatype) for d in data]
    gifti_image = gifti.GiftiImage(darrays=data_arrays)

    tf_in = NamedTemporaryFile(suffix=".gii")
    nibabel.save(gifti_image, tf_in.name)

    tf_out = NamedTemporaryFile(suffix='.gii')
    cmd = _mri_surf2surf_command(source_subj, target_subj,
                                   tf_in.name, tf_out.name, hemi)
    if subjects_dir is not None:
        env = os.environ.copy()
        env['SUBJECTS_DIR'] = subjects_dir
    else:
        env = None

    print('Calling:')
    print(' '.join(cmd))
    p = sp.Popen(cmd, env=env)
    exit_code = p.wait()
    if exit_code != 0:
        if exit_code == 255:
            raise Exception(("Missing file (see above). "
                             "If lh.sphere.reg is missing,\n"
                             "you likely need to run the 3rd "
                             "stage of freesurfer autorecon\n"
                             "(sphere registration) for this subject:\n"
                             ">>> cortex.freesurfer.autorecon('{fs_subject}', type='3')"
                             ).format(fs_subject=source_subj))
        #from subprocess import CalledProcessError # handle with this, maybe?
        raise Exception(("Exit code {exit_code} means that "
            "mri_surf2surf failed").format(exit_code=exit_code))

    tf_in.close()
    output_img = nibabel.load(tf_out.name)
    output_data = np.array([da.data for da in output_img.darrays])
    tf_out.close()
    return output_data


def get_mri_surf2surf_matrix(source_subj, hemi, surface_type,
                            target_subj='fsaverage', subjects_dir=None,
                            n_neighbors=20, random_state=0,
                            n_test_images=40, coef_threshold=None,
                            renormalize=True):

    """Creates a matrix implementing freesurfer mri_surf2surf command.
    
    A surface-to-surface transform is a linear transform between vertex spaces.
    Such a transform must be highly localized in the sense that a vertex in the
    target surface only draws its values from very few source vertices.
    This function exploits the localization to create an inverse problem for 
    each vertex.
    The source neighborhoods for each target vertex are found by using
    mri_surf2surf to transform the three coordinate maps from the source 
    surface to the target surface, yielding three coordinate values for each
    target vertex, for which we find the nearest neighbors in the source space.
    A small number of test images is transformed from source surface to
    target surface.	
    For each target vertex in the transformed test images, a regression is 
    performed using only the corresponding source image neighborhood, yielding
    the entries for a sparse matrix encoding the transform.
    
    Parameters
    ==========
    
    source_subj: str
    	Freesurfer name of source subject
    
    hemi: str in ("lh", "rh")
    	Indicator for hemisphere
    
    surface_type: str in ("white", "pial", ...)
    	Indicator for surface layer
    
    target_subj: str, default "fsaverage"
    	Freesurfer name of target subject
    
    subjects_dir: str, default os.environ["SUBJECTS_DIR"]
    	The freesurfer subjects directory
    
    n_neighbors: int, default 20
    	The size of the neighborhood to take into account when estimating
    	the source support of a vertex
    
    random_state: int, default 0
    	Random number generator or seed for generating test images
    
    n_test_images: int, default 40
    	Number of test images transformed to compute inverse problem. This 
    	should be greater than n_neighbors or equal.
    
    coef_treshold: float, default 1 / (10 * n_neighbors)
    	Value under which to set a weight to zero in the inverse problem.
    
    renormalize: boolean, default True
    	Determines whether the rows of the output matrix should add to 1,
    	implementing what is sensible: a weighted averaging
    
    Notes
    =====
    It turns out that freesurfer seems to do the following: For each target
    vertex, find, on the sphere, the nearest source vertices, and average their
    values. Try to be as one-to-one as possible.
    """

    source_verts, _, _ = get_surf(source_subj, hemi, surface_type,
                                  freesurfer_subject_dir=subjects_dir)

    transformed_coords = mri_surf2surf(source_verts.T,
                                       source_subj, target_subj, hemi,
                                       subjects_dir=subjects_dir)

    kdt = KDTree(source_verts)
    print("Getting nearest neighbors")
    distances, indices = kdt.query(transformed_coords.T, k=n_neighbors)
    print("Done")

    rng = (np.random.RandomState(random_state) 
                          if isinstance(random_state, int) else random_state)
    test_images = rng.randn(n_test_images, len(source_verts))
    transformed_test_images = mri_surf2surf(test_images, source_subj,
                                            target_subj, hemi,
                                            subjects_dir=subjects_dir)

    # Solve linear problems to get coefficients
    all_coefs = []
    residuals = []
    print("Computing coefficients")
    i = 0
    for target_activation, source_inds in zip(
                                        transformed_test_images.T, indices):
        i += 1
        print("{i}".format(i=i), end="\r")
        source_values = test_images[:, source_inds]
        r = lstsq(source_values, target_activation,
                 overwrite_a=True, overwrite_b=True)
        all_coefs.append(r[0])
        residuals.append(r[1])
    print("Done")

    all_coefs = np.array(all_coefs)

    if coef_threshold is None:  # we know now that coefs are doing averages
        coef_threshold = (1 / 10. / n_neighbors )
    all_coefs[np.abs(all_coefs) < coef_threshold] = 0
    if renormalize:
        all_coefs /= np.abs(all_coefs).sum(axis=1)[:, np.newaxis] + 1e-10

    # there seem to be like 7 vertices that don't constitute an average over
    # 20 vertices or less, but all the others are such an average.

    # Let's make a matrix that does the transform:
    col_indices = indices.ravel()
    row_indices = (np.arange(indices.shape[0])[:, np.newaxis] *
                   np.ones(indices.shape[1], dtype='int')).ravel()
    data = all_coefs.ravel()
    shape = (transformed_coords.shape[1], source_verts.shape[0])

    matrix = coo_matrix((data, (row_indices, col_indices)), shape=shape)

    return matrix


def get_curv(fs_subject, hemi, type='wm', freesurfer_subject_dir=None):
    """Load freesurfer curv file for a freesurfer subject

    Parameters
    ----------
    fs_subject : str
        freesurfer subject identifier
    hemi : str
        'lh' or 'rh' for left or right hemisphere, respectively
    type : str
        'wm' or other type of surface (e.g. 'fiducial' or 'pial')
    freesurfer_subject_dir : str
        directory for Freesurfer subjects (defaults to value for the 
        environment variable $SUBJECTS_DIR if None)
    """
    if type == "wm":
        curv_file = get_paths(fs_subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='')
    else:
        curv_file = get_paths(fs_subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='.'+type)

    return parse_curv(curv_file)


def show_surf(subject, hemi, type, patch=None, curv=True, freesurfer_subject_dir=None):
    """Show a surface from a Freesurfer subject directory

    Parameters
    ----------
    subject : str
        Freesurfer subject name
    hemi : str ['lh' | 'rh']
        Left or right hemisphere
    type :

    patch :

    curv : bool

    freesurfer_subject_dir :
    """
    warnings.warn(('Deprecated and probably broken! Try `cortex.segment.show_surf()`\n'
                  'which uses a third-party program (meshlab, available for linux & mac\n'
                  'instead of buggy mayavi code!'))
    from mayavi import mlab
    from tvtk.api import tvtk

    pts, polys, idx = get_surf(subject, hemi, type, patch, freesurfer_subject_dir=freesurfer_subject_dir)
    if curv:
        curv = get_curv(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir)
    else:
        curv = idx

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

    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)
    def picker_callback(picker):
        if picker.actor in surf.actor.actors:
            npts = np.append(cursors.data.points.to_array(), [pts[picker.point_id]], axis=0)
            cursors.data.points = npts
            print(picker.point_id)
            x, y, z = pts[picker.point_id]
            with open(os.path.join(path, 'tmp', 'edit.dat'), 'w') as fp:
                fp.write('%f %f %f\n'%(x, y, z))

    picker = fig.on_mouse_pick(picker_callback)
    picker.tolerance = 0.01
    mlab.show()
    return fig, surf

def write_dot(fname, pts, polys, name="test"):
    """
    """
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
    """
    """
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
    """
    """
    from .polyutils import boundary_edges, decimate
    dpts, dpolys = decimate(pts, polys)
    write_surf(path+'.smoothwm', dpts, dpolys)
    edges = boundary_edges(dpolys)
    data = np.zeros((len(dpts),), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    data['vert'] = np.arange(len(dpts))+1
    data['vert'][edges] *= -1
    data['x'] = dpts[:,0]
    data['y'] = dpts[:,1]
    data['z'] = dpts[:,2]
    with open(path+'.full.patch.3d', 'w') as fp:
        fp.write(struct.pack('>i', -1))
        fp.write(struct.pack('>i', len(dpts)))
        fp.write(data.tostring())


class SpringLayout(object):
    """
    """
    def __init__(self, pts, polys, dpts=None, pins=None, stepsize=1, neighborhood=0):
        self.pts = pts
        self.polys = polys
        self.stepsize = stepsize
        pinmask = np.zeros((len(pts),), dtype=bool)
        if isinstance(pins, (list, set, np.ndarray)):
            pinmask[pins] = True
        self.pins = pinmask
        self.neighbors = [set() for _ in range(len(pts))]

        for i, j, k in polys:
            self.neighbors[i].add(j)
            self.neighbors[i].add(k)
            self.neighbors[j].add(i)
            self.neighbors[j].add(k)
            self.neighbors[k].add(i)
            self.neighbors[k].add(j)

        for _ in range(neighborhood):
            _neighbors = copy.deepcopy(self.neighbors)
            for v, neighbors in enumerate(self.neighbors):
                for n in neighbors:
                    _neighbors[v] |= self.neighbors[n]
            self.neighbors = _neighbors

        for i in range(len(self.neighbors)):
            self.neighbors[i] = list(self.neighbors[i] - set([i]))

        if dpts is None:
            dpts = pts

        #self.kdt = cKDTree(self.pts)
        self._next = self.pts.copy()

        width = max(len(n) for n in self.neighbors)
        self._mask = np.zeros((len(pts), width), dtype=bool)
        self._move = np.zeros((len(pts), width, 3))
        #self._mean = np.zeros((len(pts), width))
        self._num = np.zeros((len(pts),))
        self._dists = []
        self._idx = []
        for i, n in enumerate(self.neighbors):
            self._mask[i, :len(n)] = True
            self._dists.append(np.sqrt(((dpts[n] - dpts[i])**2).sum(-1)))
            self._idx.append(np.ones((len(n),))*i)
            self._num[i] = len(n)
        self._dists = np.hstack(self._dists)
        self._idx = np.hstack(self._idx).astype(np.uint)
        self._neigh = np.hstack(self.neighbors).astype(np.uint)
        self.figure = None

    def _spring(self):
        svec = self.pts[self._neigh] - self.pts[self._idx]
        slen = np.sqrt((svec**2).sum(-1))
        force = (slen - self._dists) # / self._dists
        svec /= slen[:,np.newaxis]
        fvec = force[:, np.newaxis] * svec
        self._move[self._mask] = self.stepsize * fvec
        return self._move.sum(1) / self._num[:, np.newaxis]

    def _estatic(self, idx):
        dist, neighbors = self.kdt.query(self.pts[idx], k=20)
        valid = dist > 0
        mag = self.stepsize * (1 / dist)
        diff = self.pts[neighbors] - self.pts[idx]
        return (mag[valid] * diff[valid].T).T.mean(0)

    def step(self):
        move = self._spring()[~self.pins]
        self._next[~self.pins] += move #+ self._estatic(i)
        self.pts = self._next.copy()
        return dict(x=self.pts[:,0],y=self.pts[:, 1], z=self.pts[:,2]), move
        #self.kdt = cKDTree(self.pts)

    def run(self, n=1000):
        for _ in range(n):
            self.step()
            print(_)

    def view_step(self):
        from mayavi import mlab
        if self.figure is None:
            self.figure = mlab.triangular_mesh(self.pts[:,0], self.pts[:,1], self.pts[:,2], self.polys, representation='wireframe')
        self.step()
        self.figure.mlab_source.set(x=self.pts[:,0], y=self.pts[:,1], z=self.pts[:,2])

def stretch_mwall(pts, polys, mwall):
    """
    """
    inflated = pts.copy()
    center = pts[mwall].mean(0)
    radius = max((pts.max(0) - pts.min(0))[1:])
    angles = np.arctan2(pts[mwall][:,2], pts[mwall][:,1])
    pts[mwall, 0] = center[0]
    pts[mwall, 1] = radius * np.cos(angles) + center[1]
    pts[mwall, 2] = radius * np.sin(angles) + center[2]
    return SpringLayout(pts, polys, inflated, pins=mwall)


def upsample_to_fsaverage(
        data, data_space="fsaverage6", freesurfer_subjects_dir=None
):
    """Project data from fsaverage6 (or other fsaverage surface) to fsaverage to
    visualize it in pycortex.

    Parameters
    ----------
    data : array (n_samples, n_vertices)
        Data in space `space`. The first n_vertices/2 vertices correspond to the left
        hemisphere, and the last n_vertices/2 vertices correspond to the right
        hemisphere.
    data_space : str
        One of fsaverage[1-6], corresponding to the source template space of `data`.
    freesurfer_subjects_dir : str or None
        Path to Freesurfer subjects directory. If None, defaults to the value of the
        environment variable $SUBJECTS_DIR.

    Returns
    -------
    projected_data : array (n_samples, 327684)
        Data projected to fsaverage(7).

    Notes
    -----
    Data in the lower resolution fsaverage template is upsampled to the full resolution
    fsaverage template by nearest-neighbor interpolation. To project the data from a 
    lower resolution version of fsaverage, this code exploits the structure of fsaverage 
    surfaces. (That is, each hemisphere in fsaverage6 corresponds to the first 
    40,962 vertices of fsaverage; fsaverage5 corresponds to the first 10,242 vertices of 
    fsaverage, etc.)
    """


    def get_n_vertices_ico(icoorder):
        return 4 ** icoorder * 10 + 2

    ico_order = int(data_space[-1])
    n_ico_vertices = get_n_vertices_ico(ico_order)
    ndim = data.ndim
    data = np.atleast_2d(data)
    _, n_vertices = data.shape
    if n_vertices != 2 * n_ico_vertices:
        raise ValueError(
            f"data has {n_vertices} vertices, but {2 * n_ico_vertices} "
            f"are expected for both hemispheres in {data_space}"
        )

    if freesurfer_subjects_dir is None:
        freesurfer_subjects_dir = os.environ.get("SUBJECTS_DIR", None)
    if freesurfer_subjects_dir is None:
        raise ValueError(
            "freesurfer_subjects_dir must be specified or $SUBJECTS_DIR must be set"
        )
    
    data_hemi = np.split(data, 2, axis=-1)
    hemis = ["lh", "rh"]
    projected_data = []
    for i, (hemi, dt) in enumerate(zip(hemis, data_hemi)):
        # Load fsaverage sphere for this hemisphere
        pts, faces = nibabel.freesurfer.read_geometry(
            os.path.join(
                freesurfer_subjects_dir, "fsaverage", "surf", f"{hemi}.sphere.reg"
            )
        )
        # build kdtree using only vertices in reduced fsaverage surface
        kdtree = KDTree(pts[:n_ico_vertices])
        # figure out neighbors in reduced version for all other vertices in fsaverage
        _, neighbors = kdtree.query(pts[n_ico_vertices:], k=1)
        # now simply fill remaining vertices with original values
        projected_data.append(
            np.concatenate([dt, dt[:, neighbors]], axis=-1)
        )
    projected_data = np.hstack(projected_data)
    if ndim == 1:
        projected_data = projected_data[0]
    return projected_data


# aseg partition labels (up to 256 only)
fs_aseg_dict = {'Unknown': 0,
                'Left-Cerebral-Exterior': 1,
                'Left-Cerebral-White-Matter': 2,
                'Left-Cerebral-Cortex': 3,
                'Left-Lateral-Ventricle': 4,
                'Left-Inf-Lat-Vent': 5,
                'Left-Cerebellum-Exterior': 6,
                'Left-Cerebellum-White-Matter': 7,
                'Left-Cerebellum-Cortex': 8,
                'Left-Thalamus': 9,
                'Left-Thalamus-Proper': 10,
                'Left-Caudate': 11,
                'Left-Putamen': 12,
                'Left-Pallidum': 13,
                '3rd-Ventricle': 14,
                '4th-Ventricle': 15,
                'Brain-Stem': 16,
                'Left-Hippocampus': 17,
                'Left-Amygdala': 18,
                'Left-Insula': 19,
                'Left-Operculum': 20,
                'Line-1': 21,
                'Line-2': 22,
                'Line-3': 23,
                'CSF': 24,
                'Left-Lesion': 25,
                'Left-Accumbens-area': 26,
                'Left-Substancia-Nigra': 27,
                'Left-VentralDC': 28,
                'Left-undetermined': 29,
                'Left-vessel': 30,
                'Left-choroid-plexus': 31,
                'Left-F3orb': 32,
                'Left-lOg': 33,
                'Left-aOg': 34,
                'Left-mOg': 35,
                'Left-pOg': 36,
                'Left-Stellate': 37,
                'Left-Porg': 38,
                'Left-Aorg': 39,
                'Right-Cerebral-Exterior': 40,
                'Right-Cerebral-White-Matter': 41,
                'Right-Cerebral-Cortex': 42,
                'Right-Lateral-Ventricle': 43,
                'Right-Inf-Lat-Vent': 44,
                'Right-Cerebellum-Exterior': 45,
                'Right-Cerebellum-White-Matter': 46,
                'Right-Cerebellum-Cortex': 47,
                'Right-Thalamus': 48,
                'Right-Thalamus-Proper': 49,
                'Right-Caudate': 50,
                'Right-Putamen': 51,
                'Right-Pallidum': 52,
                'Right-Hippocampus': 53,
                'Right-Amygdala': 54,
                'Right-Insula': 55,
                'Right-Operculum': 56,
                'Right-Lesion': 57,
                'Right-Accumbens-area': 58,
                'Right-Substancia-Nigra': 59,
                'Right-VentralDC': 60,
                'Right-undetermined': 61,
                'Right-vessel': 62,
                'Right-choroid-plexus': 63,
                'Right-F3orb': 64,
                'Right-lOg': 65,
                'Right-aOg': 66,
                'Right-mOg': 67,
                'Right-pOg': 68,
                'Right-Stellate': 69,
                'Right-Porg': 70,
                'Right-Aorg': 71,
                '5th-Ventricle': 72,
                'Left-Interior': 73,
                'Right-Interior': 74,
                'Left-Lateral-Ventricles': 75,
                'Right-Lateral-Ventricles': 76,
                'WM-hypointensities': 77,
                'Left-WM-hypointensities': 78,
                'Right-WM-hypointensities': 79,
                'non-WM-hypointensities': 80,
                'Left-non-WM-hypointensities': 81,
                'Right-non-WM-hypointensities': 82,
                'Left-F1': 83,
                'Right-F1': 84,
                'Optic-Chiasm': 85,
                'Corpus_Callosum': 86,
                'Left-Amygdala-Anterior': 96,
                'Right-Amygdala-Anterior': 97,
                'Dura': 98,
                'Left-wm-intensity-abnormality': 100,
                'Left-caudate-intensity-abnormality': 101,
                'Left-putamen-intensity-abnormality': 102,
                'Left-accumbens-intensity-abnormality': 103,
                'Left-pallidum-intensity-abnormality': 104,
                'Left-amygdala-intensity-abnormality': 105,
                'Left-hippocampus-intensity-abnormality': 106,
                'Left-thalamus-intensity-abnormality': 107,
                'Left-VDC-intensity-abnormality': 108,
                'Right-wm-intensity-abnormality': 109,
                'Right-caudate-intensity-abnormality': 110,
                'Right-putamen-intensity-abnormality': 111,
                'Right-accumbens-intensity-abnormality': 112,
                'Right-pallidum-intensity-abnormality': 113,
                'Right-amygdala-intensity-abnormality': 114,
                'Right-hippocampus-intensity-abnormality': 115,
                'Right-thalamus-intensity-abnormality': 116,
                'Right-VDC-intensity-abnormality': 117,
                'Epidermis': 118,
                'Conn-Tissue': 119,
                'SC-Fat/Muscle': 120,
                'Cranium': 121,
                'CSF-SA': 122,
                'Muscle': 123,
                'Ear': 124,
                'Adipose': 125,
                'Spinal-Cord': 126,
                'Soft-Tissue': 127,
                'Nerve': 128,
                'Bone': 129,
                'Air': 130,
                'Orbital-Fat': 131,
                'Tongue': 132,
                'Nasal-Structures': 133,
                'Globe': 134,
                'Teeth': 135,
                'Left-Caudate/Putamen': 136,
                'Right-Caudate/Putamen': 137,
                'Left-Claustrum': 138,
                'Right-Claustrum': 139,
                'Cornea': 140,
                'Diploe': 142,
                'Vitreous-Humor': 143,
                'Lens': 144,
                'Aqueous-Humor': 145,
                'Outer-Table': 146,
                'Inner-Table': 147,
                'Periosteum': 148,
                'Endosteum': 149,
                'R/C/S': 150,
                'Iris': 151,
                'SC-Adipose/Muscle': 152,
                'SC-Tissue': 153,
                'Orbital-Adipose': 154,
                'Left-IntCapsule-Ant': 155,
                'Right-IntCapsule-Ant': 156,
                'Left-IntCapsule-Pos': 157,
                'Right-IntCapsule-Pos': 158,
                'Left-Cerebral-WM-unmyelinated': 159,
                'Right-Cerebral-WM-unmyelinated': 160,
                'Left-Cerebral-WM-myelinated': 161,
                'Right-Cerebral-WM-myelinated': 162,
                'Left-Subcortical-Gray-Matter': 163,
                'Right-Subcortical-Gray-Matter': 164,
                'Skull': 165,
                'Posterior-fossa': 166,
                'Scalp': 167,
                'Hematoma': 168,
                'Left-Cortical-Dysplasia': 180,
                'Right-Cortical-Dysplasia': 181,
                'Left-hippocampal_fissure': 193,
                'Left-CADG-head': 194,
                'Left-subiculum': 195,
                'Left-fimbria': 196,
                'Right-hippocampal_fissure': 197,
                'Right-CADG-head': 198,
                'Right-subiculum': 199,
                'Right-fimbria': 200,
                'alveus': 201,
                'perforant_pathway': 202,
                'parasubiculum': 203,
                'presubiculum': 204,
                'subiculum': 205,
                'CA1': 206,
                'CA2': 207,
                'CA3': 208,
                'CA4': 209,
                'GC-DG': 210,
                'HATA': 211,
                'fimbria': 212,
                'lateral_ventricle': 213,
                'molecular_layer_HP': 214,
                'hippocampal_fissure': 215,
                'entorhinal_cortex': 216,
                'molecular_layer_subiculum': 217,
                'Amygdala': 218,
                'Cerebral_White_Matter': 219,
                'Cerebral_Cortex': 220,
                'Inf_Lat_Vent': 221,
                'Perirhinal': 222,
                'Cerebral_White_Matter_Edge': 223,
                'Background': 224,
                'Ectorhinal': 225,
                'Fornix': 250,
                'CC_Posterior': 251,
                'CC_Mid_Posterior': 252,
                'CC_Central': 253,
                'CC_Mid_Anterior': 254,
                'CC_Anterior': 255}

if __name__ == "__main__":
    import sys
    show_surf(sys.argv[1], sys.argv[2], sys.argv[3])
