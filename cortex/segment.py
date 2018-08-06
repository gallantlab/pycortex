"""Controls functions for segmentation of white/gray matter and other things in the brain.
"""
import os
import time
import shlex
import numpy as np
import subprocess as sp
from builtins import input
import multiprocessing as mp

from . import formats
from . import blender
from . import freesurfer
from . import options
from .database import db

from .freesurfer import autorecon as run_freesurfer_recon
from .freesurfer import import_subj as import_freesurfer_subject

slim_path = options.config.get('dependency_paths', 'slim')

def init_subject(subject, filenames, run_all=False):
    """Run the first initial segmentation for a subject's anatomy. 

    This function runs autorecon-all, then (optionally) imports the subject
    into the pycortex database.

    Parameters
    ----------
    subject : str
        The name of the subject (this subject is created in the Freesurfer
        SUBJECTS_DIR)
    filenames : str | list
        Freesurfer-compatible filename(s) for the anatomical image(s). This can
        be the first dicom file of a series of dicoms, a nifti file, an mgz
        file, etc.
    run_all : bool
        Whether to run recon-all all the way through to importing the subject
        into pycortex. False by default, since we recommend editing (or at
        least inspecting) the brain mask and white matter segmentations prior
        to importing into pycortex.
    """
    cmd = "recon-all -i {fname} -s {subj}".format(subj=subject, fname=filename)
    print("Calling:\n%{}".format(cmd))
    sp.call(shlex.split(cmd))
    if run_all:
        run_freesurfer_recon(subject, "all")
        import_freesurfer_subject(subject)


def edit_segmentation(subject,
                      volumes=('brain.mgz', 'aseg.mgz', 'brainmask.mgz', 'wm.mgz'),
                      surfaces=('lh.smoothwm', 'rh.smoothwm', 'lh.pial', 'rh.pial'),
                      freesurfer_subject_dir=None):
    """Edit automatic segmentation results using freeview

    Opens an instance of freeview with relevant files loaded.

    Parameters
    ----------
    subject : str
        freesurfer subject identifier. Note that subject must be in your
        SUBJECTS_DIR for freesurfer. If the environment variable SUBJECTS_DIR
        is not set in your shell, then the location of the directory must be
        specified in `freesurfer_subject_dir`.
    volumes : tuple | list
        Names of volumes to load in freeview
    surfaces : tuple | list
        Names of surfaces to load in freeview
    freesurfer_subject_dir : str | None
        Location of freesurfer subjects directory. If None, defaults to value
        of SUBJECTS_DIR environment variable.

    """
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
    cmaps = {'brain': 'grayscale',
             'aseg': 'lut',
             'brainmask': 'heat',
             'wm': 'heat',
             'smoothwm': 'yellow',
             'pial': 'red'
             }
    opacity={'brain': 1.0,
             'aseg': 0.4,
             'brainmask': 0.4,
             'wm': 0.4,
             }
    vols = []
    for v in volumes:
        vpath = os.path.join(freesurfer_subject_dir, subject, 'mri', v)
        vv, _ = os.path.splitext(v)
        vextra = ':colormap={cm}:opacity={op:0.2f}'.format(cm=cmaps[vv], op=opacity[vv])
        vols.append(vpath + vextra)
    surfs = []
    for s in surfaces:
        spath = os.path.join(freesurfer_subject_dir, subject, 'surf', s)
        _, ss = s.split('.')
        sextra = ':edgecolor={col}'.format(col=cmaps[ss])
        surfs.append(spath + sextra)
    cmd = ["freeview", '-v'] + vols + ['-f'] + surfs
    print("Calling: {}".format(' '.join(cmd)))
    sp.call(cmd)
    disp = ("If you have edited the white matter surface, you should run:",
            "`run_freesurfer_recon(%s, 'wm'\n",
            "If you have edited the brainmask (pial surface), you should run:")


def flatten_slim(subject, hemi, patch, freesurfer_subject_dir=None,
                 slim_path=slim_path):
    """Flatten brain w/ slim object flattening

    Parameters
    ----------
    subject : str
        freesurfer subject
    hemi : str
        'lh' or 'rh' for left or right hemisphere
    patch : str
        name of patch, often "flatten"
    freesurfer_subject_dir : str
        path to freesurfer subejct dir. Defaults to environment variable
        SUBJECTS_DIR
    slim_path : str
        path to SLIM flattening. Defaults to path specified in config file.
    """
    if slim_path == 'None':
        raise ValueError("Please download SLIM (%s) and set the path to it in the `slim` field\n"
                         "in the `dependency_paths` section of your config file (%s) \n"
                         "if you wish to use slim!"%)
    resp = input('Flattening with SLIM will take a few mins. Continue? (type y or n and press return)')
    if not resp.lower() in ('y', 'yes'): 
        print("Not flattening...")
        return
    # File paths
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']    
    surfpath = os.path.join(freesurfer_subject_dir, subject, "surf", "flat_{hemi}.gii")
    patchpath = freesurfer.get_paths(subject, hemi, 
                                     freesurfer_subject_dir=freesurfer_subject_dir)
    patchpath = patchpath.format(name=patch)
    obj_in = patchpath.replace('.patch.3d', '.obj')
    obj_out = obj_in.replace('.obj', '_slim.obj')

    # Load freesurfer surface exported from blender
    pts, polys, _ = freesurfer.get_surf(subject, hemi, "patch", patch=patch, freesurfer_subject_dir=freesurfer_subject_dir)
    # Cull pts that are not in manifold
    pi = np.arange(len(pts))
    pii = np.in1d(pi, polys.flatten())
    idx = np.nonzero(pii)[0]
    pts_new = pts[idx]
    # Match indices in polys to new index for pts
    polys_new = np.vstack([np.searchsorted(idx, p) for p in polys.T]).T
    # save out obj file
    print("Writing input to SLIM: %s"%obj_in)
    formats.write_obj(obj_in, pts_new, polys_new)
    # Call slim to write new obj file

    print('Flattening with SLIM (will take a few minutes)...')
    out = sp.check_output([slim_path, obj_in, obj_out])
    print("SLIM code wrote %s"%obj_out)
    # Load resulting obj file
    _, _, _, uv = formats.read_obj(obj_out, uv=True)
    uv = np.array(uv)
    # Re-center UV & scale to match scale of inflated brain. It is necessary
    # to re-scale the uv coordinates generated by SLIM, since they have 
    # arbitrary units that don't match the scale of the inflated /
    # fiducial brains.    
    uv -= uv.min(0)
    uv /= uv.max()
    uv -= (uv.max(0) / 2)
    infl_scale = np.max(np.abs(pts_new.min(0)-pts_new.max(0)))
    # This is a magic number based on the approximate scale of the flatmap
    # (created by freesurfer) to the inflated map in a couple other subjects.
    # For two hemispheres in two other subjects, it ranged from 1.37 to 1.5.
    # There doesn't seem to be a principled way to set this number, since the
    # flatmap is stretched and distorted anyway, and that stretch varies by
    # subject and by hemisphere. Note, tho,that  this doesn't change
    # distortions, just the overall scale of the thing. So here we are.
    # ML 2018.07.05
    extra_scale = 1.4
    uv *= (infl_scale * extra_scale)
    # put back polys, etc that were missing
    pts_flat = pts.copy()
    pts_flat[idx, :2] = uv
    # Set z coords for the manifold vertices to 0
    pts_flat[idx, 2] = 0
    # Re-set scale for non-manifold vertices
    nz = pts_flat[:, 2] != 0
    pts_flat[nz, 2] -= np.mean(pts_flat[nz, 2])    
    # Flip X axis for right hem (necessary?)
    if hemi=='rh':
        # Flip Y axis upside down
        pts_flat[:, 1] = -pts_flat[:, 1]
        pts_flat[:, 0] = -pts_flat[:, 0]
    # Save out gii file in freesurfer directory    
    fname = surfpath.format(hemi=hemi)
    print("Writing %s"%fname)
    formats.write_gii(fname, pts=pts_flat, polys=polys)
    return

def cut_surface(cx_subject, hemi, name='flatten', fs_subject=None, data=None, 
                freesurfer_subject_dir=None, flatten_with='freesurfer'):
    """Initializes an interface to cut the segmented surface for flatmapping.
    This function creates or opens a blend file in your filestore which allows
    surfaces to be cut along hand-defined seams. Blender will automatically 
    open the file. After edits are made, remember to save the file, then exit
    Blender.

    The surface will be automatically extracted from blender then run through
    the mris_flatten command in freesurfer. The flatmap will be imported once
    that command finishes.

    Parameters
    ----------
    cx_subject : str
        Name of the subject to edit (pycortex subject ID)
    hemi : str
        Which hemisphere to flatten. Should be "lh" or "rh"
    name : str, optional
        String name of the current flatten attempt. Defaults to "flatten"
    data : Dataview
        A data view object to display on the surface as a cutting guide.
    fs_subject : str
        Name of Freesurfer subject (if different from pycortex subject)
        None defaults to `cx_subject`
    freesurfer_subject_dir : str
        Name of Freesurfer subject directory. None defaults to SUBJECTS_DIR 
        environment varible
    """
    if fs_subject is None:
        fs_subject = cx_subject
    opts = "[hemi=%s,name=%s]"%(hemi, name)
    fname = db.get_paths(cx_subject)['anats'].format(type='cutsurf', opts=opts, ext='blend')
    if not os.path.exists(fname):
        blender.fs_cut(fname, fs_subject, hemi, freesurfer_subject_dir)
    # Add localizer data to facilitate cutting
    if data is not None:
        blender.add_cutdata(fname, data, name=data.description)
    blender_cmd = options.config.get('dependency_paths', 'blender')
    sp.call([blender_cmd, fname])
    patchpath = freesurfer.get_paths(fs_subject, hemi,
                                     freesurfer_subject_dir=freesurfer_subject_dir)
    patchpath = patchpath.format(name=name)
    blender.write_patch(fname, patchpath)
    if flatten_with == 'freesurfer':
        freesurfer.flatten(fs_subject, hemi, patch=name, 
                           freesurfer_subject_dir=freesurfer_subject_dir)
        # Check to see if both hemispheres have been flattened
        other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh",
                                     freesurfer_subject_dir=freesurfer_subject_dir)
        other = other.format(name=name+".flat")
        # If so, go ahead and import subject
        if os.path.exists(other):
            freesurfer.import_flat(fs_subject, name, sname=cx_subject,
                                   flat_type='freesurfer',
                                   freesurfer_subject_dir=freesurfer_subject_dir)
    elif flatten_with == 'SLIM':
        flatten_slim(fs_subject, hemi, patch=name, freesurfer_subject_dir=freesurfer_subject_dir)
        other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh",
                                     type='slim',
                                     freesurfer_subject_dir=freesurfer_subject_dir)
        other = other.format(name=name)
        # If so, go ahead and import subject
        if os.path.exists(other):
            freesurfer.import_flat(fs_subject, name, sname=cx_subject,
                                   flat_type='slim',
                                   freesurfer_subject_dir=freesurfer_subject_dir)

    return


### DEPRECATED ###

def fix_wm(subject):
    """Initializes an interface to make white matter edits to the surface. 
    This will open two windows -- a tkmedit window that makes the actual edits,
    as well as a mayavi window to display the surface. Clicking on the mayavi window
    will drop markers which can be loaded using the "Goto Save Point" button in tkmedit.

    If you wish to load the other hemisphere, simply close the mayavi window and the
    other hemisphere will pop up. Mayavi will stop popping up once the tkmedit window
    is closed.

    Once the tkmedit window is closed, a variety of autorecon options are available.
    When autorecon finishes, the new surfaces are immediately imported into the pycortex 
    database.

    Parameters
    ----------
    subject : str
        Name of the subject to edit
    """
    warnings.warn("Deprecated! We recommend using edit_segmentation() and rerun_recon() instead of fix_wm() and fix_pia().")
    status = _cycle_surf(subject, "smoothwm")
    cmd = "tkmedit {subj} wm.mgz lh.smoothwm -aux brainmask.mgz -aux-surface rh.smoothwm"
    sp.call(shlex.split(cmd.format(subj=subject)))
    status.value = 0

    resp = input("1) Run autorecon-wm?\n2) Run autorecon-cp?\n3) Do nothing?\n (Choose 1, 2, or 3)")
    if resp == "1":
        freesurfer.autorecon(subject, "wm")
    elif resp == "2":
        freesurfer.autorecon(subject, "cp")
    elif resp == "3":
        print("Doing nothing...")
        return

    import_freesurfer_subject(subject)

def fix_pia(subject):
    """Initializes an interface to make pial surface edits.
    This function will open two windows -- a tkmedit window that makse the actual edits,
    as well as a mayavi window to display the surface. Clicking on the mayavi window
    will drop markers which can be loaded using the "Goto Save Point" button in tkmedit.

    If you wish to load the other hemisphere, simply close the mayavi window and the
    other hemisphere will pop up. Mayavi will stop popping up once the tkmedit window
    is closed.

    Once the tkmedit window is closed, a variety of autorecon options are available.
    When autorecon finishes, the new surfaces are immediately imported into the pycortex 
    database.

    Parameters
    ----------
    subject : str
        Name of the subject to edit
    """
    warnings.warn("Deprecated! We recommend using edit_segmentation() and rerun_recon() instead of fix_wm() and fix_pia().")
    status = _cycle_surf(subject, "pial")
    cmd = "tkmedit {subj} brainmask.mgz lh.smoothwm -aux T1.mgz -aux-surface rh.smoothwm"
    sp.call(shlex.split(cmd.format(subj=subject)))
    status.value = 0

    resp = input("1) Run autorecon-pia?\n2) Run autorecon-wm?\n3) Do nothing?\n (Choose 1, 2, or 3)")
    if resp == "1":
        freesurfer.autorecon(subject, "pia")
    elif resp == "2":
        freesurfer.autorecon(subject, "wm")
    elif resp == "3":
        print("Doing nothing...")
        return

    import_freesurfer_subject(subject)



def _cycle_surf(subject, surf):
    status = mp.Value('b', 1)
    def cycle_surf():
        idx, hemis = 0, ['lh', 'rh']
        while status.value > 0:
            hemi = hemis[idx%len(hemis)]
            idx += 1
            #HELLISH CODE FOLLOWS, I heavily apologize for this awful code
            #In order for this to work well, mayavi has to block until you close the window
            #Unfortunately, with IPython's event hook, mlab.show does not block anymore
            #There is no way to force mayavi to block, and hooking directly into backend vtk objects cause it to crash out
            #Thus, the only safe way is to call python using subprocess
            cmd = "python -m cortex.freesurfer {subj} {hemi} {surf}"
            sp.call(shlex.split(cmd.format(subj=subject, hemi=hemi, surf=surf)))

    proc = mp.Process(target=cycle_surf)
    proc.start()
    return status
