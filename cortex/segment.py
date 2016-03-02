import os
import time
import shlex
import subprocess as sp
import multiprocessing as mp

from . import blender
from . import freesurfer
from .database import db

def init_subject(subject, filename):
    """Run the first initial segmentation for a subject's anatomy. This function runs 
    autorecon-all, then imports the subject into the pycortex database.

    Parameters
    ----------
    subject : str
        The name of the subject
    filename : str
        Freesurfer-compatible filename for the anatomical image
    """
    cmd = "recon-all -i {fname} -s {subj}".format(subj=subject, fname=filename)
    sp.call(shlex.split(cmd))
    freesurfer.autorecon(subject, "all")
    freesurfer.import_subj(subject)

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
    status = _cycle_surf(subject, "smoothwm")
    cmd = "tkmedit {subj} wm.mgz lh.smoothwm -aux brainmask.mgz -aux-surface rh.smoothwm"
    sp.call(shlex.split(cmd.format(subj=subject)))
    status.value = 0

    resp = raw_input("1) Run autorecon-wm?\n2) Run autorecon-cp?\n3) Do nothing?\n (Choose 1, 2, or 3)")
    if resp == "1":
        freesurfer.autorecon(subject, "wm")
    elif resp == "2":
        freesurfer.autorecon(subject, "cp")
    elif resp == "3":
        print("Doing nothing...")
        return

    freesurfer.import_subj(subject)

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
    status = _cycle_surf(subject, "pial")
    cmd = "tkmedit {subj} brainmask.mgz lh.smoothwm -aux T1.mgz -aux-surface rh.smoothwm"
    sp.call(shlex.split(cmd.format(subj=subject)))
    status.value = 0

    resp = raw_input("1) Run autorecon-pia?\n2) Run autorecon-wm?\n3) Do nothing?\n (Choose 1, 2, or 3)")
    if resp == "1":
        freesurfer.autorecon(subject, "pia")
    elif resp == "2":
        freesurfer.autorecon(subject, "wm")
    elif resp == "3":
        print("Doing nothing...")
        return

    freesurfer.import_subj(subject)

def cut_surface(cx_subject, hemi, name='flatten', fs_subject=None, data=None, freesurfer_subject_dir=None):
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

    if data is not None:
        blender.add_cutdata(fname, data, name=data.description)

    sp.call(shlex.split("blender %s"%fname))
    patchpath = freesurfer.get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=name)
    blender.write_patch(fname, patchpath)
    
    freesurfer.flatten(fs_subject, hemi, patch=name, freesurfer_subject_dir=freesurfer_subject_dir)
    
    other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh").format(name=name+".flat")
    if os.path.exists(other):
        freesurfer.import_flat(fs_subject, name)

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