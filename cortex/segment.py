"""Controls functions for segmentation of white/gray matter and other things in the brain.
"""
import os
import time
import shlex
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
    """Run the first initial segmentation for a subject's anatomy. This function runs 
    autorecon-all, then imports the subject into the pycortex database.

    Parameters
    ----------
    subject : str
        The name of the subject (this subject is created in the Freesurfer
        SUBJECTS_DIR)
    filenames : str | list
        Freesurfer-compatible filename(s) for the anatomical image(s). This can be
        the first dicom file of a 
    run_all : bool
        Whether to run recon-all all the way through to importing the subject into 
        pycortex. False by default, since we recommend editing (or at least inspecting)
        the brain mask and white matter segmentations prior to importing into 
        pycortex.
    """
    cmd = "recon-all -i {fname} -s {subj}".format(subj=subject, fname=filename)
    print("Calling:\n%{}".format(cmd))
    sp.call(shlex.split(cmd))
    if run_all:
        run_freesurfer_recon(subject, "all")
        import_freesurfer_subject(subject)

def edit_segmentation(subject, volumes=('brain.mgz', 'aseg.mgz', 'brainmask.mgz', 'wm.mgz'),
                  surfaces=('lh.smoothwm', 'rh.smoothwm', 'lh.pial','rh.pial'), 
                  freesurfer_subject_dir=os.environ['SUBJECTS_DIR']):

    cmaps = {'brain':'grayscale',
             'aseg':'lut',
             'brainmask':'heat',
             'wm':'heat',
             'smoothwm':'yellow',
             'pial':'red'
             }
    opacity={'brain':1.0,
             'aseg':0.4,
             'brainmask':0.4,
             'wm':0.4,
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

def _export_obj(blend_file, obj_file):
    pass

def flatten_slim(obj_in, obj_out, slim_path=slim_path):
    """Flatten brain w/ slim object flattening"""
    sp.call([slim_path, obj_in, obj_out])
    # Call SLIM algorithm to flatten
    pts, polys, norms, (u, v) = formats.read_obj(obj_out, norm=True, uv=True)
    np.savez('/Users/mark/Desktop/test.npz', polys=polys, u=u, v=v)
    return

def tmp_flatten(cx_subject, hemi, name='flatten', freesurfer_subject_dir=None):
    fs_subject = cx_subject
    opts = "[hemi=%s,name=%s]"%(hemi, name)
    fname = db.get_paths(cx_subject)['anats'].format(type='cutsurf', opts=opts, ext='blend')
    print("Loading blender file:")
    print(fname)
    patchpath = freesurfer.get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=name)
    objpath = patchpath.replace('.patch.3d', '.obj')
    objout = objpath.replace('.obj', '_slim.obj')
    blender.write_patch(fname, objpath, flat_type='obj')
    flatten_slim(objpath, objout)

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

    if data is not None:
        blender.add_cutdata(fname, data, name=data.description)
    blender_cmd = options.config.get('dependency_paths', 'blender')
    sp.call([blender_cmd, fname])
    patchpath = freesurfer.get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=name)
    if flatten_with=='freesurfer':
        blender.write_patch(fname, patchpath)    
        freesurfer.flatten(fs_subject, hemi, patch=name, freesurfer_subject_dir=freesurfer_subject_dir)
    elif flatten_with=='SLIM':
        patchpath = freesurfer.get_paths(fs_subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=name)
        objpath = patchpath.replace('.patch.3d', '.obj')
        objout = objpath.replace('.obj', '_slim.obj')
        blender.write_patch(fname, objpath)
        flatten_slim(objpath, objout)
        return

    other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh",
                                 freesurfer_subject_dir=freesurfer_subject_dir).format(name=name+".flat")
    
    if os.path.exists(other):
        freesurfer.import_flat(fs_subject, name, sname=cx_subject, freesurfer_subject_dir=freesurfer_subject_dir)

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
