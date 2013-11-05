import os
import time
import shlex
import subprocess as sp
import multiprocessing as mp

from . import blender
from . import freesurfer
from .db import surfs

def init_subject(subject, filename):
    cmd = "recon-all -i {fname} -s {subj}".format(subj=subject, fname=filename)
    sp.call(shlex.split(cmd))
    freesurfer.autorecon(subject, "all")
    freesurfer.import_subj(subject)

def fix_wm(subject):
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

def cut_surface(subject, hemi, name='flatten', data=None):
    opts = "[hemi=lh,name=%s]"%name
    fname = surfs.getFiles(subject)['anats'].format(type='cutsurf', opts=opts, ext='blend')

    if not os.path.exists(fname):
        blender.fs_cut(fname, subject, hemi)

    if data is not None:
        blender.add_cutdata(fname, data, name=data.description)

    sp.call(shlex.split("blender %s"%fname))
    patchpath = freesurfer.get_paths(subject, hemi).format(name=name)
    blender.write_patch(fname, patchpath)
    
    freesurfer.flatten(subject, hemi, patch=name)
    
    other = freesurfer.get_paths(subject, "lh" if hemi == "rh" else "rh").format(name=name+".flat")
    if os.path.exists(other):
        freesurfer.import_flat(subject, name)

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