"""Controls functions for segmentation of white/gray matter and other things in the brain.
"""
import os
import time
import shlex
import warnings
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


def init_subject(subject, filenames, do_import_subject=False, **kwargs):
    """Run the first initial segmentation for a subject's anatomy (in Freesurfer).

    This function creates a Freesurfer subject and runs autorecon-all, 
    then (optionally) imports the subject into the pycortex database.

    NOTE: This function requires a functional Freesurfer install! 
    Also, still can't handle T2 weighted anatomical volume input. Please use
    Freesurfer directly (and then import) for advanced recon-all input 
    options; this is just a convenience function.

    Parameters
    ----------
    subject : str
        The name of the subject (this subject is created in the Freesurfer
        SUBJECTS_DIR)
    filenames : str or list
        Freesurfer-compatible filename(s) for the anatomical image(s). This can
        be the first dicom file of a series of dicoms, a nifti file, an mgz
        file, etc.
    do_import_subject : bool
        Whether to import the Freesurfer-processed subject (without further)
        editing) into pycortex. False by default, since we recommend editing 
        (or at least inspecting) the brain mask and white matter segmentations 
        prior to importing into pycortex.
    kwargs : keyword arguments passed to cortex.freesurfer.autorecon()
        useful ones: parallel=True, n_cores=4 (or more, if you have them)
    """
    if 'run_all' in kwargs:
        warnings.warn('`run_all` is deprecated - please use do_import_subject keyword arg instead!')
        do_import_subject = kwargs.pop('run_all')
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    filenames = ' '.join(['-i %s'%f for f in filenames])
    cmd = "recon-all {fname} -s {subj}".format(subj=subject, fname=filenames)
    print("Calling:\n%{}".format(cmd))
    sp.call(shlex.split(cmd))
    run_freesurfer_recon(subject, "all", **kwargs)
    if do_import_subject:
        import_freesurfer_subject(subject)


def edit_segmentation(subject,
                      volumes=('aseg.mgz', 'brainmask.mgz', 'wm.mgz'),
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
             'brainmask': 'gray',
             'wm': 'heat',
             'smoothwm': 'yellow',
             'white': 'green',
             'pial': 'blue'
             }
    opacity={'brain': 1.0,
             'aseg': 0.4,
             'brainmask': 1.0,
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
    print("If you have edited the white matter surface, you should run:\n")
    print("    `cortex.segment.run_freesurfer_recon('%s', 'wm')`\n"%subject)
    print("If you have edited the brainmask (pial surface), you should run:\n")
    print("    `cortex.segment.run_freesurfer_recon('%s', 'pia')`"%subject)


def cut_surface(cx_subject, hemi, name='flatten', fs_subject=None, data=None,
                freesurfer_subject_dir=None, flatten_with='freesurfer', 
                do_import_subject=True, blender_cmd=None, **kwargs):
    """Initializes an interface to cut the segmented surface for flatmapping.
    This function creates or opens a blend file in your filestore which allows
    surfaces to be cut along hand-defined seams. Blender will automatically
    open the file. After edits are made, remember to save the file, then exit
    Blender.

    The surface will be automatically extracted from blender then run through
    the mris_flatten command in freesurfer. The flatmap will be imported once
    that command finishes if `do_import_subject` is True (default value). 

    Parameters
    ----------
    cx_subject : str
        Name of the subject to edit (pycortex subject ID)
    hemi : str
        Which hemisphere to flatten. Should be "lh" or "rh"
    name : str, optional
        String name of the current flatten attempt. Defaults to "flatten"
    data : Dataview or List(Dataview)
        A data view object or list of data view objects to display on the 
        surface as a cutting guide.
    fs_subject : str
        Name of Freesurfer subject (if different from pycortex subject)
        None defaults to `cx_subject`
    freesurfer_subject_dir : str
        Name of Freesurfer subject directory. None defaults to SUBJECTS_DIR
        environment variable
    flatten_with : str
        'freesurfer' or 'SLIM' - 'freesurfer' (default) uses freesurfer's 
        `mris_flatten` function to flatten the cut surface. 'SLIM' uses
        the SLIM algorithm, which takes much less time but tends to leave
        more distortions in the flatmap. SLIM is an optional dependency, and 
        must be installed to work; clone the code 
        (https://github.com/MichaelRabinovich/Scalable-Locally-Injective-Mappings) 
        to your computer and set the slim dependency path in your pycortex config 
        file to point to </path/to/your/slim/install>/ReweightedARAP
    do_import_subject : bool
        set option to automatically import flatmaps when both are completed 
        (if set to false, you must import later with `cortex.freesurfer.import_flat()`)
    """
    if fs_subject is None:
        fs_subject = cx_subject
    opts = "[hemi=%s,name=%s]"%(hemi, name)
    fname = db.get_paths(cx_subject)['anats'].format(type='cutsurf', opts=opts, ext='blend')
    # Double-check that fiducial and inflated vertex counts match
    # (these may not match if a subject is initially imported from freesurfer to pycortex, 
    # and then edited further for a better segmentation and not re-imported)
    ipt, ipoly, inrm = freesurfer.get_surf(fs_subject, hemi, 'inflated')
    fpt, fpoly, fnrm = freesurfer.get_surf(fs_subject, hemi, 'fiducial')
    if ipt.shape[0] != fpt.shape[0]:
        raise ValueError("Please re-import subject - fiducial and inflated vertex counts don't match!")
    else:
        print('Vert check ok!')
    if not os.path.exists(fname):
        blender.fs_cut(fname, fs_subject, hemi, freesurfer_subject_dir)
    # Add localizer data to facilitate cutting
    if data is not None:
        if isinstance(data, list):
            for d in data:
                blender.add_cutdata(fname, d, name=d.description)
        else:
            blender.add_cutdata(fname, data, name=data.description)
    if blender_cmd is None:
        blender_cmd = options.config.get('dependency_paths', 'blender')
    # May be redundant after blender.fs_cut above...
    if os.path.exists(fname):
        blender._legacy_blender_backup(fname, blender_path=blender_cmd)
    sp.call([blender_cmd, fname])
    patchpath = freesurfer.get_paths(fs_subject, hemi,
                                     freesurfer_subject_dir=freesurfer_subject_dir)
    patchpath = patchpath.format(name=name)
    blender.write_patch(fname, patchpath, blender_path=blender_cmd)
    if flatten_with == 'freesurfer':
        done = freesurfer.flatten(fs_subject, hemi, patch=name,
                           freesurfer_subject_dir=freesurfer_subject_dir,
                           **kwargs)
        if not done:
            # If flattening is aborted, skip the rest of this function
            # (Do not attempt to import completed flatmaps)
            return
        if do_import_subject:
            # Check to see if both hemispheres have been flattened
            other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh",
                                         freesurfer_subject_dir=freesurfer_subject_dir)
            other = other.format(name=name+".flat")
            # If so, go ahead and import subject
            if os.path.exists(other):
                freesurfer.import_flat(fs_subject, name, cx_subject=cx_subject,
                            flat_type='freesurfer',
                            freesurfer_subject_dir=freesurfer_subject_dir)
    elif flatten_with == 'SLIM':
        done = flatten_slim(fs_subject, hemi, patch=name,
                            freesurfer_subject_dir=freesurfer_subject_dir,
                            **kwargs)
        if not done:
            # If flattening is aborted, skip the rest of this function
            # (Do not attempt to import completed flatmaps)
            return
        if do_import_subject:
            other = freesurfer.get_paths(fs_subject, "lh" if hemi == "rh" else "rh",
                                         type='slim',
                                         freesurfer_subject_dir=freesurfer_subject_dir)
            other = other.format(name=name)
            # If so, go ahead and import subject
            if os.path.exists(other):
                freesurfer.import_flat(fs_subject, name, cx_subject=cx_subject,
                            flat_type='slim',
                            freesurfer_subject_dir=freesurfer_subject_dir)

    return


def flatten_slim(subject, hemi, patch, n_iterations=20, freesurfer_subject_dir=None,
                 slim_path=slim_path, do_flatten=None):
    """Flatten brain w/ slim object flattening

    Parameters
    ----------
    subject : str
        freesurfer subject
    hemi : str
        'lh' or 'rh' for left or right hemisphere
    patch : str
        name of patch, often "flatten" (obj file used here is {hemi}_{patch}.obj
        in the subject's freesurfer directory)
    freesurfer_subject_dir : str
        path to freesurfer subejct dir. Defaults to environment variable
        SUBJECTS_DIR
    slim_path : str
        path to SLIM flattening. Defaults to path specified in config file.
    """
    if slim_path == 'None':
        slim_url = 'https://github.com/MichaelRabinovich/Scalable-Locally-Injective-Mappings'
        raise ValueError("Please download SLIM ({slim_url}) and set the path to it in the `slim` field\n"
                         "in the `[dependency_paths]` section of your config file ({usercfg}) \n"
                         "if you wish to use slim!".format(slim_url=slim_url, usercfg=options.usercfg))
    if do_flatten is None:
        resp = input('Flattening with SLIM will take a few mins. Continue? (type y or n and press return)')
        do_flatten = resp.lower() in ('y', 'yes')
    if not do_flatten:
        print("Not flattening...")
        return

    # File paths
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
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
    slim_cmd = [slim_path, obj_in, obj_out, str(n_iterations)]
    print('Calling: {}'.format(' '.join(slim_cmd)))
    out = sp.check_output(slim_cmd)
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
    # Modify output .obj file to reflect flattening
    #surfpath = os.path.join(freesurfer_subject_dir, subject, "surf", "flat_{hemi}.gii")
    #fname = surfpath.format(hemi=hemi)
    #print("Writing %s"%fname)
    formats.write_obj(obj_out.replace('_slim','.flat_slim'), pts=pts_flat, polys=polys)
    return


def show_surface(subject, hemi, surface_type, patch=None, flatten_step=None, freesurfer_subject_dir=None):
    """
    Parameters
    ----------
    subject: str
        freesurfer subject name
    hemi: str
        'lh' or 'rh' for left hemisphere or right hemisphere
    surface_type : str
        type of surface to show, e.g. 'patch', 'surf', etc if 'patch',
        patch name must be specified in patch kwarg
    patch: str
        name of patch, e.g. 'flatten.flat',  'flatten2.flat', etc

    """
    meshlab_path = options.config.get('dependency_paths', 'meshlab')
    if meshlab_path == 'None':
        try:
            # exists in system but not available in config
            meshlab_path = sp.check_output('command -v meshlab', shell=True).strip()
            warnings.warn('Using system meshlab: %s'%meshlab_path)
        except sp.CalledProcessError:
            raise ValueError('You must have installed meshlab to call this function.')

    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
    if surface_type in ('inflated', 'fiducial'):
        input_type = 'surf'
    else:
        input_type = surface_type
    fpath = freesurfer.get_paths(subject, hemi, input_type,
                                     freesurfer_subject_dir=freesurfer_subject_dir)

    if not 'obj' in fpath:
        pts, polys, curv = freesurfer.get_surf(subject, hemi, surface_type,
                                               patch=patch,
                                               flatten_step=flatten_step,
                                               freesurfer_subject_dir=freesurfer_subject_dir)
        # TODO: use tempfile library here
        objf = '/tmp/temp_surf.obj'
        formats.write_obj(objf, pts, polys)
    else:
        objf = fpath.format(name=patch)
    # Call meshlab to display surface
    out = sp.check_output([meshlab_path, objf])


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
    This function will open two windows -- a tkmedit window that makes the actual edits,
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
