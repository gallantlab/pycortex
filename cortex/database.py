"""
Contains a singleton object `db` of type `Database` which allows easy access to surface files, anatomical images, and transforms that are stored in the pycortex filestore.
"""
import copy
import functools
import glob
import json
import os
import re
import shutil
import tempfile
import warnings
from builtins import input
from hashlib import sha1

import numpy as np

from . import options

default_filestore = options.config.get('basic', 'filestore')


def _memo(fn):
    @functools.wraps(fn)
    def memofn(self, *args, **kwargs):
        if not hasattr(self, "_memocache"):
            setattr(self, "_memocache", dict())
        #h = sha1(str((id(fn), args, kwargs))).hexdigest()
        h = str((id(fn), args, kwargs))
        if h not in self._memocache:
            self._memocache[h] = fn(self, *args, **kwargs)
        return copy.deepcopy(self._memocache[h])

    return memofn

class SubjectDB(object):
    def __init__(self, subj, filestore=default_filestore):
        self.subject = subj
        self._warning = None
        self._transforms = None
        self._surfaces = None
        self.filestore = filestore

        try:
            with open(os.path.join(filestore, subj, "warning.txt")) as fp:
                self._warning = fp.read()
        except IOError:
            pass

    @property
    def transforms(self):
        if self._transforms is not None:
            return self._transforms
        self._transforms = XfmDB(self.subject, filestore=self.filestore)
        return self._transforms

    @property
    def surfaces(self):
        if self._surfaces is not None:
            return self._surfaces
        self._surfaces = SurfaceDB(self.subject, filestore=self.filestore)
        return self._surfaces

class SurfaceDB(object):
    def __init__(self, subj, filestore=default_filestore):
        self.subject = subj
        self.types = {}
        db = Database(filestore)
        for name in db.get_paths(subj)['surfs'].keys():
            self.types[name] = Surf(subj, name, filestore=filestore)
                
    def __repr__(self):
        return "Surfaces: [{surfs}]".format(surfs=', '.join(list(self.types.keys())))
    
    def __dir__(self):
        return list(self.types.keys())

    def __getattr__(self, attr):
        if attr in self.types:
            return self.types[attr]
        raise AttributeError(attr)

class Surf(object):
    def __init__(self, subject, surftype, filestore=default_filestore):
        self.subject, self.surftype = subject, surftype
        self.db = Database(filestore)

    def get(self, hemisphere="both"):
        return self.db.get_surf(self.subject, self.surftype, hemisphere)
    
    def show(self, hemisphere="both"):
        from mayavi import mlab
        pts, polys = self.db.get_surf(self.subject, self.surftype, hemisphere, merge=True, nudge=True)
        return mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)

class XfmDB(object):
    def __init__(self, subj, filestore=default_filestore):
        self.subject = subj
        self.filestore = filestore
        self.xfms = Database(self.filestore).get_paths(subj)['xfms']

    def __getitem__(self, name):
        if name in self.xfms:
            return XfmSet(self.subject, name, filestore=self.filestore)
        raise AttributeError
    
    def __repr__(self):
        xfms = "\n".join(sorted(self.xfms))
        return f"Available transforms for {self.subject}:\n{xfms}"

class XfmSet(object):
    def __init__(self, subj, name, filestore=default_filestore):
        self.subject = subj
        self.name = name
        jspath = os.path.join(filestore, subj, 'transforms', name, 'matrices.xfm')
        with open(jspath) as fp:
            self._jsdat = json.load(fp)
        self.masks = MaskSet(subj, name, filestore=filestore)
        self.db = Database(filestore)
    
    def __getattr__(self, attr):
        if attr in self._jsdat:
            return self.db.get_xfm(self.subject, self.name, attr)
        raise AttributeError
    
    def __repr__(self):
        return "Types: {types}".format(types=", ".join(self._jsdat.keys()))

class MaskSet(object):
    def __init__(self, subj, name, filestore=default_filestore):
        self.subject = subj
        self.xfmname = name
        maskform = Database(filestore).get_paths(subj)['masks']
        maskpath = maskform.format(xfmname=name, type='*')
        self._masks = dict((os.path.split(path)[1][5:-7], path) for path in glob.glob(maskpath))

    def __getitem__(self, item):
        import nibabel
        return nibabel.load(self._masks[item]).get_fdata().T

    def __repr__(self):
        return "Masks: [{types}]".format(types=', '.join(self._masks.keys()))

class Database(object):
    """
    Database()

    Surface database

    Attributes
    ----------
    This database object dynamically generates handles to all subjects within the filestore.
    """
    def __init__(self, filestore=default_filestore):
        self.filestore = filestore
        self._subjects = None
        self.auxfile = None
    
    def __repr__(self):
        subjs = "\n   ".join(sorted(self.subjects.keys()))
        return """Pycortex database\n  Subjects:\n   {subjs}""".format(subjs=subjs)
    
    def __getattr__(self, attr):
        if attr in self.subjects:
            if self.subjects[attr]._warning is not None:
                warnings.warn(self.subjects[attr]._warning)
            return self.subjects[attr]
        else:
            raise AttributeError
    
    def __dir__(self):
        return ["save_xfm","get_xfm", "get_surf", "get_anat", "get_surfinfo", "subjects", # "get_paths", # Add?
                "get_mask", "get_overlay","get_cache", "get_view", "save_view", "get_mnixfm",
                'get_mri_surf2surf_matrix'] + list(self.subjects.keys())

    @property
    def subjects(self):
        if self._subjects is not None:
            return self._subjects
        subjs = os.listdir(os.path.join(self.filestore))
        subjs = [s for s in subjs if os.path.isdir(os.path.join(self.filestore, s))]
        subjs = sorted(subjs)
        self._subjects = dict([(sname, SubjectDB(sname, filestore=self.filestore)) for sname in subjs])
        return self._subjects

    def reload_subjects(self):
        """Force the reload of the subject dictionary."""
        self._subjects = None
        self.subjects

    def get_anat(self, subject, type='raw', xfmname=None, recache=False, order=1, **kwargs):
        """Return anatomical information from the filestore. Anatomical information is defined as
        any volume-space anatomical information pertaining to the subject, such as T1 image,
        white matter masks, etc. Volumes not found in the database will be automatically generated.

        Parameters
        ----------
        subject : str
            Name of the subject
        type : str
            Type of anatomical volume to return. This should be the name of one of the 
        recache : bool
            Regenerate the information

        Returns
        -------
        volume : nibabel object
            Volume containing
        """
        opts = ""
        if len(kwargs) > 0:
            opts = "[%s]"%','.join(["%s=%s"%i for i in kwargs.items()])
        anatform = self.get_paths(subject)['anats']
        anatfile = anatform.format(type=type, opts=opts, ext="nii.gz")

        if not os.path.exists(anatfile) or recache:
            print("Generating %s anatomical..."%type)
            from . import anat
            getattr(anat, type)(anatfile, subject, **kwargs)

        import nibabel
        anatnib = nibabel.load(anatfile)

        if xfmname is None:
            return anatnib

        from . import volume
        return volume.anat2epispace(anatnib.get_fdata().T.astype(float), subject, xfmname, order=order)

    def get_surfinfo(self, subject, type="curvature", recache=False, **kwargs):
        """Return auxiliary surface information from the filestore. Surface info is defined as 
        anatomical information specific to a subject in surface space. A Vertex class will be returned
        as necessary. Info not found in the filestore will be automatically generated.

        See documentation in cortex.surfinfo for auto-generation code

        Parameters
        ----------
        subject: str
            Subject name for which to return info
        type: str
            Type of surface info returned, IE. curvature, distortion, sulcaldepth, etc.
        recache: bool
            Regenerate the information

        Returns
        -------
        verts : Vertex class
            If the surface information has "left" and "right" entries, a Vertex class is returned

        - OR -
        
        npz : npzfile
            Otherwise, an npz object is returned. Remember to close it!
        """
        opts = ""
        if len(kwargs) > 0:
            opts = "[%s]"%','.join(["%s=%s"%i for i in kwargs.items()])
        try:
            self.auxfile.get_surf(subject, "fiducial")
            surfifile = os.path.join(self.get_cache(subject),"%s%s.npz"%(type, opts)) 
        except (AttributeError, IOError):
            surfiform = self.get_paths(subject)['surfinfo']
            surfifile = surfiform.format(type=type, opts=opts)

            if not os.path.exists(os.path.join(self.filestore, subject, "surface-info")):
                os.makedirs(os.path.join(self.filestore, subject, "surface-info"))

        if not os.path.exists(surfifile) or recache:
            print ("Generating %s surface info..."%type)
            from . import surfinfo
            getattr(surfinfo, type)(surfifile, subject, **kwargs)

        npz = np.load(surfifile)
        if "left" in npz and "right" in npz:
            from .dataset import Vertex
            verts = np.hstack([npz['left'], npz['right']])
            npz.close()
            return Vertex(verts, subject)
        return npz
    
    def get_mri_surf2surf_matrix(self, subject, surface_type, hemi='both', 
                                 fs_subj=None, target_subj='fsaverage', 
                                 **kwargs):
        """Get matrix generated by surf2surf to map one subject's surface to another's

        Parameters
        ----------
        subject : pycortex 
            pycortex subject ID
        surface_type : str
            type of surface; one of ['fiducial', 'inflated', 'white', 
            'pial'] ... more?
        hemi : str or list
            one of: 'both', 'lh', 'rh'
        fs_subj : str
            string ID for freesurfer subject (if different from 
            pycortex subject ID; None assumes they are the same)
        target_subj : str
            string ID for freesurfer subject to which to map surface
            from `subject` (or `fs_subj`)

        Other Parameters
        ----------------
        subjects_dir : str
            Custom path to freesurfer subjects dir can be specified in the 
            keyword args
        n_neighbors : int
            ...
        random_state : scalar
            ...
        n_test_images : int
            ...
        coef_threshold : scalar or None
            ...
        renormalize : bool
            ...

        """
        from .freesurfer import get_mri_surf2surf_matrix as mri_s2s
        from .utils import load_sparse_array, save_sparse_array
        if fs_subj is None:
            fs_subj = subject
        fpath = self.get_paths(subject)['surf2surf'].format(
            source=fs_subj,
            target=target_subj,
            surface_type=surface_type
        )
        # Backward compatibility
        fdir, _ = os.path.split(fpath)
        if not os.path.exists(fdir):
            print("Creating surf2surf directory for subject %s"%(subject))
            os.makedirs(fdir)
        if hemi == 'both':
            hemis = ['lh', 'rh']
        else:
            hemis = [hemi]
        if os.path.exists(fpath):
            mats = [load_sparse_array(fpath, h) for h in hemis]
        else:
            mats = []
            for h in hemis:
                tmp = mri_s2s(fs_subj, h, surface_type, 
                    target_subj=target_subj, **kwargs)
                mats.append(tmp)
                save_sparse_array(fpath, tmp, h, mode='a')
        return mats

    def get_overlay(self, subject, overlay_file=None, **kwargs):
        from . import svgoverlay
        pts, polys = self.get_surf(subject, "flat", merge=True, nudge=True)

        paths = self.get_paths(subject)
        # NOTE: This try loop is broken, in that it does nothing for the intended 
        # use case (loading an overlay from a packed subject) - needs fixing. 
        # This hasn't come up yet due to very rare use of packed subjects.
        if self.auxfile is not None:
            try:
                # Prob should be something like:
                #self.auxfile.get_overlay(subject, **kwargs)
                tf = self.auxfile.get_overlay(subject) # kwargs??
                svgfile = tf.name
            except (AttributeError, IOError):
                # I think the rest of the code should be here (as in other functions...)
                pass
        if 'pts' in kwargs:
            pts = kwargs['pts']
            del kwargs['pts']
        # ... and `svgfile` variable should be used here...
        if os.path.exists(paths['rois']) and not os.path.exists(paths['overlays']):
            svgoverlay.import_roi(paths['rois'], paths['overlays'])

        if overlay_file is None:
            overlay_file = paths['overlays']
        return svgoverlay.get_overlay(subject, overlay_file, pts, polys, **kwargs)
    
    def save_xfm(self, subject, name, xfm, xfmtype="magnet", reference=None):
        """
        Load a transform into the surface database. If the transform exists already, update it
        If it does not exist, copy the reference epi into the filestore and insert.

        Parameters
        ----------
        subject : str
            Name of the subject
        name : str
            Name to identify the transform
        xfm : (4,4) array
            The affine transformation matrix
        xfmtype : str, optional
            Type of the provided transform, either magnet space or coord space.
            Defaults to 'magnet'.
        reference : str, optional
            The nibabel-compatible reference image associated with this transform.
            Required if name not in database
        """
        if xfmtype not in ["magnet", "coord"]:
            raise TypeError("Unknown transform type")

        import nibabel

        path = os.path.join(self.filestore, subject, "transforms", name)
        fname = os.path.join(path, "matrices.xfm")
        if os.path.exists(fname):
            with open(fname) as fp:
                jsdict = json.load(fp)
        else:
            os.mkdir(path)
            if reference is None:
                raise ValueError("Please specify a reference")
            fpath = os.path.join(path, "reference.nii.gz")
            nib = nibabel.load(reference)
            data = nib.get_fdata()
            if len(data.shape) > 3:
                import warnings
                warnings.warn('You are importing a 4D dataset, automatically selecting the first volume as reference')
                data = data[...,0]
            out = nibabel.Nifti1Image(data, nib.affine, header=nib.header)
            nibabel.save(out, fpath)

            jsdict = dict()

        nib = nibabel.load(os.path.join(path, "reference.nii.gz"))
        if xfmtype == "magnet":
            jsdict['magnet'] = np.array(xfm).tolist()
            jsdict['coord'] = np.dot(np.linalg.inv(nib.affine), xfm).tolist()
        elif xfmtype == "coord":
            jsdict['coord'] = np.array(xfm).tolist()
            jsdict['magnet'] = np.dot(nib.affine, xfm).tolist()
        
        files = self.get_paths(subject)
        if len(glob.glob(files['masks'].format(xfmname=name, type="*"))) > 0:
            raise ValueError('Refusing to change a transform with masks')
            
        with open(fname, "w") as fp:
            json.dump(jsdict, fp, sort_keys=True, indent=4)
    
    def get_xfm(self, subject, name, xfmtype="coord"):
        """Retrieves a transform from the filestore

        Parameters
        ----------
        subject : str
            Name of the subject
        name : str
            Name of the transform
        xfmtype : str, optional
            Type of transform to return. Defaults to coord.
        """
        from .xfm import Transform
        if xfmtype == 'coord':
            try:
                return self.auxfile.get_xfm(subject, name)
            except (AttributeError, IOError):
                pass

        if name == "identity":
            nib = self.get_anat(subject, 'raw')
            return Transform(np.linalg.inv(nib.affine), nib)

        fname = os.path.join(self.filestore, subject, "transforms", name, "matrices.xfm")
        reference = os.path.join(self.filestore, subject, "transforms", name, "reference.nii.gz")
        with open(fname) as f:
            xfmdict = json.load(f)
        return Transform(xfmdict[xfmtype], reference)

    @_memo
    def get_surf(self, subject, type, hemisphere="both", merge=False, nudge=False):
        '''Return the surface pair for the given subject, surface type, and hemisphere.

        Parameters
        ----------
        subject : str
            Name of the subject
        type : str
            Type of surface to return, probably in (fiducial, inflated, 
            veryinflated, hyperinflated, superinflated, flat)
        hemisphere : "lh", "rh"
            Which hemisphere to return
        merge : bool
            Vstack the hemispheres, if requesting both
        nudge : bool
            Nudge the hemispheres apart from each other, for overlapping surfaces
            (inflated, etc)

        Returns
        -------
        left, right :
            If request is for both hemispheres, otherwise:
        pts, polys, norms : ((p,3) array, (f,3) array, (p,3) array or None)
            For single hemisphere
        '''
        try:
            return self.auxfile.get_surf(subject, type, hemisphere, merge=merge, nudge=nudge)
        except (AttributeError, IOError):
            pass

        files = self.get_paths(subject)['surfs']

        if hemisphere.lower() == "both":
            left, right = [ self.get_surf(subject, type, hemisphere=h) for h in ["lh", "rh"]]
            if type != "fiducial" and nudge:
                left[0][:,0] -= left[0].max(0)[0]
                right[0][:,0] -= right[0].min(0)[0]
            
            if merge:
                pts   = np.vstack([left[0], right[0]])
                polys = np.vstack([left[1], right[1]+len(left[0])])
                return pts, polys

            return left, right
        elif hemisphere.lower() in ("lh", "left"):
            hemi = "lh"
        elif hemisphere.lower() in ("rh", "right"):
            hemi = "rh"
        else:
            raise TypeError("Not a valid hemisphere name")
        
        if type == 'fiducial' and 'fiducial' not in files:
            wpts, polys = self.get_surf(subject, 'wm', hemi)
            ppts, _     = self.get_surf(subject, 'pia', hemi)
            return (wpts + ppts) / 2, polys

        try:
            from . import formats
            fnm = str(os.path.splitext(files[type][hemi])[0])
            return formats.read(fnm)
        except KeyError:
            raise IOError

    def save_mask(self, subject, xfmname, type, mask):
        fname = self.get_paths(subject)['masks'].format(xfmname=xfmname, type=type)
        if os.path.exists(fname):
            raise IOError('Refusing to overwrite existing mask')

        import nibabel
        xfm = self.get_xfm(subject, xfmname)
        if xfm.shape != mask.shape:
            raise ValueError("Invalid mask shape: must match shape of reference image")
        affine = xfm.reference.affine
        nib = nibabel.Nifti1Image(mask.astype(np.uint8).T, affine)
        nib.to_filename(fname)

    def get_mask(self, subject, xfmname, type='thick'):
        if hasattr(type, 'decode'):
            type = type.decode('utf8')        

        try:
            self.auxfile.get_mask(subject, xfmname, type)
        except (AttributeError, IOError):
            pass

        fname = self.get_paths(subject)['masks'].format(xfmname=xfmname, type=type)
        try:
            import nibabel
            nib = nibabel.load(fname)
            return nib.get_fdata().T != 0
        except IOError:
            print('Mask not found, generating...')
            from .utils import get_cortical_mask
            mask = get_cortical_mask(subject, xfmname, type)
            self.save_mask(subject, xfmname, type, mask)
            return mask

    def get_shared_voxels(self, subject, xfmname, hemi="both", merge=True, use_astar=True, recache=False):
        """Get an array indicating which vertices are inappropriately mapped to the same voxel.

        For a given transform and surface, returns an array containing a list of vertices which 
        are spatially distant on the cortical surface but that map to the same voxels (this occurs
        at sulcal crossings)

        """
        # Test for packed subjects
        try:
            voxels = self.auxfile.get_shared_voxels(subject, xfmname, hemi=hemi, merge=merge, use_astar=use_astar)
            return voxels
        except (AttributeError, IOError):
            pass
        # Proceed w/ potential load
        shared_voxel_file = os.path.join(self.get_cache(subject), 'shared_vertices_{xfmname}_{hemi}.npy'.format(xfmname=xfmname, hemi=hemi))
        if not os.path.exists(shared_voxel_file) or recache:
            print('Shared voxel array not found, generating...')
            from .utils import get_shared_voxels as _get_shared_voxels
            voxels = _get_shared_voxels(subject, xfmname, hemi=hemi, merge=merge, use_astar=use_astar)
            np.save(shared_voxel_file, voxels)
            return voxels
        else:
            voxels = np.load(shared_voxel_file)
            return voxels

    def get_coords(self, subject, xfmname, hemisphere="both", magnet=None):
        """Calculate the coordinates of each vertex in the epi space by transforming the fiducial to the coordinate space

        Parameters
        ----------
        subject : str
            Name of the subject
        xfmname : str
            Name of the transform
        hemisphere : str, optional
            Which hemisphere to return. If "both", return concatenated. Defaults to "both".
        """
        import warnings
        warnings.warn('Please use a Mapper object instead', DeprecationWarning)

        if magnet is None:
            xfm = self.get_xfm(subject, xfmname, xfmtype="coord")
        else:
            xfm = self.get_xfm(subject, xfmname, xfmtype="magnet")
            xfm = np.linalg.inv(magnet) * xfm

        coords = []
        vtkTmp = self.get_surf(subject, "fiducial", hemisphere=hemisphere, nudge=False)
        if not isinstance(vtkTmp,(tuple,list)):
            vtkTmp = [vtkTmp]
        for pts, polys in vtkTmp:
            wpts = np.vstack([pts.T, np.ones(len(pts))])
            coords.append(np.dot(xfm.xfm, wpts)[:3].round().astype(int).T)

        return coords

    def get_cache(self, subject):
        try:
            self.auxfile.get_surf(subject, "fiducial")
            #generate the hashed name of the filename and subject as the directory name
            import hashlib
            hashname = "pycx_%s"%hashlib.md5(self.auxfile.h5.filename).hexdigest()[-8:]
            cachedir = os.path.join(tempfile.gettempdir(), hashname, subject)
        except (AttributeError, IOError):
            try:
                # Get cache dir from config file
                cachedir = os.path.join(options.config.get('basic', 'cache'),
                                        subject, 'cache')
            except options.configparser.NoOptionError:
                # If not defined, go with default cache
                cachedir = os.path.join(self.filestore, subject, "cache")
        cachedir = os.path.expanduser(cachedir)
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        return cachedir

    def clear_cache(self, subject, clear_all_caches=True):
        """Clears config-specified and default file caches for a subject.
        
        """
        local_cachedir = self.get_cache(subject)
        shutil.rmtree(local_cachedir)
        os.makedirs(local_cachedir)
        # Check on default cache for subject as well, on the off chance 
        # that other people have cached files here for this subject
        default_cachedir = os.path.join(self.filestore, subject, "cache")
        if clear_all_caches is True:
            # Just delete them.
            shutil.rmtree(default_cachedir)
            os.makedirs(default_cachedir)
        if (len(os.listdir(default_cachedir)) > 0):
            # Make sure you didn't want to delete them. You probably should.
            proceed = input("Files exist in %s too! Delete them? [y]/n: "%default_cachedir)
            if proceed.lower() in ('', 'y'):
                shutil.rmtree(default_cachedir)
                os.makedirs(default_cachedir)

    def get_paths(self, subject):
        """Get a dictionary with a list of all candidate filenames for associated data, such as roi overlays, flatmap caches, and ctm caches.
        """
        surfpath = os.path.join(self.filestore, subject, "surfaces")

        if self.subjects[subject]._warning is not None:
            warnings.warn(self.subjects[subject]._warning)

        surfs = dict()
        for surf in os.listdir(surfpath):
            ssurf = os.path.splitext(surf)[0].split('_')
            name = '_'.join(ssurf[:-1])
            hemi = ssurf[-1]

            if name not in surfs:
                surfs[name] = dict()
            surfs[name][hemi] = os.path.abspath(os.path.join(surfpath,surf))

        viewsdir = os.path.join(self.filestore, subject, "views")
        if not os.path.exists(viewsdir):
            os.makedirs(viewsdir)
        views = os.listdir(viewsdir)

        filenames = dict(
            surfs=surfs,
            xfms=sorted(os.listdir(os.path.join(self.filestore, subject, "transforms"))),
            xfmdir=os.path.join(self.filestore, subject, "transforms", "{xfmname}", "matrices.xfm"),
            anats=os.path.join(self.filestore, subject, "anatomicals", '{type}{opts}.{ext}'), 
            surfinfo=os.path.join(self.filestore, subject, "surface-info", '{type}{opts}.npz'),
            masks=os.path.join(self.filestore, subject, 'transforms', '{xfmname}', 'mask_{type}.nii.gz'),
            rois=os.path.join(self.filestore, subject, "rois.svg").format(subj=subject),
            overlays=os.path.join(self.filestore, subject, "overlays.svg").format(subj=subject),
            views=sorted([os.path.splitext(f)[0] for f in views]),
            surf2surf=os.path.join(self.filestore, subject, "surf2surf", "{source}_to_{target}", "matrices_{surface_type}.hdf"),
        )

        return filenames

    def make_subj(self, subject):
        if os.path.exists(os.path.join(self.filestore, subject)):
            if input("Are you sure you want to overwrite this existing subject?\n"
                     "This will delete all files for this subject in the filestore, "
                     "including all blender cuts. Type YES\n") == "YES":
                shutil.rmtree(os.path.join(self.filestore, subject))
            else:
                raise ValueError('Do not overwrite')

        for dirname in ['transforms', 'anatomicals', 'cache', 'surfaces', 'surface-info','views']:
            try:
                path = os.path.join(self.filestore, subject, dirname)
                os.makedirs(path)
            except OSError:
                print("Error making directory %s"%path)
    
    def save_view(self,vw,subject,name,is_overwrite=False):
        """Set the view for an open webshow instance from a saved view

        Sets the view in a currently-open cortex.webshow instance (with handle `vw`)
        to the saved view named `name`

        Parameters
        ----------
        vw : handle for pycortex webgl viewer
            Handle for open webgl session (returned by cortex.webshow)
        subject : string
            pycortex subject id
        name : string
            Name of stored view to re-load
        
        Notes
        -----
        Equivalent to call to vw.save_view(subject,name)
        For a list of the view parameters saved, see viewer._capture_view
        """
        view = vw._capture_view()
        sName = os.path.join(self.filestore, subject, "views", name+'.json')
        if os.path.exists(sName):
            if not is_overwrite:
                raise IOError('Refusing to over-write extant view If you want to do this, set is_overwrite=True!')
        with open(sName,'w') as fp:
            json.dump(view, fp)

    def get_view(self,vw,subject,name):
        """Set the view for an open webshow instance from a saved view

        Sets the view in a currently-open cortex.webshow instance (with handle `vw`)
        to the saved view named `name`

        Parameters
        ----------
        vw : handle for cortex.webshow
            Handle for open webshow session (returned by cortex.webshow)
        subject : string, subject name
        name : string
            Name of stored view to re-load

        Notes
        -----
        Equivalent to call to vw.get_view(subject,name)
        For a list of the view parameters saved, see viewer._capture_view
        """
        sName = os.path.join(self.filestore, subject, "views", name+'.json')
        with open(sName) as fp:
            view = json.load(fp)
        vw._set_view(**view)

    def get_mnixfm(self, subject, xfm, template=None):
        """Get transform from the space specified by `xfm` to MNI space.

        Parameters
        ----------
        subject : str
            Subject identifier
        xfm : str
            Name of functional space transform. Can be 'identity' for anat space.
        template : str or None, optional
            Path to MNI template volume. If None, uses default specified in cortex.mni

        Returns
        -------
        mnixfm : numpy.ndarray
            Transformation matrix from the space specified by `xfm` to MNI space.

        Notes
        -----
        Equivalent to cortex.mni.compute_mni_transform, but this function also caches
        the result (which is nice because computing it can be slow).

        See Also
        --------
        cortex.mni.compute_mni_transform
        cortex.mni.transform_to_mni
        cortex.mni.transform_mni_to_subject
        """
        from . import mni

        if template is None:
            templatehash = "default"
        else:
            templatehash = sha1(template).hexdigest()

        # Check cache first
        mnixfmfile = os.path.join(self.get_cache(subject), "mni_xfm-%s-%s.txt"%(xfm, templatehash))
        if os.path.exists(mnixfmfile):
            mnixfm = np.loadtxt(mnixfmfile)
        else:
            # Run the transform
            if template is None:
                mnixfm = mni.compute_mni_transform(subject, xfm)
            else:
                mnixfm = mni.compute_mni_transform(subject, xfm, template)

            # Cache the result
            mni._save_fsl_xfm(mnixfmfile, mnixfm)

        return mnixfm

db = Database()
