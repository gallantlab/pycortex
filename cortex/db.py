"""
Surface database functions
==========================

This module creates a singleton object surfs_ which allows easy access to surface files in the filestore.

.. _surfs: :class:`Database`
"""
import os
import re
import glob
import time
import json
import shutil
import numpy as np

from . import options

filestore = options.config.get('basic', 'filestore')

class SubjectDB(object):
    def __init__(self, subj):
        self.subject = subj
        self._transforms = None
        self._surfaces = None

    @property
    def transforms(self):
        if self._transforms is not None:
            return self._transforms
        self._transforms = XfmDB(self.subject)
        return self._transforms

    @property
    def surfaces(self):
        if self._surfaces is not None:
            return self._surfaces
        self._surfaces = SurfaceDB(self.subject)
        return self._surfaces

class SurfaceDB(object):
    def __init__(self, subj):
        self.subject = subj
        self.types = {}
        for name in surfs.getFiles(subj)['surfs'].keys():
            self.types[name] = Surf(subj, name)
                
    def __repr__(self):
        return "Surfaces: [{surfs}]".format(surfs=', '.join(list(self.types.keys())))
    
    def __dir__(self):
        return list(self.types.keys())

    def __getattr__(self, attr):
        if attr in self.types:
            return self.types[attr]
        raise AttributeError(attr)

class Surf(object):
    def __init__(self, subject, surftype):
        self.subject, self.surftype = subject, surftype

    def get(self, hemisphere="both"):
        return surfs.getSurf(self.subject, self.surftype, hemisphere)
    
    def show(self, hemisphere="both"):
        from mayavi import mlab
        pts, polys = surfs.getSurf(self.subject, self.surftype, hemisphere, merge=True, nudge=True)
        return mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)

class XfmDB(object):
    def __init__(self, subj):
        self.subject = subj
        self.xfms = surfs.getFiles(subj)['xfms']

    def __getitem__(self, name):
        if name in self.xfms:
            return XfmSet(self.subject, name)
        raise AttributeError
    
    def __repr__(self):
        return "Transforms: [{xfms}]".format(xfms=",".join(self.xfms))

class XfmSet(object):
    def __init__(self, subj, name):
        self.subject = subj
        self.name = name
        jspath = os.path.join(filestore, subj, 'transforms', name, 'matrices.xfm')
        self._jsdat = json.load(open(jspath))
        self.masks = MaskSet(subj, name)
    
    def __getattr__(self, attr):
        if attr in self._jsdat:
            return surfs.getXfm(self.subject, self.name, attr)
        raise AttributeError
    
    def __repr__(self):
        return "Types: {types}".format(types=", ".join(self._jsdat.keys()))

class MaskSet(object):
    def __init__(self, subj, name):
        self.subject = subj
        self.xfmname = name
        maskpath = surfs.getFiles(subj)['masks'].format(xfmname=name, type='*')
        self._masks = dict((os.path.split(path)[1][5:-7], path) for path in glob.glob(maskpath))

    def __getitem__(self, item):
        import nibabel
        return nibabel.load(self._masks[item]).get_data().T

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
    def __init__(self):
        subjs = os.listdir(os.path.join(filestore))
        self.subjects = dict([(sname, SubjectDB(sname)) for sname in subjs])
        self.auxfile = None
    
    def __repr__(self):
        subjs = ", ".join(sorted(self.subjects.keys()))
        return """Pycortex database\n    Subjects:  {subjs}""".format(subjs=subjs)
    
    def __getattr__(self, attr):
        if attr in self.subjects:
            return self.subjects[attr]
        else:
            raise AttributeError
    
    def __dir__(self):
        return ["loadXfm","getXfm", "getSurf"] + list(self.subjects.keys())

    def getAnat(self, subject, type='raw', recache=False, **kwargs):
        types = dict(
            raw='nii.gz',
            brainmask='nii.gz',
            whitematter='nii.gz',
            curvature='npz',
            flatmask='npz',
            thickness='npz',
            distortion='npz',
            voxelize="nii.gz",
        )
        
        opts = ""
        if len(kwargs) > 0:
            opts = "[%s]"%','.join(["%s=%s"%i for i in kwargs.items()])
        anatform = self.getFiles(subject)['anats']
        anatfile = anatform.format(type=type, opts=opts, ext=types[type])

        if not os.path.exists(anatfile) or recache:
            print("Generating %s anatomical..."%type)
            from . import anat
            getattr(anat, type)(anatfile, subject, **kwargs)

        if types[type] == 'nii.gz':
            import nibabel
            return nibabel.load(anatfile)
        elif types[type] == 'npz':
            return np.load(anatfile)
        return anatfile

    def getOverlay(self, subject, type='rois', **kwargs):
        if type == "rois":
            from . import svgroi
            pts, polys = self.getSurf(subject, "flat", merge=True, nudge=True)
            svgfile = self.getFiles(subject)['rois']
            if self.auxfile is not None:
                try:
                    tf = self.auxfile.getOverlay(subject, type)
                    svgfile = tf.name
                except (AttributeError, IOError):
                    pass
                    
            if 'pts' in kwargs:
                pts = kwargs['pts']
                del kwargs['pts']
            return svgroi.get_roipack(svgfile, pts, polys, **kwargs)

        raise TypeError('Invalid overlay type')
    
    def loadXfm(self, subject, name, xfm, xfmtype="magnet", reference=None):
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
            Type of the provided transform, either magnet space or coord space. Defaults to magnet.
        reference : str, optional
            The nibabel-compatible reference image associated with this transform. Required if name not in database
        """
        if xfmtype not in ["magnet", "coord"]:
            raise TypeError("Unknown transform type")

        import nibabel

        path = os.path.join(filestore, subject, "transforms", name)
        fname = os.path.join(path, "matrices.xfm")
        if os.path.exists(fname):
            jsdict = json.load(open(fname))
        else:
            os.mkdir(path)
            if reference is None:
                raise ValueError("Please specify a reference")
            fpath = os.path.join(path, "reference.nii.gz")
            nib = nibabel.load(reference)
            data = nib.get_data()
            if len(data.shape) > 3:
                import warnings
                warnings.warn('You are importing a 4D dataset, automatically selecting the first volume as reference')
                data = data[...,0]
            out = nibabel.Nifti1Image(data, nib.get_affine(), header=nib.get_header())
            nibabel.save(out, fpath)

            jsdict = dict()

        nib = nibabel.load(os.path.join(path, "reference.nii.gz"))
        if xfmtype == "magnet":
            jsdict['magnet'] = xfm.tolist()
            jsdict['coord'] = np.dot(np.linalg.inv(nib.get_affine()), xfm).tolist()
        elif xfmtype == "coord":
            jsdict['coord'] = xfm.tolist()
            jsdict['magnet'] = np.dot(nib.get_affine(), xfm).tolist()
        
        files = self.getFiles(subject)
        if len(glob.glob(files['masks'].format(xfmname=name, type="*"))) > 0:
            raise ValueError('Refusing to change a transfrom with masks')
            
        json.dump(jsdict, open(fname, "w"), sort_keys=True, indent=4)
    
    def getXfm(self, subject, name, xfmtype="coord"):
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
                return self.auxfile.getXfm(subject, name)
            except (AttributeError, IOError):
                pass

        if name == "identity":
            import nibabel
            nib = self.getAnat(subject, 'raw')
            return Transform(np.linalg.inv(nib.get_affine()), nib)

        fname = os.path.join(filestore, subject, "transforms", name, "matrices.xfm")
        reference = os.path.join(filestore, subject, "transforms", name, "reference.nii.gz")
        xfmdict = json.load(open(fname))
        return Transform(xfmdict[xfmtype], reference)

    def getSurf(self, subject, type, hemisphere="both", merge=False, nudge=False):
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
            return self.auxfile.getSurf(subject, type, hemisphere, merge=merge, nudge=nudge)
        except (AttributeError, IOError):
            pass

        files = self.getFiles(subject)['surfs']

        if hemisphere.lower() == "both":
            left, right = [ self.getSurf(subject, type, hemisphere=h) for h in ["lh", "rh"]]
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
            wpts, polys = self.getSurf(subject, 'wm', hemi)
            ppts, _     = self.getSurf(subject, 'pia', hemi)
            return (wpts + ppts) / 2, polys

        try:
            from . import formats
            return formats.read(os.path.splitext(files[type][hemi])[0])
        except KeyError:
            raise IOError

    def loadMask(self, subject, xfmname, type, mask):
        fname = self.getFiles(subject)['masks'].format(xfmname=xfmname, type=type)
        if os.path.exists(fname):
            raise IOError('Refusing to overwrite existing mask')

        import nibabel
        xfm = self.getXfm(subject, xfmname)
        if xfm.shape != mask.shape:
            raise ValueError("Invalid mask shape: must match shape of reference image")
        affine = xfm.reference.get_affine()
        nib = nibabel.Nifti1Image(mask.astype(np.uint8).T, affine)
        nib.to_filename(fname)

    def getMask(self, subject, xfmname, type='thick'):
        try:
            self.auxfile.getMask(subject, xfmname, type)
        except (AttributeError, IOError):
            pass

        fname = self.getFiles(subject)['masks'].format(xfmname=xfmname, type=type)
        try:
            import nibabel
            nib = nibabel.load(fname)
            return nib.get_data().T != 0
        except IOError:
            print('Mask not found, generating...')
            from .utils import get_cortical_mask
            mask = get_cortical_mask(subject, xfmname, type)
            self.loadMask(subject, xfmname, type, mask)
            return mask

    def getCoords(self, subject, xfmname, hemisphere="both", magnet=None):
        """Calculate the coordinates of each vertex in the epi space by transforming the fiducial to the coordinate space

        Parameters
        ----------
        subject : str
            Name of the subject
        name : str
            Name of the transform
        hemisphere : str, optional
            Which hemisphere to return. If "both", return concatenated. Defaults to "both".
        """
        import warnings
        warnings.warn('Please use a Mapper object instead', DeprecationWarning)

        if magnet is None:
            xfm = self.getXfm(subject, xfmname, xfmtype="coord")
        else:
            xfm = self.getXfm(subject, xfmname, xfmtype="magnet")
            xfm = np.linalg.inv(magnet) * xfm

        coords = []
        vtkTmp = self.getSurf(subject, "fiducial", hemisphere=hemisphere, nudge=False)
        if not isinstance(vtkTmp,(tuple,list)):
            vtkTmp = [vtkTmp]
        for pts, polys, norms in vtkTmp:
            wpts = np.vstack([pts.T, np.ones(len(pts))])
            coords.append(np.dot(xfm, wpts)[:3].round().astype(int).T)

        return coords

    def getFiles(self, subject):
        """Get a dictionary with a list of all candidate filenames for associated data, such as roi overlays, flatmap caches, and ctm caches.
        """
        surfparse = re.compile(r'(.*)/([\w-]+)_([\w-]+)_(\w+).*')
        surfpath = os.path.join(filestore, subject, "surfaces")

        ctmcache = "%s_[{types}]_{method}_{level}.json"%subject
        projcache = "{xfmname}_{projection}.npz"

        surfs = dict()
        for surf in os.listdir(surfpath):
            ssurf = os.path.splitext(surf)[0].split('_')
            name = '_'.join(ssurf[:-1])
            hemi = ssurf[-1]

            if name not in surfs:
                surfs[name] = dict()
            surfs[name][hemi] = os.path.abspath(os.path.join(surfpath,surf))

        filenames = dict(
            surfs=surfs,
            xfms=os.listdir(os.path.join(filestore, subject, "transforms")),
            xfmdir=os.path.join(filestore, subject, "transforms", "{xfmname}", "matrices.xfm"),
            anats=os.path.join(filestore, subject, "anatomicals", '{type}{opts}.{ext}'), 
            masks=os.path.join(filestore, subject, 'transforms', '{xfmname}', 'mask_{type}.nii.gz'),
            cachedir=os.path.join(filestore, subject, "cache"),
            ctmcache=os.path.join(filestore, subject, "cache", ctmcache),
            projcache=os.path.join(filestore, subject, "cache", projcache),
            rois=os.path.join(filestore, subject, "rois.svg").format(subj=subject),
        )

        return filenames

surfs = Database()
