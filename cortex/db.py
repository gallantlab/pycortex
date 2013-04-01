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

from . import xfm
from . import options

filestore = options.config.get('basic', 'filestore')

class AnatDB(object):
    def __init__(self, subj):
        self.subj = subj

class SubjectDB(object):
    def __init__(self, subj):
        self.transforms = XfmDB(subj)
        self.surfaces = SurfaceDB(subj)

class SurfaceDB(object):
    def __init__(self, subj):
        self.subject = subj
        self.types = {}
        pname = os.path.join(filestore, "surfaces", "{subj}_*.*").format(subj=subj)
        for fname in glob.glob(pname):
            fname = os.path.splitext(os.path.split(fname)[1])[0].split('_') 
            subj = fname.pop(0)
            hemi = fname.pop()
            name = "_".join(fname)
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
        self.fname = os.path.join(filestore, "surfaces", "{subj}_{name}_{hemi}.*")

    def get(self, hemisphere="both"):
        return surfs.getSurf(self.subject, self.surftype, hemisphere)
    
    def show(self, hemisphere="both"):
        from mayavi import mlab
        pts, polys = surfs.getSurf(self.subject, self.surftype, hemisphere, merge=True, nudge=True)
        return mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys)

class XfmDB(object):
    def __init__(self, subj):
        self.subj = subj
        xfms = glob.glob(os.path.join(filestore, "transforms", "{subj}_*.xfm".format(subj=subj)))
        self.xfms = ['_'.join(os.path.splitext(os.path.split(x)[1])[0].split('_')[1:]) for x in xfms]

    def __getitem__(self, name):
        if name in self.xfms:
            return XfmSet(self.subj, name)
        raise AttributeError
    
    def __repr__(self):
        return "Transforms: [{xfms}]".format(xfms=",".join(self.xfms))

class XfmSet(object):
    def __init__(self, subj, name):
        self.subject = subj
        self.name = name
        fname = "{subj}_{name}.xfm".format(subj=subj, name=name)
        self.jsdat = json.load(open(os.path.join(filestore, "transforms", fname)))
        self.reffile = os.path.join(filestore, "references", self.jsdat['epifile'])
    
    def get_ref(self):
        import nibabel
        return nibabel.load(self.reffile).get_data()
    
    def __getattr__(self, attr):
        if attr in self.jsdat:
            return np.array(self.jsdat[attr])
        raise AttributeError
    
    def __repr__(self):
        names = set(self.jsdat.keys())
        names -= set(["epifile", "subject"])
        return "Types: {types}".format(types=", ".join(names))


class Database(object):
    """
    Database()

    Surface database

    Attributes
    ----------
    This database object dynamically generates handles to all subjects within the filestore.
    """
    def __init__(self):
        surfs = glob.glob(os.path.join(filestore, "surfaces", "*.*"))
        subjs = set([os.path.split(surf)[1].split('_')[0] for surf in surfs])
        xfms = glob.glob(os.path.join(filestore, "transforms", "*.xfm"))

        self.subjects = dict([(sname, SubjectDB(sname)) for sname in subjs])
        self.xfms = [os.path.splitext(os.path.split(xfm)[1])[0].split('_') for xfm in xfms]
        self.xfms = [(s[0], '_'.join(s[1:])) for s in self.xfms]
    
    def __repr__(self):
        subjs = ", ".join(sorted(self.subjects.keys()))
        xfms = "[%s]"%", ".join('(%s, %s)'% p for p in set(self.xfms))
        return """Flatmapping database
        Subjects:   {subjs}
        Transforms: {xfms}""".format(subjs=subjs, xfms=xfms)
    
    def __getattr__(self, attr):
        if attr in self.subjects:
            return self.subjects[attr]
        else:
            raise AttributeError
    
    def __dir__(self):
        return ["loadXfm","getXfm", "getSurf"] + list(self.subjects.keys())

    def loadAnat(self, subject, anatfile, type='raw', process=True):
        fname = os.path.join(filestore, "anatomicals", "{subj}_{type}.nii.gz").format(subj=subject, type=type)
        import nibabel
        data = nibabel.load(anatfile)
        nibabel.save(data, fname)
        if type == "raw" and process:
            from . import anat
            anat.whitematter(subject)

    def getAnat(self, subject, type='raw', recache=False, **kwargs):
        assert type in ('raw', 'brainmask', 'whitematter', 'curvature', 'fiducial')
        anatform = self.getFiles(subject)['anats']
        anatfile = anatform.format(type=type)
        if type == "curvature":
            path, ext = os.path.splitext(anatform.format(type=type))
            anatfile = "%s.npz"%path
            
        if not os.path.exists(anatfile) or recache:
            print("%s anatomical not found, generating..."%type)
            from . import anat
            getattr(anat, type)(subject, **kwargs)
            
        return anatfile
    
    def loadXfm(self, subject, name, xfm, xfmtype="magnet", epifile=None):
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
        epifile : str, optional
            The nibabel-compatible reference image associated with this transform. Required if name not in database
        """
        assert xfmtype in ["magnet", "coord"], "Unknown transform type"
        fname = os.path.join(filestore, "transforms", "{subj}_{name}.xfm".format(subj=subject, name=name))
        if os.path.exists(fname):
            jsdict = json.load(open(fname))
        else:
            if epifile is None:
                raise ValueError("Please specify a reference epi")
            import nibabel
            outname = "{subj}_{name}_refepi.nii.gz".format(subj=subject, name=name)
            fpath = os.path.join(filestore, "references", outname)
            nib = nibabel.load(epifile)
            nibabel.save(nib, fpath)

            jsdict = dict(epifile=outname, subject=subject)

        import nibabel
        nib = nibabel.load(os.path.join(filestore, "references", jsdict['epifile']))
        if xfmtype == "magnet":
            jsdict['magnet'] = xfm.tolist()
            aff = np.linalg.inv(nib.get_affine())
            jsdict['coord'] = np.dot(aff, xfm).tolist()
        elif xfmtype == "coord":
            jsdict['coord'] = xfm.tolist()
            jsdict['magnet'] = np.dot(nib.get_affine(), xfm).tolist()
        
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
        if name == "identity":
            import nibabel
            nib = nibabel.load(self.getAnat(subject, 'raw'))
            return xfm.Transform(np.linalg.inv(nib.get_affine()), nib)

        fname = os.path.join(filestore, "transforms", "{subj}_{name}.xfm".format(subj=subject, name=name))
        xfmdict = json.load(open(fname))
        if xfmdict['subject'] != subject:
            raise ValueError("Incorrect subject for the name")
        epifile = os.path.join(filestore, "references", xfmdict['epifile'])
        return xfm.Transform(xfmdict[xfmtype], epifile)

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

        import formats
        fname = os.path.join(filestore, "surfaces", "{subj}_{type}_{hemi}")

        if hemisphere == "both":
            left, right = [ self.getSurf(subject, type, hemisphere=h) for h in ["lh", "rh"]]
            if type != "fiducial" and nudge:
                left[0][:,0] -= left[0].max(0)[0]
                right[0][:,0] -= right[0].min(0)[0]
            
            if merge:
                pts   = np.vstack([left[0], right[0]])
                polys = np.vstack([left[1], right[1]+len(left[0])])
                return pts, polys
            else:
                return left, right
        else:
            if hemisphere.lower() in ("lh", "left"):
                hemi = "lh"
            elif hemisphere.lower() in ("rh", "right"):
                hemi = "rh"
            else:
                raise TypeError("Not a valid hemisphere name")
            
            if type == 'fiducial':
                try:
                    wpts, polys = self.getSurf(subject, 'wm', hemi)
                    ppts, _     = self.getSurf(subject, 'pia', hemi)
                    return (wpts + ppts) / 2, polys
                except IOError:
                    pass

            return formats.read(fname.format(subj=subject, type=type, hemi=hemi))

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
        surffiles = os.path.join(filestore, "surfaces", "{subj}_*.*").format(subj=subject)
        anatfiles = '%s_{type}.nii.gz'%subject
        xfms = "%s_{xfmname}.xfm"%subject
        ctmcache = "%s_{xfmname}_[{types}]_{method}_{level}.json"%subject
        flatcache = "%s_{xfmname}_{height}_{date}_v2.pkl"%subject
        projcache = "%s_{xfmname}_{projection}.npz"%subject

        surfs = dict()
        for surf in glob.glob(surffiles):
            path, subj, stype, hemi = surfparse.match(surf).groups()
            if stype not in surfs:
                surfs[stype] = dict()
            surfs[stype][hemi] = os.path.abspath(surf)

        filenames = dict(
            surfs=surfs,
            anats=os.path.join(filestore, "anatomicals", anatfiles), 
            xfms=os.path.join(filestore, "transforms", xfms),
            ctmcache=os.path.join(filestore, "ctmcache", ctmcache),
            flatcache=os.path.join(filestore, "flatcache", flatcache),
            projcache=os.path.join(filestore, "projcache", projcache),
            rois=os.path.join(filestore, "overlays", "{subj}_rois.svg").format(subj=subject),
        )

        return filenames


surfs = Database()
